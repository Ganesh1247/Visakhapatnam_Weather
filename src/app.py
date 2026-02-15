
# Fix path for imports if running from src
import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import pandas as pd
import requests
import pickle
import sqlite3
from flask import Flask, render_template, jsonify, request, redirect, url_for, session
from tensorflow.keras.models import load_model, Model  # pyright: ignore[reportMissingImports]
from preprocessing import DataPreprocessor
from datetime import datetime, timedelta
from auth import (
    init_db, generate_otp, send_otp_email, login_required,
    user_has_credentials, set_user_credentials, verify_password
)
from backend.uncertainty.mc_dropout import MCDropoutPredictor
from backend.uncertainty.quantile import QuantileRegressor
from backend.uncertainty.conformal import ConformalPredictor
import time
import threading

# Initialize Flask with correct template and static folders
# Since app.py is in src/, templates are in ../templates
app = Flask(__name__, template_folder='../templates', static_folder='../static')
app.secret_key = 'your-secret-key-here-change-in-production'

# Initialize Database
# Database is in ../data/users.db relative to src/
init_db()

# Config
SEQ_LENGTH = 14
LAT = 17.6868
LON = 83.2185

# 1. Init Preprocessor & Models
print("Loading scientifically improved models...")
preprocessor = DataPreprocessor(sequence_length=SEQ_LENGTH)
# Init scaler logic
# Data files are in ../data/ relative to src/
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, 'data')
MODELS_DIR = os.path.join(BASE_DIR, 'models')

df_weather = pd.read_csv(os.path.join(DATA_DIR, "final_weather_dataset_2010-2025.csv"))
df_combined = pd.read_csv(os.path.join(DATA_DIR, "final_dataset.csv"))

df_weather_log = preprocessor.apply_log_transform(df_weather)
df_combined_log = preprocessor.apply_log_transform(df_combined)
preprocessor.fit_scalers(df_weather_log, df_combined_log)

# Load LSTM
try:
    lstm_full = load_model(os.path.join(MODELS_DIR, "lstm_hybrid_chain.h5"), compile=False)
    feature_extractor = Model(inputs=lstm_full.input, outputs=lstm_full.get_layer('lstm_embeddings').output)
    print("LSTM Feature Extractor loaded.")
except Exception as e:
    print(f"Error loading LSTM: {e}")

# Load XGB Models
xgb_models = {}
# Filter ozone out
active_targets = [t for t in preprocessor.target_columns if t != 'ozone']

for target in active_targets:
    try:
        path = os.path.join(MODELS_DIR, f"xgb_chain_{target}.pkl")
        if os.path.exists(path):
            with open(path, "rb") as f:
                xgb_models[target] = pickle.load(f)
    except:
        print(f"Warning: XGB model for {target} not found.")

# Initialize MC Dropout Predictor
# Note: We need the full LSTM model for training=True (Dropout)
# In current code, 'lstm_full' is the full model. 'feature_extractor' is a sub-model.
# If feature_extractor was created by `Model(inputs=lstm_full.input, ...)` it shares layers.
# So calling feature_extractor(x, training=True) should enable dropout in the shared layers.
mc_predictor = None
try:
    if lstm_full and xgb_models:
        mc_predictor = MCDropoutPredictor(
            lstm_model=lstm_full,
            feature_extractor=feature_extractor,
            xgb_models=xgb_models,
            preprocessor=preprocessor,
            n_iter=50
        )
        print("MC Dropout Predictor Initialized.")
except Exception as e:
    print(f"Error initializing MC Predictor: {e}")

quantile_predictor = None
try:
    quantile_predictor = QuantileRegressor()
    print("Quantile Predictor Initialized.")
except Exception as e:
    print(f"Error initializing Quantile Predictor: {e}")

conformal_predictor = None
try:
    conformal_predictor = ConformalPredictor()
    print("Conformal Predictor Initialized.")
except Exception as e:
    print(f"Error initializing Conformal Predictor: {e}")

# Caching
# Simple dictionary: { 'last_updated': timestamp, 'data': response_json }
# Key could be just 'forecast' since we only have one location.
forecast_cache = {
    'last_updated': 0.0,
    'data': None,
    'lock': threading.Lock()
}
CACHE_DURATION = 3600  # 1 Hour
def fetch_weather_data():
    """
    Fetches:
    1. Past 14 days (Archive API)
    2. Future 7 days (Forecast API)
    """
    # 1. Past Data
    # Archive API usually has a 2-day delay.
    today = datetime.now().date()
    end_date = today - timedelta(days=2)
    start_date = end_date - timedelta(days=SEQ_LENGTH + 5) # Pull extra just in case
    
    url_hist = f"https://archive-api.open-meteo.com/v1/archive?latitude={LAT}&longitude={LON}&start_date={start_date}&end_date={end_date}&daily=temperature_2m_max,temperature_2m_min,temperature_2m_mean,rain_sum,wind_speed_10m_max,wind_direction_10m_dominant,shortwave_radiation_sum,surface_pressure_mean,relative_humidity_2m_mean,cloud_cover_mean&timezone=auto"
    r_hist = requests.get(url_hist).json()
    
    # 2. Future Forecast (7 Days)
    # Forecast API starts from Today.
    url_fore = f"https://api.open-meteo.com/v1/forecast?latitude={LAT}&longitude={LON}&daily=temperature_2m_max,temperature_2m_min,temperature_2m_mean,rain_sum,wind_speed_10m_max,wind_direction_10m_dominant,shortwave_radiation_sum,surface_pressure_mean,relative_humidity_2m_mean,cloud_cover_mean&forecast_days=8&timezone=auto"
    r_fore = requests.get(url_fore).json()
    
    return r_hist, r_fore

def parse_meteo(data_json):
    daily = data_json.get('daily', {})
    if not daily: return None
    
    df = pd.DataFrame({
        'date': daily['time'],
        'temp_max': daily['temperature_2m_max'],
        'temp_min': daily['temperature_2m_min'],
        'temp_avg': daily['temperature_2m_mean'],
        'rainfall': daily['rain_sum'],
        'wind_speed': daily['wind_speed_10m_max'],
        'wind_direction': daily['wind_direction_10m_dominant'],
        'solar_radiation': daily['shortwave_radiation_sum'],
        'pressure': daily['surface_pressure_mean'],
        'humidity': daily['relative_humidity_2m_mean'],
        'cloud_cover': daily['cloud_cover_mean']
    })
    return df

def get_bias_correction(date_str):
    """
    Residual-Based Bias Correction.
    Winter (Oct-Feb): +60 offset for PM.
    """
    d = pd.to_datetime(date_str)
    month = d.month
    # Winter months: Oct(10), Nov(11), Dec(12), Jan(1), Feb(2)
    if month in [10, 11, 12, 1, 2]:
        return 60.0
    return 0.0

def get_aqi_recommendations(pm25, status):
    """Return actionable recommendations based on air quality level."""
    if status == "Good":
        return {
            "title": "Air quality is good",
            "summary": "Ideal conditions for outdoor activities.",
            "do": [
                "Enjoy outdoor exercises like jogging, cycling, or walking",
                "Open windows for fresh air ventilation",
                "Great time for gardening or outdoor hobbies",
                "Safe for children and sensitive groups to play outside"
            ],
            "avoid": [],
            "icon": "😊"
        }
    elif status == "Satisfactory":
        return {
            "title": "Air quality is satisfactory",
            "summary": "Generally acceptable. Sensitive people may experience minor effects.",
            "do": [
                "Most people can enjoy normal outdoor activities",
                "Consider shorter outdoor sessions if you have respiratory sensitivity",
                "Keep indoor air fresh with moderate ventilation"
            ],
            "avoid": [
                "Prolonged heavy exertion outdoors if you have asthma or heart conditions"
            ],
            "icon": "👍"
        }
    elif status == "Moderate":
        return {
            "title": "Moderate air quality",
            "summary": "Sensitive individuals should reduce prolonged outdoor exertion.",
            "do": [
                "Limit strenuous outdoor activities",
                "Consider wearing an N95 mask for extended outdoor exposure",
                "Use air purifiers at home if available",
                "Stay hydrated and take breaks if exercising outside"
            ],
            "avoid": [
                "Heavy outdoor exercise",
                "Spending long hours in traffic or congested areas"
            ],
            "icon": "⚠️"
        }
    elif status == "Poor":
        return {
            "title": "Poor air quality",
            "summary": "Everyone may experience health effects. Sensitive groups at greater risk.",
            "do": [
                "Reduce outdoor activities significantly",
                "Wear N95/KN95 masks when going outside",
                "Keep windows and doors closed; use AC or air purifier",
                "Children, elderly, and those with lung/heart conditions should stay indoors"
            ],
            "avoid": [
                "Outdoor exercise and sports",
                "Opening windows for extended periods",
                "Burning candles or incense indoors",
                "Unnecessary travel in high-traffic areas"
            ],
            "icon": "😷"
        }
    else:  # Very Poor
        return {
            "title": "Very poor air quality",
            "summary": "Health alert: everyone may experience serious health effects.",
            "do": [
                "Stay indoors as much as possible",
                "Use air purifiers with HEPA filters",
                "If you must go out, wear N95/KN95 mask properly",
                "Reschedule non-essential outdoor plans",
                "Keep medication (inhalers, etc.) handy if you have respiratory conditions"
            ],
            "avoid": [
                "All outdoor physical activities",
                "Leaving windows open",
                "Outdoor gatherings and events",
                "Venturing out without a proper mask"
            ],
            "icon": "🚨"
        }

# Helper: Get season from month
def get_season(month):
    if month in [1, 2]: return 0  # Winter
    elif month in [3, 4, 5]: return 1  # Summer
    elif month in [6, 7, 8, 9]: return 2  # Monsoon
    elif month in [10, 11, 12]: return 3  # Post-Monsoon
    return 0

# Authentication Routes
@app.route('/login', methods=['GET'])
def login_page():
    return render_template('login.html')

@app.route('/signup', methods=['POST'])
def signup():
    email = request.json.get('email')
    if not email:
        return jsonify({'error': 'Email required'}), 400
    
    otp = generate_otp()
    otp_expiry = datetime.now() + timedelta(minutes=5)
    
    conn = sqlite3.connect('users.db')
    c = conn.cursor()
    try:
        # Update OTP only; don't overwrite username/password
        c.execute('SELECT id FROM users WHERE email = ?', (email,))
        if c.fetchone():
            c.execute('UPDATE users SET otp = ?, otp_expiry = ? WHERE email = ?',
                      (otp, otp_expiry, email))
        else:
            c.execute('INSERT INTO users (email, otp, otp_expiry) VALUES (?, ?, ?)',
                      (email, otp, otp_expiry))
        conn.commit()
        
        # Send OTP
        if send_otp_email(email, otp):
            return jsonify({'success': True, 'message': 'OTP sent to your email'})
        else:
            return jsonify({'error': 'Failed to send OTP'}), 500
    except Exception as e:
        return jsonify({'error': str(e)}), 500
    finally:
        conn.close()

@app.route('/verify', methods=['POST'])
def verify_otp():
    email = request.json.get('email')
    otp = request.json.get('otp')
    
    if not email or not otp:
        return jsonify({'error': 'Email and OTP required'}), 400
    
    conn = sqlite3.connect('users.db')
    c = conn.cursor()
    c.execute('SELECT otp, otp_expiry FROM users WHERE email = ?', (email,))
    result = c.fetchone()
    conn.close()
    
    if not result:
        return jsonify({'error': 'User not found'}), 404
    
    stored_otp, expiry = result
    if datetime.now() > datetime.fromisoformat(expiry):
        return jsonify({'error': 'OTP expired'}), 400
    
    if stored_otp != otp:
        return jsonify({'error': 'Invalid OTP'}), 400
    
    # Returning user with credentials -> login directly
    if user_has_credentials(email):
        session['user_email'] = email
        return jsonify({'success': True, 'message': 'Login successful', 'redirect': True})
    
    # New user -> must set username & password first
    session['pending_email'] = email  # temporary; create session after credentials set
    return jsonify({'success': True, 'needs_setup': True, 'message': 'Set your username and password'})

@app.route('/set-credentials', methods=['POST'])
def set_credentials():
    email = session.get('pending_email')
    if not email:
        return jsonify({'error': 'Session expired. Please start over.'}), 400
    
    username = request.json.get('username', '').strip()
    password = request.json.get('password', '')
    
    if not username or len(username) < 3:
        return jsonify({'error': 'Username must be at least 3 characters'}), 400
    if not password or len(password) < 6:
        return jsonify({'error': 'Password must be at least 6 characters'}), 400
    
    if not set_user_credentials(email, username, password):
        return jsonify({'error': 'Username already taken'}), 400
    
    session.pop('pending_email', None)
    session['user_email'] = email
    return jsonify({'success': True, 'message': 'Account created! Redirecting...', 'redirect': True})

@app.route('/login-password', methods=['POST'])
def login_password():
    username_or_email = request.json.get('username', '').strip()
    password = request.json.get('password', '')
    
    if not username_or_email or not password:
        return jsonify({'error': 'Username/email and password required'}), 400
    
    success, email = verify_password(username_or_email, password)
    if success:
        session['user_email'] = email
        return jsonify({'success': True, 'message': 'Login successful', 'redirect': True})
    return jsonify({'error': 'Invalid username or password'}), 401

@app.route('/logout', methods=['GET'])
def logout():
    session.pop('user_email', None)
    return redirect(url_for('login_page'))

@app.route('/')
@login_required
def index():
    return render_template('index.html')

@app.route('/predict', methods=['GET'])
def predict():
    try:
        method = request.args.get('method', 'mc_dropout')
        
        # Check Cache (only if method is mc_dropout for now, or key it by method)
        cache_key = f'forecast_{method}'
        # We need to change cache structure or just bypass cache for non-default methods for now
        # Or simplistic: clear cache if method changes? No.
        # Let's bypass cache if method is not default, or update cache key logic.
        
        # For this demo, let's keep cache simple: caching only default 'mc_dropout'.
        if method == 'mc_dropout':
             with forecast_cache['lock']:
                now = time.time()
                if forecast_cache['data'] and (now - forecast_cache['last_updated'] < CACHE_DURATION):
                    print("Serving from cache.")
                    return jsonify(forecast_cache['data'])

        # If not cached or expired, compute.
        # Fetch Data
        hist_json, fore_json = fetch_weather_data()
        
        df_hist = parse_meteo(hist_json)
        df_fore = parse_meteo(fore_json)
        
        if df_hist is None or len(df_hist) == 0:
            return jsonify({'error': 'Failed to fetch historical data (Open-Meteo Archive empty)'}), 500

        # Combine Data: [Past Archive] + [Forecast as filler] + [Future 7]
        # Open-Meteo Archive has a 2-day delay.
        # df_hist usually ends 2 days ago.
        # df_fore starts today.
        # missing_days = [Yesterday, Today]
        
        # Ensure 'date' is datetime objects for comparison
        df_hist['date'] = pd.to_datetime(df_hist['date'])
        df_fore['date'] = pd.to_datetime(df_fore['date'])
        
        last_hist_date = df_hist['date'].iloc[-1]
        first_fore_date = df_fore['date'].iloc[0]
        
        # Missing data filler: from forecast API but for the dates between hist and fore
        # Usually this covers 'Yesterday' if history ends 2 days ago.
        df_gap_filler = df_fore[df_fore['date'] < first_fore_date] # Usually empty based on Open-Meteo behavior
        
        # Correct approach:
        # We need a continuous timeline. 
        # Archive might end at T-2. Forecast starts at T. 
        # We use Archive for up to T-2.
        # We use the 'forecast' API for T-1, T, T+1... 
        # Wait, the forecast API usually includes 'today' and maybe 'yesterday' in some configurations.
        # Let's check our diagnostic. Diagnostic showed: 2026-02-09 (Archive) and 2026-02-11 (Forecast).
        # So 2026-02-10 is missing.
        
        # Simple fix: Use the forecast data for everything it provides, and history for the rest.
        df_combined_full = pd.concat([df_hist, df_fore], ignore_index=True)
        df_combined_full = df_combined_full.drop_duplicates(subset=['date']).sort_values('date').reset_index(drop=True)
        
        # Interpolate missing days if any (like Feb 10)
        df_combined_full = df_combined_full.set_index('date').resample('D').asfreq().reset_index()
        
        # Fill missing weather features using interpolation or forward fill
        for col in preprocessor.weather_features:
            if col != 'season':
                df_combined_full[col] = df_combined_full[col].interpolate(method='linear').ffill().bfill()
        
        # Identify the start of "official" forecast (from df_fore)
        # We want to predict for the days in df_fore future
        df_fore_future = df_fore[df_fore['date'] >= pd.to_datetime(datetime.now().date())].reset_index(drop=True)
        
        # We need to ensure df_combined_full has enough history for the FIRST forecast day
        first_forecast_date = df_fore_future.iloc[0]['date']
        hist_before_forecast = df_combined_full[df_combined_full['date'] < first_forecast_date]
        
        if len(hist_before_forecast) < SEQ_LENGTH:
            # If still insufficient, we might need a longer historical pull
            # But let's assume the previous days in combo are enough.
            # If not, seq_length windowing will fail later.
            pass

        df_full = df_combined_full.copy()
        
        # Clean NaNs and Season
        df_full['date_temp'] = pd.to_datetime(df_full['date'])
        df_full['season'] = df_full['date_temp'].dt.month.apply(get_season)
        df_full = df_full.drop('date_temp', axis=1)
        
        for col in preprocessor.weather_features:
            df_full[col] = df_full[col].fillna(0)

        forecasts = []
        forecast_cache_days = min(7, len(df_fore_future)) # Predict for up to 7 days
        
        for i in range(forecast_cache_days):
            target_date = df_fore_future.iloc[i]['date']
            
            # 1. Prepare LSTM Input
            window = df_full.iloc[i : i + SEQ_LENGTH]
            X_data = window[preprocessor.lstm_features].values
            X_scaled = preprocessor.scaler_lstm.transform(X_data)
            X_input = X_scaled.reshape(1, SEQ_LENGTH, len(preprocessor.lstm_features))
            
            # 2. Prepare Base Features for XGBoost
            target_row = df_full.iloc[i + SEQ_LENGTH]
            
            # Base features dict
            feat_dict = {}
            for col in preprocessor.lstm_features:
                feat_dict[col] = float(target_row[col])
                
            d = pd.to_datetime(target_date)
            feat_dict['month'] = d.month
            feat_dict['day_of_week'] = d.dayofweek
            feat_dict['day'] = d.day
            feat_dict['is_weekend'] = 1 if d.weekday() in (5, 6) else 0
            
            if 'wind_direction' in target_row:
                angle_rad = np.deg2rad(float(target_row['wind_direction']))
                feat_dict['wind_dir_sin'] = float(np.sin(angle_rad))
                feat_dict['wind_dir_cos'] = float(np.cos(angle_rad))
            else:
                feat_dict['wind_dir_sin'] = 0.0
                feat_dict['wind_dir_cos'] = 0.0

            if 'pressure' in df_full.columns:
                prev_row = df_full.iloc[i + SEQ_LENGTH - 1]
                try:
                    pressure_delta = float(target_row['pressure']) - float(prev_row['pressure'])
                except:
                    pressure_delta = 0.0
                feat_dict['pressure_delta'] = pressure_delta
            else:
                feat_dict['pressure_delta'] = 0.0
            
            # 3. Uncertainty Inference (Method Selection)
            day_result = {'date': target_date}
            bias = get_bias_correction(target_date)
            
            # Helper to finalize standard prediction
            def run_standard_predict(day_result):
                 # Predict Embeddings
                embeddings = feature_extractor.predict(X_input, verbose=0)
                # ... (Standard XGB Feature Construction) ...
                # Reuse code logic effectively...
                # Note: This block is getting large. For production, refactor into helper function.
                # Inline for now to minimize risk of breaking during refactor.
                
                feat_dict_std = feat_dict.copy()
                for j in range(embeddings.shape[1]):
                    feat_dict_std[f'emb_{j}'] = float(embeddings[0][j])
                
                # Define the exact feature names and order expected by XGBoost models
                XGB_FEATURE_NAMES = [f'emb_{j}' for j in range(32)] + \
                                  preprocessor.lstm_features + \
                                  ['month', 'day_of_week', 'day', 'is_weekend', 'wind_dir_sin', 'wind_dir_cos', 'pressure_delta']

                X_xgb_std = pd.DataFrame([feat_dict_std])[XGB_FEATURE_NAMES].astype('float32')
                
                for target in active_targets:
                    if target in xgb_models:
                        try:
                            if target in preprocessor.pm_targets:
                                val = np.expm1(xgb_models[target].predict(X_xgb_std)[0])
                                val = max(0, val) + bias
                                day_result[target] = val
                                if target == 'pm10': pm10_val = val
                                if target == 'pm2_5': pm25_val = val
                            else:
                                val = xgb_models[target].predict(X_xgb_std)[0]
                                day_result[target] = val
                        except: pass
                
                return day_result, X_xgb_std # Return X_xgb_std for quantile if needed

            # Dispatch
            if method == 'mc_dropout' and mc_predictor:
                # MC Dropout Logic
                mc_out = mc_predictor.predict_with_uncertainty(X_input, feat_dict)
                
                # We still need standard prediction for weather variables not covered by MC
                # So let's run standard first (it's fast) then overwrite PM
                day_result, _ = run_standard_predict(day_result)
                
                for target in ['pm2_5', 'pm10']:
                    if target in mc_out:
                         res = mc_out[target]
                         val = res['prediction'] + bias
                         unc = res['uncertainty'].copy()
                         unc['confidence_90'] = [round(x + bias, 2) for x in unc['confidence_90']]
                         unc['confidence_95'] = [round(x + bias, 2) for x in unc['confidence_95']]
                         day_result[target] = val
                         day_result[f'{target}_uncertainty'] = unc
            
            elif method == 'quantile' and quantile_predictor:
                # Quantile Regression Logic
                 # 1. We need Features + Embeddings (Single Pass)
                day_result, X_xgb_std = run_standard_predict(day_result)
                
                # 2. Get Quantiles
                # We need embeddings in X_xgb. `run_standard_predict` creates it.
                # But we need it OUTSIDE.
                # Refactored `run_standard_predict` to return `X_xgb_std`.
                
                q_out = quantile_predictor.predict(X_xgb_std, bias)
                
                for target in ['pm2_5', 'pm10']:
                    if target in q_out:
                        # Overwrite PM with median? Or keep standard mean?
                        # Quantile median is robust. Let's use it.
                        res = q_out[target]
                        # Bias already applied in predict() of QuantileRegressor
                        
                        day_result[target] = res['prediction']
                        # QuantileRegressor returns 'confidence_90'
                        day_result[f'{target}_uncertainty'] = res['uncertainty']

                        day_result[f'{target}_uncertainty'] = res['uncertainty']

            elif method == 'conformal' and conformal_predictor:
                # Conformal Prediction Logic
                # 1. Standard Prediction
                day_result, _ = run_standard_predict(day_result)
                
                # 2. Apply Conformal Intervals
                for target in ['pm2_5', 'pm10']:
                    val = day_result.get(target)
                    if val is not None:
                        # Conformal Interval is [val - q, val + q]
                        # Bias is already included in 'val' inside run_standard_predict
                        res = conformal_predictor.predict(val, target)
                        if res:
                             day_result[f'{target}_uncertainty'] = res

            else:
                 # Default / Fallback Standard
                 day_result, _ = run_standard_predict(day_result)

            # Guardrails (PM2.5 vs PM10)
            pm25_val = day_result.get('pm2_5', 0)
            pm10_val = day_result.get('pm10', 0)
            
            if pm25_val > pm10_val:
                pm25_val = pm10_val
            if pm25_val < 0.25 * pm10_val:
                pm25_val = 0.25 * pm10_val
                
            day_result['pm2_5'] = round(pm25_val, 2)
            day_result['pm10'] = round(pm10_val, 2)
            
            # Rounding & Conversion for other values
            for k, v in day_result.items():
                if isinstance(v, (float, np.float32, np.float64)) and not k.endswith('_uncertainty'):
                    day_result[k] = round(float(v), 2)
                if isinstance(v, (int, np.int32, np.int64)):
                    day_result[k] = int(v)
            
            forecasts.append(day_result)

        # Response construction
        main_pred = forecasts[0]
        
        # AQI Logic
        pm25 = main_pred['pm2_5']
        aqi_status = "Good"
        aqi_color = "#00e400"
        if pm25 > 30: 
            aqi_status = "Satisfactory"
            aqi_color = "#ffff00"
        if pm25 > 60:
            aqi_status = "Moderate"
            aqi_color = "#ff7e00"
        if pm25 > 90:
            aqi_status = "Poor"
            aqi_color = "#ff0000"
        if pm25 > 120:
            aqi_status = "Very Poor"
            aqi_color = "#99004c"
            
        aqi_recommendations = get_aqi_recommendations(pm25, aqi_status)
        
        response = {
            'prediction_date': main_pred['date'],
            'data': main_pred,
            'aqi': {'status': aqi_status, 'color': aqi_color, 'recommendations': aqi_recommendations},
            'forecast': forecasts
        }
        
        # Update Cache
        with forecast_cache['lock']:
            forecast_cache['data'] = response
            forecast_cache['last_updated'] = time.time()
            
        return jsonify(response)
        
    except Exception as e:
        print(f"Error: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/stats', methods=['GET'])
def get_stats():
    # Return metrics
    try:
        if os.path.exists("metrics_scientific.csv"):
            df = pd.read_csv("metrics_scientific.csv")
            return jsonify(df.to_dict(orient='records'))
        return jsonify([])
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    print("\n" + "="*50)
    print("  EcoGlance - Open in browser: http://127.0.0.1:5000")
    print("="*50 + "\n")
    app.run(host='127.0.0.1', debug=True, port=5000)
