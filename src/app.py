
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
    user_has_credentials, set_user_credentials, verify_password, DB_PATH
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

# Models global state
lstm_full = None
feature_extractor = None
xgb_models = {}
mc_predictor = None
quantile_predictor = None
conformal_predictor = None
active_targets = []
models_loaded = False
models_lock = threading.Lock()

def load_models_lazy():
    global lstm_full, feature_extractor, xgb_models, mc_predictor, quantile_predictor, conformal_predictor, active_targets, models_loaded
    with models_lock:
        if models_loaded:
            return
        
        print("Loading scientifically improved models (lazy)...")
        # Base DIRs
        BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        MODELS_DIR = os.path.join(BASE_DIR, 'models')
        
        # Load LSTM
        try:
            lstm_full = load_model(os.path.join(MODELS_DIR, "lstm_hybrid_chain.h5"), compile=False)
            feature_extractor = Model(inputs=lstm_full.input, outputs=lstm_full.get_layer('lstm_embeddings').output)
            print("LSTM Feature Extractor loaded.")
        except Exception as e:
            print(f"Error loading LSTM: {e}")
        
        # Load XGB Models
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
        
        try:
            quantile_predictor = QuantileRegressor()
            print("Quantile Predictor Initialized.")
        except Exception as e:
            print(f"Error initializing Quantile Predictor: {e}")
        
        try:
            conformal_predictor = ConformalPredictor()
            print("Conformal Predictor Initialized.")
        except Exception as e:
            print(f"Error initializing Conformal Predictor: {e}")
            
        # Final Verification
        if feature_extractor is None:
            print("[CRITICAL] feature_extractor failed to initialize!")
        if not xgb_models:
            print("[CRITICAL] No XGB models were loaded!")
        if mc_predictor is None:
            print("[CRITICAL] mc_predictor failed to initialize!")
            
        models_loaded = True
        print(f"Model initialization complete. Success status: {not (feature_extractor is None)}")

# 1. Preprocessor fit (needs global data at startup for scalers)
print("Initializing Preprocessor...")
preprocessor = DataPreprocessor(sequence_length=SEQ_LENGTH)
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, 'data')
df_weather = pd.read_csv(os.path.join(DATA_DIR, "final_weather_dataset_2010-2025.csv"))
df_combined = pd.read_csv(os.path.join(DATA_DIR, "final_dataset.csv"))
df_weather_log = preprocessor.apply_log_transform(df_weather)
df_combined_log = preprocessor.apply_log_transform(df_combined)
preprocessor.fit_scalers(df_weather_log, df_combined_log)


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
    r_hist = requests.get(url_hist, timeout=15).json()
    
    # 2. Future Forecast (7 Days)
    # Forecast API starts from Today.
    url_fore = f"https://api.open-meteo.com/v1/forecast?latitude={LAT}&longitude={LON}&daily=temperature_2m_max,temperature_2m_min,temperature_2m_mean,rain_sum,wind_speed_10m_max,wind_direction_10m_dominant,shortwave_radiation_sum,surface_pressure_mean,relative_humidity_2m_mean,cloud_cover_mean&forecast_days=8&timezone=auto"
    r_fore = requests.get(url_fore, timeout=15).json()
    
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
    
    conn = sqlite3.connect(DB_PATH)
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
    
    conn = sqlite3.connect(DB_PATH)
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

@app.route('/continue-guest', methods=['GET'])
def continue_guest():
    session['guest_access'] = True
    session.pop('user_email', None)
    return redirect(url_for('index'))

@app.route('/logout', methods=['GET'])
def logout():
    session.pop('user_email', None)
    session.pop('guest_access', None)
    return redirect(url_for('login_page'))

@app.route('/')
@login_required
def index():
    return render_template('index.html')

@app.route('/predict', methods=['GET'])
def predict():
    try:
        # Lazy load models if not already loaded
        load_models_lazy()
        
        global lstm_full, feature_extractor, xgb_models, mc_predictor, quantile_predictor, conformal_predictor, active_targets, preprocessor
        
        method = request.args.get('method', 'mc_dropout')
        
        # 1. Caching Check (Method-specific)
        cache_key = f'forecast_{method}'
        with forecast_cache['lock']:
            now = time.time()
            if forecast_cache.get(cache_key) and (now - forecast_cache.get(f'{cache_key}_time', 0) < CACHE_DURATION):
                print(f"Serving {method} from cache.")
                return jsonify(forecast_cache[cache_key])

        # 2. Data Fetching & Preparation
        hist_json, fore_json = fetch_weather_data()
        df_hist = parse_meteo(hist_json)
        df_fore = parse_meteo(fore_json)
        
        if df_hist is None or len(df_hist) == 0:
            return jsonify({'error': 'Failed to fetch historical data'}), 500

        # Continuous timeline logic
        df_hist['date'] = pd.to_datetime(df_hist['date'])
        df_fore['date'] = pd.to_datetime(df_fore['date'])
        df_combined_full = pd.concat([df_hist, df_fore], ignore_index=True)
        df_combined_full = df_combined_full.drop_duplicates(subset=['date']).sort_values('date').reset_index(drop=True)
        df_combined_full = df_combined_full.set_index('date').resample('D').asfreq().reset_index()
        
        for col in preprocessor.weather_features:
            if col != 'season':
                df_combined_full[col] = df_combined_full[col].interpolate(method='linear').ffill().bfill()
        
        df_fore_future = df_fore[df_fore['date'] >= pd.to_datetime(datetime.now().date())].reset_index(drop=True)
        df_full = df_combined_full.copy()
        df_full['date_temp'] = pd.to_datetime(df_full['date'])
        df_full['season'] = df_full['date_temp'].dt.month.apply(get_season)
        df_full = df_full.drop('date_temp', axis=1)
        
        for col in preprocessor.weather_features:
            df_full[col] = df_full[col].fillna(0)

        # 3. Batch Preparation for Vectorized Inference
        forecast_days = min(7, len(df_fore_future))
        X_lstm_batch = []
        base_feat_list = []
        target_dates = []
        
        for i in range(forecast_days):
            target_date = df_fore_future.iloc[i]['date']
            target_dates.append(target_date)
            
            # LSTM Window
            window = df_full.iloc[i : i + SEQ_LENGTH]
            X_data = window[preprocessor.lstm_features].values
            X_scaled = preprocessor.scaler_lstm.transform(X_data)
            X_lstm_batch.append(X_scaled)
            
            # Static Features for XGBoost
            target_row = df_full.iloc[i + SEQ_LENGTH]
            feat_dict = {col: float(target_row[col]) for col in preprocessor.lstm_features}
            d = pd.to_datetime(target_date)
            feat_dict.update({
                'month': d.month,
                'day_of_week': d.dayofweek,
                'day': d.day,
                'is_weekend': 1 if d.weekday() in (5, 6) else 0,
                'wind_dir_sin': float(np.sin(np.deg2rad(float(target_row['wind_direction'])))),
                'wind_dir_cos': float(np.cos(np.deg2rad(float(target_row['wind_direction'])))),
                'pressure_delta': float(target_row['pressure']) - float(df_full.iloc[i + SEQ_LENGTH - 1]['pressure'])
            })
            base_feat_list.append(feat_dict)

        X_lstm_batch = np.array(X_lstm_batch) # (7, 14, 10)

        # 4. Vectorized Inference
        forecasts = []
        
        if method == 'mc_dropout' and mc_predictor:
            # Entire 7-day forecast in ONE vectorized call
            mc_batch_results = mc_predictor.predict_with_uncertainty(X_lstm_batch, base_feat_list)
            
            # For non-PM variables, we still need standard predictions (could be vectorized too, but they're fast)
            # Embedding extraction for standard path (one pass for all 7 days)
            embeddings_batch = feature_extractor.predict(X_lstm_batch, verbose=0)
            
            for i in range(forecast_days):
                res = mc_batch_results[i]
                day_res = {'date': target_dates[i]}
                
                # Standard weather predictions
                XGB_FEATURE_NAMES = [f'emb_{j}' for j in range(32)] + \
                                  preprocessor.lstm_features + \
                                  ['month', 'day_of_week', 'day', 'is_weekend', 'wind_dir_sin', 'wind_dir_cos', 'pressure_delta']
                
                # Consolidate feature dict and ensure all values are floats
                feat_dict_std = base_feat_list[i].copy()
                for j in range(32): 
                    feat_dict_std[f'emb_{j}'] = float(embeddings_batch[i][j])
                
                # Create DataFrame with exact column names and float32 type
                X_xgb = pd.DataFrame([feat_dict_std])[XGB_FEATURE_NAMES].astype('float32')
                
                # Non-PM Targets
                for target in active_targets:
                    if target not in ['pm2_5', 'pm10'] and target in xgb_models:
                        day_res[target] = xgb_models[target].predict(X_xgb)[0]
                
                # PM Targets with MC Dropout Uncertainty
                bias = get_bias_correction(target_dates[i])
                for target in ['pm2_5', 'pm10']:
                    if target in res:
                        day_res[target] = res[target]['prediction'] + bias
                        unc = res[target]['uncertainty'].copy()
                        unc['confidence_90'] = [round(x + bias, 2) for x in unc['confidence_90']]
                        unc['confidence_95'] = [round(x + bias, 2) for x in unc['confidence_95']]
                        day_res[f'{target}_uncertainty'] = unc
                
                forecasts.append(day_res)

        else:
            # Fallback for Quantile/Conformal/Standard (Looping standard is still fast)
            embeddings_batch = feature_extractor.predict(X_lstm_batch, verbose=0)
            for i in range(forecast_days):
                day_res = {'date': target_dates[i]}
                bias = get_bias_correction(target_dates[i])
                
                feat_dict_std = base_feat_list[i].copy()
                for j in range(32): 
                    feat_dict_std[f'emb_{j}'] = float(embeddings_batch[i][j])
                
                XGB_FEATURE_NAMES = [f'emb_{j}' for j in range(32)] + \
                                  preprocessor.lstm_features + \
                                  ['month', 'day_of_week', 'day', 'is_weekend', 'wind_dir_sin', 'wind_dir_cos', 'pressure_delta']
                
                X_xgb = pd.DataFrame([feat_dict_std])[XGB_FEATURE_NAMES].astype('float32')
                
                # Standard Forward
                for target in active_targets:
                    if target in xgb_models:
                        val = xgb_models[target].predict(X_xgb)[0]
                        if target in ['pm2_5', 'pm10']:
                            val = np.expm1(val) + bias
                        day_res[target] = max(0, val)
                
                # Add specialized uncertainty if needed
                if method == 'quantile' and quantile_predictor:
                    q_res = quantile_predictor.predict(X_xgb, bias)
                    for target in ['pm2_5', 'pm10']:
                        if target in q_res:
                            day_res[target] = q_res[target]['prediction']
                            day_res[f'{target}_uncertainty'] = q_res[target]['uncertainty']
                elif method == 'conformal' and conformal_predictor:
                   for target in ['pm2_5', 'pm10']:
                        val = day_res.get(target)
                        if val is not None:
                            day_res[f'{target}_uncertainty'] = conformal_predictor.predict(val, target)
                
                forecasts.append(day_res)

        # 5. Final Post-processing & Guardrails
        for day in forecasts:
            pm25 = day.get('pm2_5', 0)
            pm10 = day.get('pm10', 0)
            if pm25 > pm10: pm25 = pm10
            if pm25 < 0.25 * pm10: pm25 = 0.25 * pm10
            day['pm2_5'] = round(pm25, 2)
            day['pm10'] = round(pm10, 2)
            for k, v in day.items():
                if isinstance(v, (float, np.float32, np.float64)) and not k.endswith('_uncertainty'):
                    day[k] = round(float(v), 2)

        # 6. Response Construction
        main_pred = forecasts[0]
        pm25 = main_pred['pm2_5']
        aqi_status = "Good"
        aqi_color = "#00e400"
        if pm25 > 30: aqi_status = "Satisfactory"; aqi_color = "#ffff00"
        if pm25 > 60: aqi_status = "Moderate"; aqi_color = "#ff7e00"
        if pm25 > 90: aqi_status = "Poor"; aqi_color = "#ff0000"
        if pm25 > 120: aqi_status = "Very Poor"; aqi_color = "#99004c"
        
        response = {
            'prediction_date': main_pred['date'].strftime('%Y-%m-%d'),
            'data': main_pred,
            'aqi': {'status': aqi_status, 'color': aqi_color, 'recommendations': get_aqi_recommendations(pm25, aqi_status)},
            'forecast': [{**d, 'date': d['date'].strftime('%Y-%m-%d')} for d in forecasts]
        }
        
        with forecast_cache['lock']:
            forecast_cache[cache_key] = response
            forecast_cache[f'{cache_key}_time'] = time.time()
            
        return jsonify(response)
        
    except Exception as e:
        print(f"Error in predict: {e}")
        import traceback
        traceback.print_exc()
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
