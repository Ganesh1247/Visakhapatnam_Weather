
# Fix path for imports if running from src
import sys
import os
from dotenv import load_dotenv
load_dotenv()

# Suppress TF logs
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import pandas as pd
import requests
import pickle
import sqlite3
import xgboost as xgb
from flask import Flask, render_template, jsonify, request, redirect, url_for, session
from flask.json.provider import DefaultJSONProvider
from tensorflow.keras.models import load_model, Model  # pyright: ignore[reportMissingImports]
from preprocessing import DataPreprocessor
from datetime import datetime, timedelta
from auth import (
    init_db, generate_otp, send_otp_email, login_required,
    user_has_credentials, set_user_credentials, verify_password,
    save_otp, get_otp, DB_PATH
)
from backend.uncertainty.mc_dropout import MCDropoutPredictor
from backend.uncertainty.quantile import QuantileRegressor
from backend.uncertainty.conformal import ConformalPredictor
import time
import threading

# Initialize Flask with correct template and static folders
# Since app.py is in src/, templates are in ../templates
app = Flask(__name__, template_folder='../templates', static_folder='../static')
# Use environment variable for secret key in production
app.secret_key = os.environ.get('SECRET_KEY', 'dev-secret')

# Session cookie settings:
# Hugging Face embeds apps in an iframe on huggingface.co; third-party cookie
# restrictions can break login/session unless SameSite=None and Secure are set.
is_hf_space = bool(os.environ.get("SPACE_ID") or os.environ.get("HF_SPACE_ID"))
if is_hf_space:
    app.config.update(
        SESSION_COOKIE_SAMESITE="None",
        SESSION_COOKIE_SECURE=True,
        SESSION_COOKIE_HTTPONLY=True,
    )

# Flask 3 uses JSON provider classes (app.json_encoder is ignored).
class NumpyJSONProvider(DefaultJSONProvider):
    def default(self, obj):
        if isinstance(obj, (np.integer,)):
            return int(obj)
        if isinstance(obj, (np.floating,)):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return DefaultJSONProvider.default(self, obj)

app.json = NumpyJSONProvider(app)

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
            model_path = os.path.join(MODELS_DIR, "lstm_hybrid_chain.h5")
            if os.path.exists(model_path):
                lstm_full = load_model(model_path, compile=False)
                feature_extractor = Model(inputs=lstm_full.input, outputs=lstm_full.get_layer('lstm_embeddings').output)
                print("LSTM Feature Extractor loaded.")
            else:
                print(f"[CRITICAL] LSTM model file not found at {model_path}")
        except Exception as e:
            print(f"Error loading LSTM: {e}")
        
        # Load XGB Models
        active_targets = [t for t in preprocessor.target_columns if t != 'ozone']
        loaded_xgb_count = 0
        for target in active_targets:
            try:
                path = os.path.join(MODELS_DIR, f"xgb_chain_{target}.pkl")
                if os.path.exists(path):
                    with open(path, "rb") as f:
                        xgb_models[target] = pickle.load(f)
                        loaded_xgb_count += 1
            except Exception as e:
                print(f"Warning: Failed to load XGB model for {target}: {e}")
        
        # Initialize Predictors
        try:
            if feature_extractor and xgb_models:
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
            # Attempt Fallback to Standard XGB Models
            print("Attempting to load Standard Engine (non-chain models)...")
            for target in active_targets:
                try:
                    path = os.path.join(MODELS_DIR, f"xgb_{target}.pkl")
                    if os.path.exists(path):
                        with open(path, "rb") as f:
                            xgb_models[f"std_{target}"] = pickle.load(f)
                except: pass

        # Only set loaded if critical components are present
        if (feature_extractor and xgb_models) or (len([k for k in xgb_models if k.startswith('std_')]) > 0):
            models_loaded = True
            print("Model initialization finished (with fallbacks if needed).")
        else:
            print("Model initialization failed completely.")

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
forecast_cache = {
    'last_updated': 0.0,
    'data': None,
    'lock': threading.Lock()
}
CACHE_DURATION = 3600  # 1 Hour

# Helper: Predict with XGBoost model (handles both Booster and XGBRegressor)
def xgb_predict(model, X_df):
    """Predict using DMatrix to ensure feature names are always preserved."""
    dmat = xgb.DMatrix(X_df.values, feature_names=list(X_df.columns))
    if isinstance(model, xgb.Booster):
        return model.predict(dmat)
    else:
        # XGBRegressor: extract internal Booster and predict directly
        return model.get_booster().predict(dmat)

def fetch_nasa_history(start_date, end_date):
    """
    Fetches historical data from NASA POWER API.
    Returns DataFrame or None if failed.
    """
    try:
        # fmt = YYYYMMDD
        s_str = start_date.strftime('%Y%m%d')
        e_str = end_date.strftime('%Y%m%d')
        
        # Parameters mapping to our needs
        # T2M -> temp_avg, T2M_MAX -> temp_max, T2M_MIN -> temp_min
        # PRECTOTCORR -> rainfall (or PRECTOT)
        # WS10M -> wind_speed, WD10M -> wind_direction
        # PS -> pressure
        # RH2M -> humidity
        # ALLSKY_SFC_SW_DWN -> solar_radiation
        # CLOUD_AMT -> cloud_cover
        params = "T2M,T2M_MAX,T2M_MIN,PRECTOTCORR,WS10M,WD10M,PS,RH2M,ALLSKY_SFC_SW_DWN,CLOUD_AMT"
        
        url = (
            "https://power.larc.nasa.gov/api/temporal/daily/point?"
            f"latitude={LAT}&longitude={LON}"
            f"&start={s_str}&end={e_str}"
            f"&parameters={params}"
            "&community=RE"
            "&format=JSON"
        )
        print(f"Fetching NASA Data: {url}")
        response = requests.get(url, timeout=20)
        data = response.json()
        
        if 'properties' not in data:
            print("NASA Data Error: 'properties' not found")
            return None
            
        records = data['properties']['parameter']
        
        # Convert to DataFrame
        dates = sorted(records['T2M'].keys())
        df = pd.DataFrame({'date': dates})
        df['date'] = pd.to_datetime(df['date'], format='%Y%m%d')
        
        # Map NASA columns to our columns
        df['temp_avg'] = [records['T2M'][d] for d in dates]
        df['temp_max'] = [records['T2M_MAX'][d] for d in dates]
        df['temp_min'] = [records['T2M_MIN'][d] for d in dates]
        df['rainfall'] = [records['PRECTOTCORR'][d] for d in dates]
        df['wind_speed'] = [records['WS10M'][d] for d in dates] 
        df['wind_direction'] = [records['WD10M'][d] for d in dates]
        df['pressure'] = [records['PS'][d] for d in dates] # kPa usually
        df['humidity'] = [records['RH2M'][d] for d in dates]
        df['solar_radiation'] = [records['ALLSKY_SFC_SW_DWN'][d] for d in dates] # kW-hr/m^2/day usually
        df['cloud_cover'] = [records['CLOUD_AMT'][d] for d in dates]
        
        # Unit Conversions
        # Pressure: NASA is kPa, we usually use hPa. 1 kPa = 10 hPa
        df['pressure'] = df['pressure'] * 10.0
        
        # Solar: NASA kW-hr/m^2/day -> W/m^2 (approx avg? or sum?)
        # Open-Meteo gives MJ/m^2 or W/m^2. 
        # 1 kW-hr = 3.6 MJ. 
        # Let's keep it consistent with training. If training was Open-Meteo MJ, we convert.
        # Assuming training scaled 0-1, relative magnitude matters.
        # NASA radiation is often ~3-6. Open-Meteo raw SW radiation sum is often ~15-25 (MJ).
        # 1 kWh = 3.6 MJ. So NASA * 3.6 = MJ.
        df['solar_radiation'] = df['solar_radiation'] * 3.6

        return df
        
    except Exception as e:
        print(f"Failed to fetch NASA data: {e}")
        return None

def fetch_weather_data():
    """
    Fetches:
    1. Past 14 days (Hybrid: NASA preferred + OpenMeteo Recent Fill)
    2. Future 7 days (Forecast API)
    """
    today = datetime.now().date()
    
    # 1. Past Data Strategy
    # We need SEQ_LENGTH (14) days ending yesterday.
    end_date = today - timedelta(days=1)
    start_date = end_date - timedelta(days=SEQ_LENGTH + 5) # Buffer
    
    # Try NASA first
    df_nasa = fetch_nasa_history(start_date, end_date)
    
    # Fetch Open-Meteo Archive as Backup/Gap-Fill
    url_hist = f"https://archive-api.open-meteo.com/v1/archive?latitude={LAT}&longitude={LON}&start_date={start_date}&end_date={end_date}&daily=temperature_2m_max,temperature_2m_min,temperature_2m_mean,rain_sum,wind_speed_10m_max,wind_direction_10m_dominant,shortwave_radiation_sum,surface_pressure_mean,relative_humidity_2m_mean,cloud_cover_mean&timezone=auto"
    try:
        r_hist = requests.get(url_hist, timeout=15).json()
        df_om = parse_meteo(r_hist)
    except:
        df_om = None
        
    # Hybrid Merge
    if df_nasa is not None and not df_nasa.empty:
        # Check for -999 (NASA error value) and replace with NaN
        df_nasa.replace(-999.0, np.nan, inplace=True)
        
        if df_om is not None:
            # Align dates
            df_nasa['date'] = pd.to_datetime(df_nasa['date'])
            df_om['date'] = pd.to_datetime(df_om['date'])
            
            # Use Open-Meteo to fill NaNs in NASA (especially recent days)
            # Merge on date
            df_final_hist = pd.merge(df_nasa, df_om, on='date', how='outer', suffixes=('_nasa', '_om'))
            
            for col in ['temp_max', 'temp_min', 'temp_avg', 'rainfall', 'wind_speed', 'wind_direction', 'pressure', 'humidity', 'solar_radiation', 'cloud_cover']:
                # Prefer NASA, fill with OM
                if f'{col}_nasa' in df_final_hist and f'{col}_om' in df_final_hist:
                    df_final_hist[col] = df_final_hist[f'{col}_nasa'].fillna(df_final_hist[f'{col}_om'])
                elif f'{col}_nasa' in df_final_hist:
                    df_final_hist[col] = df_final_hist[f'{col}_nasa']
                elif f'{col}_om' in df_final_hist:
                     df_final_hist[col] = df_final_hist[f'{col}_om']
            
            # Keep only clean columns
            keep_cols = ['date', 'temp_max', 'temp_min', 'temp_avg', 'rainfall', 'wind_speed', 'wind_direction', 'pressure', 'humidity', 'solar_radiation', 'cloud_cover']
            df_hist_final = df_final_hist[keep_cols].sort_values('date').tail(SEQ_LENGTH+2) # Ensure we have enough
            
            # If NASA had huge gaps, OM might have filled them.
        else:
            df_hist_final = df_nasa
            
        print("Using Hybrid NASA+OpenMeteo History.")
    else:
        # Fallback to pure Open-Meteo
        print("NASA fetch failed, using Open-Meteo only.")
        df_hist_final = df_om

    # 2. Future Forecast (7 Days)
    url_fore = f"https://api.open-meteo.com/v1/forecast?latitude={LAT}&longitude={LON}&daily=temperature_2m_max,temperature_2m_min,temperature_2m_mean,rain_sum,wind_speed_10m_max,wind_direction_10m_dominant,shortwave_radiation_sum,surface_pressure_mean,relative_humidity_2m_mean,cloud_cover_mean&forecast_days=8&timezone=auto"
    r_fore = requests.get(url_fore, timeout=15).json()
    
    # Return DataFrames not JSON to simplify downstream
    return df_hist_final, r_fore

@app.route('/hourly/<date_str>', methods=['GET'])
def get_hourly(date_str):
    try:
        # Fetch hourly data for specific date from Open-Meteo
        # Need start_date and end_date to be the same
        url = f"https://api.open-meteo.com/v1/forecast?latitude={LAT}&longitude={LON}&start_date={date_str}&end_date={date_str}&hourly=temperature_2m,relative_humidity_2m,rain,surface_pressure,cloud_cover,wind_speed_10m&timezone=auto"
        r = requests.get(url, timeout=10).json()
        
        hourly = r.get('hourly', {})
        if not hourly:
            return jsonify({'error': 'No hourly data'}), 404
            
        # Structure for Frontend
        result = []
        for i, time_str in enumerate(hourly['time']):
            # time_str is ISO "2023-10-27T00:00"
            dt = datetime.fromisoformat(time_str)
            result.append({
                'time': dt.strftime('%H:%M'), # "14:00"
                'temp': hourly['temperature_2m'][i],
                'humidity': hourly['relative_humidity_2m'][i],
                'rain': hourly['rain'][i],
                'wind': hourly['wind_speed_10m'][i],
                'condition': 'Rainy' if hourly['rain'][i] > 0.5 else ('Cloudy' if hourly['cloud_cover'][i] > 50 else 'Sunny')
            })
            
        return jsonify(result)
    except Exception as e:
        return jsonify({'error': str(e)}), 500

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
    
    # Use agnostic save_otp (Supabase or SQLite)
    if save_otp(email, otp, otp_expiry):
        # Send OTP
        if send_otp_email(email, otp):
            return jsonify({'success': True, 'message': 'OTP sent to your email'})
        else:
            return jsonify({'error': 'Failed to send OTP'}), 500
    else:
        return jsonify({'error': 'Database Error'}), 500

@app.route('/verify', methods=['POST'])
def verify_otp():
    email = request.json.get('email')
    otp = request.json.get('otp')
    
    if not email or not otp:
        return jsonify({'error': 'Email and OTP required'}), 400
    
    stored_otp, expiry = get_otp(email)
    
    if not stored_otp:
        return jsonify({'error': 'User not found or no OTP requested'}), 404
    
    # Handle both string (SQLite/Supabase) and datetime objects (if adapter returns object)
    if isinstance(expiry, str):
        expiry_dt = datetime.fromisoformat(expiry.replace('Z', '+00:00'))
    else:
        expiry_dt = expiry
        
    # Naive vs Aware check
    now = datetime.now()
    if expiry_dt.tzinfo:
        expiry_dt = expiry_dt.replace(tzinfo=None)

    if now > expiry_dt:
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
        # Now returns (DataFrame, JSON)
        df_hist, fore_json = fetch_weather_data()
        
        # History is already a DataFrame now (from Hybrid logic)
        # Forecast is still JSON
        df_fore = parse_meteo(fore_json)
        
        if df_hist is None or len(df_hist) == 0:
            return jsonify({'error': 'Failed to fetch historical data'}), 500

        # Verify Models are ready
        if feature_extractor is None and not any(k.startswith('std_') for k in xgb_models):
            return jsonify({
                'error': 'AI Intelligence Core failed to initialize.',
                'status': 'off',
                'suggestion': 'Check Render logs for [CRITICAL] messages. Ensure models/ folder is complete.'
            }), 503

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
            
            # Embedding extraction for standard path (one pass for all 7 days)
            if feature_extractor is None:
                raise ValueError("Neural Feature Extractor not initialized")
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
                        day_res[target] = xgb_predict(xgb_models[target], X_xgb)[0]
                
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
            if feature_extractor is None:
                raise ValueError("Neural Feature Extractor not initialized")
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
                        val = xgb_predict(xgb_models[target], X_xgb)[0]
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
                
                # Standard Fallback logic for when Neural Core is missing
                if feature_extractor is None:
                    # Overwrite/fill using standard models if available
                    for target in active_targets:
                        std_key = f"std_{target}"
                        if std_key in xgb_models:
                            # Standard models don't need embeddings
                            FEAT_COLS = preprocessor.lstm_features + ['month', 'day_of_week', 'day', 'is_weekend', 'wind_dir_sin', 'wind_dir_cos', 'pressure_delta']
                            X_std = pd.DataFrame([base_feat_list[i]])[FEAT_COLS].astype('float32')
                            val = xgb_predict(xgb_models[std_key], X_std)[0]
                            if target in ['pm2_5', 'pm10']:
                                val = np.expm1(val) + bias
                            day_res[target] = round(float(max(0, val)), 2)
                            day_res['engine'] = 'Standard (Lighter)'
                
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

from generate_metrics import generate_metrics

@app.route('/stats', methods=['GET'])
def get_stats():
    # Return metrics
    try:
        # Trigger live update of metrics
        generate_metrics()
        
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
    port = int(os.environ.get("PORT", 5000))
    app.run(host='0.0.0.0', debug=True, port=port)
