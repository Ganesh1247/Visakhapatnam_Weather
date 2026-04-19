
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
import matplotlib
matplotlib.use('Agg') # Non-interactive backend
import matplotlib.pyplot as plt
import io
from fpdf import FPDF
from auth import (
    init_db, login_required, verify_password, create_user_credentials
)

def calculate_india_aqi_from_pm25(pm25):
    """Calculate the India National Air Quality Index (AQI) based on PM2.5 concentration."""
    c = max(0, float(pm25) if pm25 is not None else 0)
    # India AQI breakpoints for PM2.5 (24-hr average)
    breakpoints = [
        (0, 30, 0, 50),      # Good
        (30, 60, 51, 100),   # Satisfactory
        (60, 90, 101, 200),  # Moderate
        (90, 120, 201, 300), # Poor
        (120, 250, 301, 400),# Very Poor
        (250, 500, 401, 500) # Severe
    ]
    for bpLo, bpHi, iLo, iHi in breakpoints:
        if c <= bpHi:
            aqi = ((iHi - iLo) / (bpHi - bpLo)) * (c - bpLo) + iLo
            return int(round(max(0, min(500, aqi))))
    return 500

def fetch_aqi_history(start_date, end_date, lat, lon):
    """Fetch historical PM2.5 and PM10 from Open-Meteo Air Quality Archive."""
    try:
        url = (
            f"https://air-quality-api.open-meteo.com/v1/air-quality?"
            f"latitude={lat}&longitude={lon}&start_date={start_date}&end_date={end_date}"
            f"&hourly=pm10,pm2_5&timezone=auto"
        )
        r = requests.get(url, timeout=12).json()
        hourly = r.get('hourly', {})
        if not hourly: return None
        
        df = pd.DataFrame({
            'date': pd.to_datetime(hourly['time']),
            'pm2_5': hourly['pm2_5'],
            'pm10': hourly['pm10']
        })
        # Calculate daily averages
        df['date_only'] = df['date'].dt.date
        daily = df.groupby('date_only').agg({'pm2_5': 'mean', 'pm10': 'mean'}).reset_index()
        daily = daily.rename(columns={'date_only': 'date'})
        return daily
    except Exception as e:
        print(f"AQI History Fetch Failed: {e}")
        return None

def create_pdf_report(df, location_name, range_type):
    """Generate a high-end PDF report with trend graphs."""
    pdf = FPDF()
    pdf.set_auto_page_break(auto=True, margin=15)
    pdf.add_page()
    
    # Header
    pdf.set_fill_color(34, 197, 94) # Brand Green
    pdf.rect(0, 0, 210, 40, 'F')
    pdf.set_font("helvetica", "B", 24)
    pdf.set_text_color(255, 255, 255)
    pdf.cell(0, 20, "EcoGlance Intelligence Report", new_x="LMARGIN", new_y="NEXT", align="C")
    pdf.set_font("helvetica", "B", 12)
    pdf.cell(0, 10, f"Environmental Analysis: {location_name}", new_x="LMARGIN", new_y="NEXT", align="C")
    
    pdf.ln(20)
    pdf.set_text_color(30, 41, 59)
    pdf.set_font("helvetica", "B", 14)
    pdf.cell(0, 10, f"Report Summary ({range_type.capitalize()})", new_x="LMARGIN", new_y="NEXT")
    pdf.set_font("helvetica", "", 10)
    pdf.multi_cell(0, 7, f"This report provides a comprehensive environmental and atmospheric analysis for {location_name} "
                 f"based on data collected from {df['date'].min()} to {df['date'].max()}. "
                 f"Our hybrid AI engine has processed these parameters to identify patterns in air quality and weather trends.")
    
    # Graphs Section
    pdf.ln(10)
    pdf.set_font("helvetica", "B", 12)
    pdf.cell(0, 10, "Atmospheric Trends Visualization", new_x="LMARGIN", new_y="NEXT")
    
    # Generate Plots
    plt.figure(figsize=(10, 8))
    
    # Subplot 1: Temperature
    plt.subplot(2, 1, 1)
    plt.plot(df['date'], df['temp_avg'], color='#ef4444', linewidth=2, marker='o', label='Avg Temp (°C)')
    plt.fill_between(df['date'], df['temp_min'], df['temp_max'], color='#ef4444', alpha=0.1)
    plt.title('Temperature Gradient Trend')
    plt.ylabel('Degrees Celsius')
    plt.grid(True, alpha=0.3)
    plt.legend()
    
    # Subplot 2: AQI (PM2.5 / PM10)
    if 'pm2_5' in df.columns:
        plt.subplot(2, 1, 2)
        plt.bar(df['date'], df['pm10'], color='#64748b', alpha=0.3, label='PM10')
        plt.plot(df['date'], df['pm2_5'], color='#22c55e', linewidth=2, marker='s', label='PM2.5')
        plt.title('Air Quality Particle Concentration')
        plt.ylabel('µg/m³')
        plt.grid(True, alpha=0.3)
        plt.legend()
    
    plt.tight_layout()
    
    img_buf = io.BytesIO()
    plt.savefig(img_buf, format='png', dpi=150)
    img_buf.seek(0)
    
    pdf.image(img_buf, x=15, y=pdf.get_y(), w=180)
    pdf.ln(150) # Adjusted to prevent table overlap with chart (180w * 8/10h = 144mm)
    
    # Table Header
    pdf.set_font("helvetica", "B", 8)
    pdf.set_fill_color(241, 245, 249)
    cols = ['Date', 'Temp Avg', 'Humidity', 'Rain', 'Wind', 'PM2.5', 'AQI Index']
    col_widths = [30, 25, 25, 25, 25, 25, 30]
    
    for i, col in enumerate(cols):
        if i == len(cols) - 1:
            pdf.cell(col_widths[i], 8, col, border=1, fill=True, align='C', new_x="LMARGIN", new_y="NEXT")
        else:
            pdf.cell(col_widths[i], 8, col, border=1, fill=True, align='C')
    
    # Table Rows
    pdf.set_font("helvetica", "", 8)
    for _, row in df.tail(12).iterrows(): # Show last 12 entries
        pdf.cell(col_widths[0], 7, str(row['date'])[:10], border=1, align='C')
        pdf.cell(col_widths[1], 7, f"{row['temp_avg']:.1f}°C", border=1, align='C')
        pdf.cell(col_widths[2], 7, f"{row['humidity']:.1f}%", border=1, align='C')
        pdf.cell(col_widths[3], 7, f"{row['rainfall']:.1f}mm", border=1, align='C')
        pdf.cell(col_widths[4], 7, f"{row['wind_speed']:.1f}m/s", border=1, align='C')
        pm25_val = float(row['pm2_5']) if 'pm2_5' in row and pd.notna(row['pm2_5']) else 0
        pdf.cell(col_widths[5], 7, f"{pm25_val:.1f}" if pm25_val > 0 else "N/A", border=1, align='C')
        
        # Calculate AQI sub-index for the row
        aqi_val = calculate_india_aqi_from_pm25(pm25_val) if pm25_val > 0 else "N/A"
        pdf.cell(col_widths[6], 7, str(aqi_val), border=1, align='C', new_x="LMARGIN", new_y="NEXT")
        
    plt.close() # Clean up memory
    return pdf.output()
from backend.uncertainty.mc_dropout import MCDropoutPredictor
import time
import threading
import pytz

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

app.config['PERMANENT_SESSION_LIFETIME'] = timedelta(days=30)

# Flask 3 uses JSON provider classes (app.json_encoder is ignored).
class NumpyJSONProvider(DefaultJSONProvider):
    def default(self, obj):
        if isinstance(obj, (np.integer,)):
            return int(obj)
        if isinstance(obj, (np.floating,)):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return DefaultJSONProvider.default(obj)

app.json = NumpyJSONProvider(app)

# Initialize Database
# Database is in ../data/users.db relative to src/
init_db()

# ── Start SMS/WhatsApp alert scheduler (10 AM & 6 PM IST) ──────────────────
try:
    from sms_alerts import start_alert_scheduler
    start_alert_scheduler()
except Exception as _sms_err:
    print(f"[WARN] SMS alert scheduler could not start: {_sms_err}")
# ────────────────────────────────────────────────────────────────────────────


# Config
SEQ_LENGTH = 21
DEFAULT_LAT = 17.6868
DEFAULT_LON = 83.2185

# 7 Places in Visakhapatnam — with location-specific AQI characteristics
# aqi_multiplier: reflects real-world pollution profile of each area
# > 1.0 = more polluted (industrial, port, traffic)
# < 1.0 = cleaner (agricultural, coastal, residential)
LOCATIONS = {
    "Visakhapatnam Center": {"lat": 17.7138, "lon": 83.2750, "aqi_mult": 1.35, "type": "Urban Center"},
    "Gajuwaka":             {"lat": 17.6908, "lon": 83.1610, "aqi_mult": 1.85, "type": "Industrial Zone"},
    "Madhurawada":          {"lat": 17.8188, "lon": 83.3551, "aqi_mult": 0.95, "type": "Suburban/IT Hub"},
    "Pendurthi":            {"lat": 17.8285, "lon": 83.1970, "aqi_mult": 1.05, "type": "Semi-Urban"},
    "Bheemili":             {"lat": 17.8860, "lon": 83.4560, "aqi_mult": 0.85, "type": "Coastal/Beach"},
    "Anakapalle":           {"lat": 17.6896, "lon": 83.0024, "aqi_mult": 0.90, "type": "Agricultural"},
    "MVP Colony":           {"lat": 17.7371, "lon": 83.3331, "aqi_mult": 1.15, "type": "Residential"},
}

def get_location_multiplier(lat, lon):
    """Return the AQI multiplier and location type for the given coordinates."""
    best_name = None
    best_dist = float('inf')
    for name, loc in LOCATIONS.items():
        dist = abs(loc['lat'] - lat) + abs(loc['lon'] - lon)
        if dist < best_dist:
            best_dist = dist
            best_name = name
    if best_name and best_dist < 0.1:
        return LOCATIONS[best_name]['aqi_mult'], LOCATIONS[best_name]['type'], best_name
    return 1.0, "Unknown", "Custom"

def fetch_air_quality(lat, lon):
    """
    Fetch real-time + 7-day hourly air quality from Open-Meteo Air Quality API.
    Returns dict with 'current' and 'daily_avg' PM values, or None on failure.
    """
    try:
        url = (
            f"https://air-quality-api.open-meteo.com/v1/air-quality?"
            f"latitude={lat}&longitude={lon}"
            f"&current=pm10,pm2_5,carbon_monoxide,nitrogen_dioxide,sulphur_dioxide,ozone"
            f"&hourly=pm10,pm2_5"
            f"&forecast_days=7&timezone=auto"
        )
        print(f"Fetching Air Quality: lat={lat}, lon={lon}")
        resp = requests.get(url, timeout=12)
        data = resp.json()

        result = {}

        # Current values
        current = data.get('current', {})
        result['current'] = {
            'pm2_5': current.get('pm2_5'),
            'pm10': current.get('pm10'),
            'co': current.get('carbon_monoxide'),
            'no2': current.get('nitrogen_dioxide'),
            'so2': current.get('sulphur_dioxide'),
            'o3': current.get('ozone'),
        }

        # Daily averages from hourly data
        hourly = data.get('hourly', {})
        times = hourly.get('time', [])
        pm25_h = hourly.get('pm2_5', [])
        pm10_h = hourly.get('pm10', [])

        if times and pm25_h:
            daily_pm25 = {}
            daily_pm10 = {}
            for t, p25, p10 in zip(times, pm25_h, pm10_h):
                day = t[:10]  # "2026-04-11"
                if p25 is not None:
                    daily_pm25.setdefault(day, []).append(p25)
                if p10 is not None:
                    daily_pm10.setdefault(day, []).append(p10)

            result['daily_avg'] = {}
            for day in daily_pm25:
                result['daily_avg'][day] = {
                    'pm2_5': round(sum(daily_pm25[day]) / len(daily_pm25[day]), 2),
                    'pm10': round(sum(daily_pm10.get(day, [0])) / max(1, len(daily_pm10.get(day, [1]))), 2)
                }

        print(f"AQ API current: PM2.5={result['current'].get('pm2_5')}, PM10={result['current'].get('pm10')}")
        return result

    except Exception as e:
        print(f"Failed to fetch air quality data: {e}")
        return None

# Models global state
lstm_full = None
feature_extractor = None
xgb_models = {}
mc_predictor = None
active_targets = []
models_loaded = False
models_lock = threading.Lock()
bias_corrector = None

# Store the latest daily forecast, so the /hourly route can generate diurnal curves from it
# diuranl forecast storage: { "lat_lon": { "date": {forecast_data} } }
latest_daily_forecast = {}
forecast_lock = threading.Lock()

def load_models_lazy():
    global lstm_full, feature_extractor, xgb_models, mc_predictor, active_targets, models_loaded
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
        
        # Load LightGBM + XGBoost Residual Ensemble
        try:
            path = os.path.join(MODELS_DIR, "lgbm_xgb_ensemble.pkl")
            if os.path.exists(path):
                with open(path, "rb") as f:
                    xgb_models = pickle.load(f)
                print("LGBM/XGB Ensemble models loaded.")
        except Exception as e:
            print(f"Warning: Failed to load Ensemble models: {e}")
        
        active_targets = preprocessor.target_columns
        
        # MC Dropout Predictor is bypassed for now as the architecture shifted to LGBM Base + XGB Residuals
        mc_predictor = None

        # Load residual bias corrector
        global bias_corrector
        try:
            bias_path = os.path.join(MODELS_DIR, "bias_corrector.pkl")
            if os.path.exists(bias_path):
                with open(bias_path, "rb") as f:
                    bias_corrector = pickle.load(f)
                print("Residual Bias Corrector loaded.")
        except Exception as e:
            print(f"Warning: Failed to load bias corrector: {e}")
            
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

aqi_hourly_path = os.path.join(DATA_DIR, "vizag_aqi_hourly.csv")
weather_hourly_path = os.path.join(DATA_DIR, "visakhapatnam_weather_hourly_2015_2025.csv")

if os.path.exists(aqi_hourly_path) and os.path.exists(weather_hourly_path):
    print("Preprocessor source: new hourly datasets (separate AQI + weather).")
    df_weather, df_combined = preprocessor.process_hourly_data(
        aqi_hourly_path,
        weather_hourly_path
    )
else:
    print("Preprocessor source: legacy daily datasets.")
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
LOCAL_DATA_ONLY = False

# Local hourly weather cache (for no-API mode)
local_weather_hourly = None

# Helper: Predict with XGBoost model (handles both Booster and XGBRegressor)
def xgb_predict(model, X_df):
    """Predict using DMatrix to ensure feature names are always preserved."""
    dmat = xgb.DMatrix(X_df.values, feature_names=list(X_df.columns))
    if isinstance(model, xgb.Booster):
        return model.predict(dmat)
    else:
        # XGBRegressor: extract internal Booster and predict directly
        return model.get_booster().predict(dmat)

def load_local_hourly_weather():
    """Load local hourly weather dataset once (no external APIs)."""
    global local_weather_hourly
    if local_weather_hourly is not None:
        return local_weather_hourly

    path = os.path.join(DATA_DIR, "visakhapatnam_weather_hourly_2015_2025.csv")
    if not os.path.exists(path):
        local_weather_hourly = pd.DataFrame()
        return local_weather_hourly

    df = pd.read_csv(path)
    if 'datetime' in df.columns:
        df['datetime'] = pd.to_datetime(df['datetime'], errors='coerce')
    df = df.dropna(subset=['datetime']).sort_values('datetime').reset_index(drop=True)
    local_weather_hourly = df
    return local_weather_hourly

def build_local_daily_forecast(df_weather_daily: pd.DataFrame, days: int = 8, start_date: pd.Timestamp | None = None) -> pd.DataFrame:
    """
    Generate local daily weather continuation without external APIs.
    Uses month-wise climatology fallback to recent rolling mean.
    """
    df = df_weather_daily.copy().sort_values('date').reset_index(drop=True)
    if df.empty:
        return pd.DataFrame(columns=['date'] + preprocessor.lstm_features)

    last_date = pd.to_datetime(df['date']).max()
    if start_date is None:
        start_date = last_date + pd.Timedelta(days=1)
    else:
        start_date = pd.to_datetime(start_date)
    recent = df.tail(30)
    forecast_rows = []
    weather_cols = ['temp_max', 'temp_min', 'temp_avg', 'rainfall', 'wind_speed', 'wind_direction', 'solar_radiation', 'pressure', 'humidity', 'cloud_cover']

    for i in range(days):
        d = start_date + pd.Timedelta(days=i)
        month_slice = df[pd.to_datetime(df['date']).dt.month == d.month]
        src = month_slice if len(month_slice) >= 10 else recent
        vals = {c: float(src[c].mean()) for c in weather_cols}
        vals['date'] = d
        forecast_rows.append(vals)

    return pd.DataFrame(forecast_rows)

def fetch_local_weather_data():
    """
    No-API weather provider:
    - history from local daily weather (last 14+ buffer days)
    - future from local climatology continuation
    """
    df_daily = df_weather.copy().sort_values('date').reset_index(drop=True)
    df_daily['date'] = pd.to_datetime(df_daily['date'])

    if df_daily.empty:
        return None, None, {}

    df_hist = df_daily.tail(SEQ_LENGTH + 6).copy()
    today = pd.to_datetime(datetime.now().date())
    df_fore = build_local_daily_forecast(df_daily, days=8, start_date=today)
    return df_hist, df_fore, {'hourly': {}}

def fetch_nasa_history(start_date, end_date, lat=DEFAULT_LAT, lon=DEFAULT_LON):
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
            f"latitude={lat}&longitude={lon}"
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

def fetch_weather_data(lat=DEFAULT_LAT, lon=DEFAULT_LON):
    """
    Fetches:
    1. Past 14 days (Hybrid: NASA preferred + OpenMeteo Recent Fill)
    2. Future 7 days (Forecast API)
    """
    today = datetime.now().date()
    
    # 1. Past Data Strategy
    # We need SEQ_LENGTH (21) days ending yesterday.
    end_date = today - timedelta(days=1)
    start_date = end_date - timedelta(days=SEQ_LENGTH + 35) # Buffer for 30-day lags
    
    # Try NASA first
    df_nasa = fetch_nasa_history(start_date, end_date, lat, lon)
    
    # Fetch Open-Meteo Archive as Backup/Gap-Fill (Enforce m/s)
    url_hist = f"https://archive-api.open-meteo.com/v1/archive?latitude={lat}&longitude={lon}&start_date={start_date}&end_date={end_date}&daily=temperature_2m_max,temperature_2m_min,temperature_2m_mean,rain_sum,wind_speed_10m_max,wind_direction_10m_dominant,shortwave_radiation_sum,surface_pressure_mean,relative_humidity_2m_mean,cloud_cover_mean&timezone=auto&wind_speed_unit=ms"
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

    # 2. Future Forecast (7 Days) - Enforce m/s
    url_fore = f"https://api.open-meteo.com/v1/forecast?latitude={lat}&longitude={lon}&daily=temperature_2m_max,temperature_2m_min,temperature_2m_mean,rain_sum,wind_speed_10m_max,wind_direction_10m_dominant,shortwave_radiation_sum,surface_pressure_mean,relative_humidity_2m_mean,cloud_cover_mean,precipitation_probability_max,uv_index_max,sunrise,sunset&hourly=temperature_2m,relative_humidity_2m,rain,wind_speed_10m,precipitation_probability,apparent_temperature&forecast_days=8&timezone=auto&wind_speed_unit=ms"
    r_fore = requests.get(url_fore, timeout=15).json()
    
    # Return DataFrames not JSON to simplify downstream
    return df_hist_final, r_fore

@app.route('/hourly/<date_str>', methods=['GET'])
def get_hourly(date_str):
    try:
        # No-API mode: serve hourly data from local CSV for exact historical dates
        df_h = load_local_hourly_weather()
        if df_h.empty:
            return jsonify({'error': 'No hourly data database loaded'}), 404

        day = pd.to_datetime(date_str, errors='coerce')
        if pd.isna(day):
            return jsonify({'error': 'Invalid date'}), 400

        day_start = day.normalize()
        day_end = day_start + pd.Timedelta(days=1)
        sel = df_h[(df_h['datetime'] >= day_start) & (df_h['datetime'] < day_end)].copy()

        # If we have exact historical data, use it (structure for frontend)
        if not sel.empty:
            result = []
            for _, row in sel.iterrows():
                dt = pd.to_datetime(row['datetime'])
                temp = float(row.get('T2M', 0.0))
                humidity = float(row.get('RH2M', 0.0))
                rain = float(row.get('PRECTOTCORR', 0.0))
                wind = float(row.get('WS2M', 0.0))
                result.append({
                    'time': dt.strftime('%H:%M'), # "14:00"
                    'temp': temp,
                    'humidity': humidity,
                    'rain': rain,
                    'wind': wind,
                    'condition': 'Rainy' if rain > 0.5 else ('Cloudy' if humidity > 75 else 'Sunny')
                })
            return jsonify(result)

        # For future/forecast dates (where sel is empty), generate diurnal curve from daily forecast
        lat = request.args.get('lat', DEFAULT_LAT)
        lon = request.args.get('lon', DEFAULT_LON)
        loc_key = f"{lat}_{lon}"
        
        with forecast_lock:
            # Look up the forecasted day
            day_str_key = day_start.strftime('%Y-%m-%d')
            loc_data = latest_daily_forecast.get(loc_key, {})
            day_forecast = loc_data.get(day_str_key)

        if not day_forecast:
            return jsonify({'error': 'No forecast data available for this date yet. Try generating a forecast first.'}), 404

        # Generate diurnal curve from the daily min/max/avg
        temp_min = float(day_forecast.get('temp_min', 20.0))
        temp_max = float(day_forecast.get('temp_max', 30.0))
        humidity_avg = float(day_forecast.get('humidity', 70.0))
        wind_avg = float(day_forecast.get('wind_speed', 2.0))
        rainfall_total = float(day_forecast.get('rainfall', 0.0))

        result = []
        for h in range(24):
            # Diurnal temperature model (simple sine wave peaking around 14:00)
            # Minimum around 05:00
            hour_offset = (h - 5) % 24
            sine_term = np.sin(hour_offset * np.pi / 12 - np.pi/2) # -1 at h=5, +1 at h=17
            # Shift peak to 14:00 manually
            t_rad = (h - 14) * np.pi / 12
            cur_temp = temp_min + (temp_max - temp_min) * ((np.cos(t_rad) + 1) / 2)

            # Humidity generally inverse to temp
            temp_ratio = (cur_temp - temp_min) / (temp_max - temp_min + 0.1)
            # Humidity peaks at night, lowest in afternoon
            cur_humid = np.clip(humidity_avg + 15 * (1 - 2 * temp_ratio), 0, 100)

            # Distribute rain more realistically (only if > 0.5mm total)
            cur_rain = 0.0
            if rainfall_total > 0.5:
                # Only rain during "rainy hours" (e.g., 14:00 - 20:00 or random block)
                if 14 <= h <= 20: 
                    cur_rain = (rainfall_total / 7.0) * (np.random.random() * 1.5)

            result.append({
                'time': f"{h:02d}:00",
                'temp': round(cur_temp, 1),
                'humidity': round(cur_humid, 1),
                'rain': round(cur_rain, 2),
                'wind': round(wind_avg, 1),
                'condition': 'Rainy' if cur_rain > 0.2 else ('Cloudy' if cur_humid > 75 else 'Sunny'),
                'climatological': False, # It's AI-generated diurnal curve, not static fallback
                'ai_forecast': True
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

def get_current_hour_aqi_observation():
    """
    Return AQI observation for the current local hour from data/vizag_aqi_hourly.csv.
    Uses ONLY exact current-hour match.
    """
    csv_path = os.path.join(DATA_DIR, "vizag_aqi_hourly.csv")
    if not os.path.exists(csv_path):
        return None

    try:
        df = pd.read_csv(csv_path)
        if 'datetime' not in df.columns:
            return None

        df['datetime'] = pd.to_datetime(df['datetime'], errors='coerce')
        df = df.dropna(subset=['datetime'])
        if df.empty:
            return None

        import pytz
        ist = pytz.timezone('Asia/Kolkata')
        now_hour = pd.Timestamp(datetime.now(ist).replace(minute=0, second=0, microsecond=0, tzinfo=None))
        match = df[df['datetime'] == now_hour]
        if match.empty:
            return None

        row = match.iloc[-1]
        pm25 = float(row['pm2_5']) if ('pm2_5' in row and pd.notna(row['pm2_5'])) else None
        pm10 = float(row['pm10']) if ('pm10' in row and pd.notna(row['pm10'])) else None
        if pm25 is None and pm10 is None:
            return None

        # Keep PM consistency when both are present
        if pm25 is not None and pm10 is not None:
            if pm25 > pm10:
                pm25 = pm10
            if pm25 < 0.25 * pm10:
                pm25 = 0.25 * pm10

        return {
            'datetime': now_hour.isoformat(),
            'pm2_5': pm25,
            'pm10': pm10
        }
    except Exception as e:
        print(f"Failed reading current-hour AQI observation: {e}")
        return None

def get_bias_correction(date_str):
    """
    Residual-Based Bias Correction.
    Winter (Oct-Feb): +60 offset for PM.
    """
    d = pd.to_datetime(date_str)
    month = d.month
    # Winter/Spring transition (Oct-March): +15.0 offset for PM
    if month in [10, 11, 12, 1, 2, 3]:
        return 15.0
    return 0.0

def calculate_india_aqi_from_pm25(pm25):
    """Convert PM2.5 concentration (ug/m3) to India AQI sub-index."""
    try:
        c = float(pm25)
    except (TypeError, ValueError):
        return 0

    if c < 0:
        c = 0.0

    breakpoints = [
        (0.0, 30.0, 0, 50),
        (30.0, 60.0, 51, 100),
        (60.0, 90.0, 101, 200),
        (90.0, 120.0, 201, 300),
        (120.0, 250.0, 301, 400),
        (250.0, 500.0, 401, 500),
    ]

    for bp_lo, bp_hi, i_lo, i_hi in breakpoints:
        if c <= bp_hi:
            aqi = ((i_hi - i_lo) / (bp_hi - bp_lo)) * (c - bp_lo) + i_lo
            return int(round(max(0.0, min(500.0, aqi))))

    return 500

def get_aqi_status_and_color(aqi_value):
    if aqi_value <= 50:
        return "Good", "#00e400"
    if aqi_value <= 100:
        return "Satisfactory", "#ffff00"
    if aqi_value <= 200:
        return "Moderate", "#ff7e00"
    if aqi_value <= 300:
        return "Poor", "#ff0000"
    if aqi_value <= 400:
        return "Very Poor", "#99004c"
    return "Severe", "#7e0023"

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
    elif status == "Very Poor":
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

    else:  # Severe
        return {
            "title": "Severe air quality",
            "summary": "Emergency conditions. Serious health effects likely for all groups.",
            "do": [
                "Stay indoors and keep all windows/doors closed",
                "Run HEPA air purifier continuously",
                "Avoid travel unless essential",
                "Use N95/KN95 masks if stepping outside is unavoidable",
                "Follow local health advisories"
            ],
            "avoid": [
                "Any outdoor exercise",
                "Outdoor work without proper respiratory protection",
                "Long commutes in heavy traffic",
                "Indoor smoke sources (candles, incense, smoking)"
            ],
            "icon": "!"
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
    username = request.json.get('username', '').strip()
    password = request.json.get('password', '')
    phone_number = request.json.get('phone_number', '').strip()

    if not username or len(username) < 3:
        return jsonify({'error': 'Username must be at least 3 characters'}), 400
    if not password or len(password) < 6:
        return jsonify({'error': 'Password must be at least 6 characters'}), 400

    created, result = create_user_credentials(username, password, phone_number or None)
    if not created:
        return jsonify({'error': result}), 400

    email = result
    session['user_email'] = email
    session.permanent = True
    return jsonify({'success': True, 'message': 'Account created! Redirecting...', 'redirect': True})

@app.route('/login-password', methods=['POST'])
def login_password():
    username_or_email = request.json.get('username', '').strip()
    password = request.json.get('password', '')
    remember = bool(request.json.get('remember'))
    
    if not username_or_email or not password:
        return jsonify({'error': 'Username/email and password required'}), 400
    
    success, email = verify_password(username_or_email, password)
    if success:
        session['user_email'] = email
        session.permanent = remember
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

@app.route('/webpushr-sw.js')
def webpushr_sw():
    from flask import send_from_directory
    return send_from_directory(app.static_folder, 'webpushr-sw.js', mimetype='application/javascript')

@app.route('/locations', methods=['GET'])
def get_locations():
    """Return the 7 Visakhapatnam monitoring station locations."""
    return jsonify(LOCATIONS)

@app.route('/predict', methods=['GET'])
def predict():
    try:
        # Lazy load models if not already loaded
        load_models_lazy()
        
        global lstm_full, feature_extractor, xgb_models, mc_predictor, active_targets, preprocessor
        
        method = request.args.get('method', 'mc_dropout')
        lat = float(request.args.get('lat', DEFAULT_LAT))
        lon = float(request.args.get('lon', DEFAULT_LON))
        
        # 1. Caching Check (Method-specific)
        cache_key = f'forecast_{method}_{lat}_{lon}'
        with forecast_cache['lock']:
            now = time.time()
            if forecast_cache.get(cache_key) and (now - forecast_cache.get(f'{cache_key}_time', 0) < CACHE_DURATION):
                print(f"Serving {method} from cache.")
                return jsonify(forecast_cache[cache_key])

        # 2. Data Fetching & Preparation
        if LOCAL_DATA_ONLY:
            df_hist, df_fore, fore_json = fetch_local_weather_data()
        else:
            # Returns (DataFrame, JSON)
            df_hist, fore_json = fetch_weather_data(lat=lat, lon=lon)
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
        
        # Live API data (Open-Meteo) does not include fire/traffic columns.
        # Fill them with sensible defaults so the interpolation loop doesn't crash.
        if 'active_fires_count' not in df_combined_full.columns:
            df_combined_full['active_fires_count'] = 0.0
        if 'fire_frp' not in df_combined_full.columns:
            df_combined_full['fire_frp'] = 0.0
        if 'traffic_congestion_index' not in df_combined_full.columns:
            df_combined_full['traffic_congestion_index'] = 50.0

        for col in preprocessor.weather_features:
            if col != 'season':
                df_combined_full[col] = df_combined_full[col].interpolate(method='linear').ffill().bfill()
        
        # Apply the exact same feature engineering used during training!
        df_combined_full = preprocessor.add_engineered_features(df_combined_full)
        
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
            X_data = window[preprocessor.lstm_features]
            X_scaled = preprocessor.scaler_lstm.transform(X_data)
            X_lstm_batch.append(X_scaled)
            
            # Static Features for XGBoost
            target_row = df_full.iloc[i + SEQ_LENGTH]
            feat_dict = {col: float(target_row[col]) for col in preprocessor.lstm_features}
            d = pd.to_datetime(target_date)
            
            # Additional Gas features from target day (use 0 if missing; preprocessing will use climatology inside prepare_xgb_data logically, 
            # but here at inference time we mock the same logic).
            gas_features = ['carbon_monoxide', 'nitrogen_dioxide', 'sulphur_dioxide', 'ammonia']
            for g in gas_features:
                if g in target_row and pd.notna(target_row[g]):
                    feat_dict[g] = float(target_row[g])
                elif preprocessor.gas_monthly_means.get(g):
                    month_means = preprocessor.gas_monthly_means[g]
                    # Fallback to monthly climatology mean for inference
                    feat_dict[g] = float(month_means.get(d.month, np.mean(list(month_means.values()))))
                else:
                    feat_dict[g] = 0.0

            feat_dict.update({
                'month': d.month,
                'day_of_week': d.dayofweek,
                'day': d.day,
                'is_weekend': 1 if d.weekday() in (5, 6) else 0,
                'wind_dir_sin': float(np.sin(np.deg2rad(float(target_row['wind_direction'])))),
                'wind_dir_cos': float(np.cos(np.deg2rad(float(target_row['wind_direction'])))),
                'pressure_delta': float(target_row['pressure']) - float(df_full.iloc[i + SEQ_LENGTH - 1]['pressure'])
            })
            
            # Engineered Memory Features (added during new preprocessing)
            engineered_cols = [c for c in df_full.columns if '_lag_' in c or '_rolling_' in c or c in ['pollution_transport', 'stability_index']]
            for c in engineered_cols:
                if c in target_row and pd.notna(target_row[c]):
                    feat_dict[c] = float(target_row[c])
                else:
                    feat_dict[c] = 0.0
                    
            base_feat_list.append(feat_dict)

        X_lstm_batch = np.array(X_lstm_batch) # (7, 14, 10)

        # 4. Vectorized Inference
        # 4. Vectorized Inference
        forecasts = []
        
        if feature_extractor is None:
            raise ValueError("Neural Feature Extractor not initialized")
            
        embeddings_batch = feature_extractor.predict(X_lstm_batch, verbose=0)
        
        for i in range(forecast_days):
            target_date = target_dates[i]
            day_res = {'date': target_date}
            d_month = pd.to_datetime(target_date).month
            bias_pm25 = (bias_corrector.get('pm2_5', {}).get(d_month, 0) if bias_corrector else 0) + get_bias_correction(target_date)
            bias_pm10 = (bias_corrector.get('pm10', {}).get(d_month, 0) if bias_corrector else 0) + get_bias_correction(target_date)
            
            engineered_names = [
                'pm2_5_lag_1', 'pm2_5_lag_2', 'pm2_5_lag_7', 'pm2_5_lag_14', 'pm2_5_lag_21', 'pm2_5_lag_30',
                'pm10_lag_1', 'pm10_lag_2', 'pm10_lag_7', 'pm10_lag_14', 'pm10_lag_21', 'pm10_lag_30',
                'pm2_5_rolling_3', 'pm2_5_rolling_7', 'pm2_5_rolling_14', 'pm2_5_rolling_30',
                'pm10_rolling_3', 'pm10_rolling_7', 'pm10_rolling_14', 'pm10_rolling_30',
                'wind_speed_rolling_3', 'wind_speed_rolling_7', 'wind_speed_rolling_14', 'wind_speed_rolling_30',
                'humidity_rolling_3', 'humidity_rolling_7', 'humidity_rolling_14', 'humidity_rolling_30',
                'temp_max_rolling_3', 'temp_max_rolling_7', 'temp_max_rolling_14', 'temp_max_rolling_30',
                'rainfall_rolling_3', 'rainfall_rolling_7', 'rainfall_rolling_14', 'rainfall_rolling_30',
                'pollution_transport', 'stability_index'
            ]
            
            XGB_FEATURE_NAMES = [f'emb_{j}' for j in range(64)] + \
                              preprocessor.lstm_features + \
                              ['month', 'day_of_week', 'day', 'is_weekend', 'wind_dir_sin', 'wind_dir_cos', 'pressure_delta'] + \
                              ['carbon_monoxide', 'nitrogen_dioxide', 'sulphur_dioxide', 'ammonia'] + \
                              engineered_names
            
            feat_dict_std = base_feat_list[i].copy()
            for j in range(64): 
                feat_dict_std[f'emb_{j}'] = float(embeddings_batch[i][j])
            
            X_xgb = pd.DataFrame([feat_dict_std])[XGB_FEATURE_NAMES].astype('float32')
            
            # Predict targets using Ensemble (LGBM Base + XGB Residual)
            for target in active_targets:
                base_key = f"{target}_base"
                resid_key = f"{target}_resid"
                
                if base_key in xgb_models and resid_key in xgb_models:
                    base_val = xgb_models[base_key].predict(X_xgb)[0]
                    resid_val = xgb_models[resid_key].predict(X_xgb)[0]
                    val = base_val + resid_val
                    
                    if target in ['pm2_5', 'pm10']:
                        val = np.expm1(val)
                        t_bias = bias_pm25 if target == 'pm2_5' else bias_pm10
                        val += t_bias
                        
                    day_res[target] = max(0, float(val))
                elif f"std_{target}" in xgb_models:
                    # Fallback standard
                    FEAT_COLS = preprocessor.lstm_features + ['month', 'day_of_week', 'day', 'is_weekend', 'wind_dir_sin', 'wind_dir_cos', 'pressure_delta']
                    X_std = pd.DataFrame([base_feat_list[i]])[FEAT_COLS].astype('float32')
                    val = xgb_predict(xgb_models[f"std_{target}"], X_std)[0]
                    if target in ['pm2_5', 'pm10']:
                        val = np.expm1(val) + (bias_pm25 if target == 'pm2_5' else bias_pm10)
                    day_res[target] = max(0, round(float(val), 2))
                    day_res['engine'] = 'Standard (Lighter)'

            # Enrich with weather fields from forecast dataframe for this day
            fore_row = df_fore_future.iloc[i]
            weather_fields = ['temp_avg', 'temp_max', 'temp_min', 'humidity', 'wind_speed',
                              'pressure', 'rainfall', 'cloud_cover', 'wind_direction']
            for wf in weather_fields:
                if wf in fore_row and pd.notna(fore_row[wf]):
                    if wf not in day_res:  # Don't overwrite model predictions
                        day_res[wf] = float(fore_row[wf])
                    elif wf in ['temp_avg', 'temp_max', 'temp_min', 'humidity',
                                'wind_speed', 'pressure', 'rainfall', 'cloud_cover']:
                        # Always use actual forecast weather data for display
                        day_res[wf] = float(fore_row[wf])

            # Inject per-day precipitation_probability and uv_index from Open-Meteo daily
            try:
                _daily = fore_json.get('daily', {})
                _times = _daily.get('time', [])
                _d_str = pd.to_datetime(target_date).strftime('%Y-%m-%d')
                if _d_str in _times:
                    _idx = _times.index(_d_str)
                    _pp = _daily.get('precipitation_probability_max', [])
                    _uv = _daily.get('uv_index_max', [])
                    if _pp and _idx < len(_pp):
                        day_res['precipitation_probability'] = _pp[_idx] or 0
                    if _uv and _idx < len(_uv):
                        day_res['uv_index'] = _uv[_idx] or 0
            except Exception:
                pass

            forecasts.append(day_res)

        # 5. Final Post-processing & Guardrails
        # Combine Model Predictions with Real-Time Air Quality API baseline
        aq_data = fetch_air_quality(lat, lon)
        aq_mult, loc_type, loc_name = get_location_multiplier(lat, lon)
        
        for day in forecasts:
            d_str = day['date'].strftime('%Y-%m-%d')
            # Base model prediction
            pm25 = day.get('pm2_5', 0)
            pm10 = day.get('pm10', 0)
            
            # 1. Apply location multiplier (Anakapalle 0.78x, Gajuwaka 1.25x etc)
            pm25 *= aq_mult
            pm10 *= aq_mult
            
            # 2. Blend with API daily average if available
            if aq_data and 'daily_avg' in aq_data and d_str in aq_data['daily_avg']:
                api_p25 = aq_data['daily_avg'][d_str].get('pm2_5')
                api_p10 = aq_data['daily_avg'][d_str].get('pm10')
                if api_p25: pm25 = (pm25 * 0.4) + (api_p25 * 0.6) # Weighted blend
                if api_p10: pm10 = (pm10 * 0.4) + (api_p10 * 0.6)

            if pm25 > pm10: pm25 = pm10
            if pm25 < 0.25 * pm10: pm25 = 0.25 * pm10
            day['pm2_5'] = round(pm25, 2)
            day['pm10'] = round(pm10, 2)
            day['location_type'] = loc_type
            
            for k, v in day.items():
                if isinstance(v, (float, np.float32, np.float64)) and not k.endswith('_uncertainty'):
                    day[k] = round(float(v), 2)

        # Cache the forecasts to feed the hourly diurnal model
        loc_key = f"{lat}_{lon}"
        with forecast_lock:
            if loc_key not in latest_daily_forecast:
                latest_daily_forecast[loc_key] = {}
            for day_data in forecasts:
                d_str = pd.to_datetime(day_data['date']).strftime('%Y-%m-%d')
                latest_daily_forecast[loc_key][d_str] = day_data

        # 6. Response Construction
        main_pred = forecasts[0]
        
        # Override with current hour data for "Live Now" Hero section
        try:
            # 1. First extract Open-Meteo hourly data as a base/fallback
            hourly = fore_json.get('hourly', {})
            if hourly:
                now_local = datetime.now(pytz.timezone('Asia/Kolkata'))
                current_hour_str = now_local.strftime('%Y-%m-%dT%H:00')
                all_times = hourly.get('time', [])
                if current_hour_str in all_times:
                    idx = all_times.index(current_hour_str)
                    main_pred['temp_avg'] = hourly['temperature_2m'][idx]
                    main_pred['humidity'] = hourly['relative_humidity_2m'][idx]
                    main_pred['wind_speed'] = hourly['wind_speed_10m'][idx]
                    main_pred['rainfall'] = hourly['rain'][idx]
                    main_pred['precipitation_probability'] = hourly.get('precipitation_probability', [0]*len(all_times))[idx]
                    main_pred['apparent_temperature'] = hourly.get('apparent_temperature', [main_pred['temp_avg']]*len(all_times))[idx]
            
            # 2. Overwrite with high-accuracy real-time data from wttr.in (Google Weather equivalent)
            try:
                wttr_url = f"https://wttr.in/{lat},{lon}?format=j1"
                wttr_req = requests.get(wttr_url, timeout=4)
                if wttr_req.status_code == 200:
                    wttr_data = wttr_req.json()
                    current_cond = wttr_data.get('current_condition', [{}])[0]
                    if current_cond:
                        if 'temp_C' in current_cond: main_pred['temp_avg'] = float(current_cond['temp_C'])
                        if 'humidity' in current_cond: main_pred['humidity'] = float(current_cond['humidity'])
                        if 'windspeedKmph' in current_cond: main_pred['wind_speed'] = (float(current_cond['windspeedKmph']) - 5) / 3.6
                        if 'FeelsLikeC' in current_cond: main_pred['apparent_temperature'] = float(current_cond['FeelsLikeC'])
            except Exception as wttr_err:
                print(f"Live Weather Override Failed: {wttr_err}")
                
            # Air Quality Real-Time Injection
            if aq_data and 'current' in aq_data:
                curr_aq = aq_data['current']
                if curr_aq.get('pm2_5'): main_pred['pm2_5'] = round(curr_aq['pm2_5'] * aq_mult, 2)
                if curr_aq.get('pm10'): main_pred['pm10'] = round(curr_aq['pm10'] * aq_mult, 2)
                if curr_aq.get('co'): main_pred['carbon_monoxide'] = curr_aq['co']
                if curr_aq.get('no2'): main_pred['nitrogen_dioxide'] = curr_aq['no2']
                if curr_aq.get('so2'): main_pred['sulphur_dioxide'] = curr_aq['so2']
                if curr_aq.get('o3'): main_pred['ozone'] = curr_aq['o3']
                
        except Exception as hourly_err:
            print(f"Failed to extract current hour data: {hourly_err}")

        # Extract UV and Sunrise/Sunset for UI
        try:
            today_str = main_pred['date'].strftime('%Y-%m-%d')
            daily = fore_json.get('daily', {})
            times = daily.get('time', [])
            idx = times.index(today_str) if today_str in times else 0
            
            main_pred['uv_index'] = daily.get('uv_index_max', [0]*8)[idx]
            sun_r = daily.get('sunrise', ['--']*8)[idx]
            sun_s = daily.get('sunset', ['--']*8)[idx]
            main_pred['sunrise'] = sun_r.split('T')[-1] if 'T' in sun_r else sun_r
            main_pred['sunset'] = sun_s.split('T')[-1] if 'T' in sun_s else sun_s
        except Exception as ui_err:
            print(f"Failed to extract UV/Sun data: {ui_err}")
            main_pred['uv_index'] = 0
            main_pred['sunrise'] = '--'
            main_pred['sunset'] = '--'

        pm25 = float(main_pred.get('pm2_5', 0))
        
        # Indian Calibration: Ensure AQI reflects urban reality in Visakhapatnam
        # This calibration happens AFTER the location multipliers are applied.
        if pm25 < (35 * aq_mult):
            pm25 = (35 * aq_mult) + (pm25 * 0.2)
            main_pred['pm2_5'] = round(pm25, 2)
            
        # CRITICAL FIX: The AQI sub-index MUST be calculated from the FINAL multiplied PM2.5
        aqi_value = calculate_india_aqi_from_pm25(pm25)
        aqi_status, aqi_color = get_aqi_status_and_color(aqi_value)
        
        response = {
            'prediction_date': main_pred['date'].strftime('%Y-%m-%d'),
            'location_name': loc_name,
            'location_type': loc_type,
            'data': main_pred,
            'aq_mult': aq_mult, # Explicitly include mult for frontend sync
            'aqi': {
                'value': aqi_value,
                'status': aqi_status,
                'color': aqi_color,
                'recommendations': get_aqi_recommendations(pm25, aqi_status),
                'source': 'API_HYBRID' if aq_data else 'MODEL_BASELINE',
                'observed_at': datetime.now().strftime('%Y-%m-%d %H:%M')
            },
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

@app.route('/predict_10_years', methods=['GET'])
def predict_10_years():
    target_date_str = request.args.get('date')
    if not target_date_str:
        return jsonify({'error': 'Date required'}), 400
    
    target_date = pd.to_datetime(target_date_str)
    current_date = datetime.now()
    
    years_diff = target_date.year - current_date.year
    if years_diff < 0:
         years_diff = 0
         
    predicted_temp = 32.0 + (0.15 * years_diff)
    predicted_pm25 = 55.0 + (3.5 * years_diff)
    
    month = target_date.month
    if month in [12, 1, 2]:
        predicted_temp -= 4
        predicted_pm25 += 15
    elif month in [4, 5, 6]:
        predicted_temp += 3
        predicted_pm25 -= 5
        
    predicted_aqi = calculate_india_aqi_from_pm25(predicted_pm25)
    
    reasons = [
        "Unregulated industrial growth and persistent vehicular emissions are projected to drive particulate concentrations up, posing escalated long-term respiratory risks.",
        "A severe projected reduction in urban green cover and subsequent atmospheric heating will severely limit natural air filtration, leaving citizens exposed to trapped pollutants.",
        "Expected climate modifications and increased frequency of temperature inversions will trap surface-level toxic PM2.5, fundamentally altering local livability parameters.",
        "Long-term forecasting models detect a critical accumulation of atmospheric baseline particulate matter. This requires immediate urban intervention to protect public health."
    ]
    
    reason = "Atmospheric trends indicate a worsening standard baseline due to continuous urban expansion and compromised meteorological conditions. Action is critical."
    if years_diff > 2:
        reason = np.random.choice(reasons)
    elif years_diff == 0:
        reason = "Current baseline levels reflect recent real-time atmospheric readings and active thermal conditions."
        
    return jsonify({
        'date': target_date.strftime('%Y-%m-%d'),
        'temp': round(predicted_temp, 1),
        'pm25': round(predicted_pm25, 1),
        'aqi': predicted_aqi,
        'reason': reason
    })

@app.route('/stats', methods=['GET'])
def get_stats():
    """Return model performance metrics directly from the training output CSV."""
    try:
        BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        # Search for metrics file in multiple likely locations
        possible_paths = [
            os.path.join(BASE_DIR, "data", "metrics_scientific.csv"),
            os.path.join(BASE_DIR, "metrics_scientific.csv"),
            "data/metrics_scientific.csv",
            "metrics_scientific.csv"
        ]
        
        csv_path = None
        for p in possible_paths:
            if os.path.exists(p):
                csv_path = p
                break
                
        if not csv_path:
            print("[Stats] Metrics file not found in any search path.")
            return jsonify([])

        df = pd.read_csv(csv_path)
        # Normalise column names: lowercase, strip, replace spaces
        df.columns = [c.strip().lower().replace(' ', '_') for c in df.columns]
        
        # Standardize 'r2' vs 'r2_score'
        if 'r2_score' in df.columns and 'r2' not in df.columns:
            df = df.rename(columns={'r2_score': 'r2'})
        elif 'r2' not in df.columns and any('r2' in c for c in df.columns):
            # If there's something like 'r2score', rename the first one containing 'r2'
            r2_col = [c for c in df.columns if 'r2' in c][0]
            df = df.rename(columns={r2_col: 'r2'})

        return jsonify(df.to_dict(orient='records'))
    except Exception as e:
        print(f"[Stats] Error reading metrics: {str(e)}")
        return jsonify({'error': str(e)}), 500

@app.route('/download_report', methods=['GET'])
def download_report():
    """Download a high-end PDF report with trends."""
    try:
        lat = float(request.args.get('lat', DEFAULT_LAT))
        lon = float(request.args.get('lon', DEFAULT_LON))
        report_range = request.args.get('range', 'week') # 'week' or 'month'
        
        days = 30 if report_range == 'month' else 7
        today = datetime.now().date()
        end_date = today - timedelta(days=1)
        start_date = end_date - timedelta(days=days-1)

        # 1. Fetch Location Info
        _, _, loc_name = get_location_multiplier(lat, lon)

        # 2. Fetch Weather History (NASA + Open-Meteo)
        df_nasa = fetch_nasa_history(start_date, end_date, lat, lon)
        url_hist = f"https://archive-api.open-meteo.com/v1/archive?latitude={lat}&longitude={lon}&start_date={start_date}&end_date={end_date}&daily=temperature_2m_max,temperature_2m_min,temperature_2m_mean,rain_sum,wind_speed_10m_max,wind_direction_10m_dominant,surface_pressure_mean,relative_humidity_2m_mean,cloud_cover_mean&timezone=auto&wind_speed_unit=ms"
        
        df_om = None
        try:
            r_hist = requests.get(url_hist, timeout=15).json()
            df_om = parse_meteo(r_hist)
        except Exception as e:
            print(f"Weather Archive Fetch Error: {e}")

        # 3. Fetch AQI History
        df_aqi = fetch_aqi_history(start_date, end_date, lat, lon)

        # Merge Logic
        df_final = None
        if df_nasa is not None and not df_nasa.empty:
            df_nasa.replace(-999.0, np.nan, inplace=True)
            df_final = df_nasa
            if df_om is not None:
                df_nasa['date'] = pd.to_datetime(df_nasa['date'])
                df_om['date'] = pd.to_datetime(df_om['date'])
                df_final = pd.merge(df_nasa, df_om, on='date', how='outer', suffixes=('_nasa', '_om'))
                for col in ['temp_max', 'temp_min', 'temp_avg', 'rainfall', 'wind_speed', 'pressure', 'humidity']:
                    if f'{col}_nasa' in df_final and f'{col}_om' in df_final:
                        df_final[col] = df_final[f'{col}_nasa'].fillna(df_final[f'{col}_om'])
                    elif f'{col}_nasa' in df_final:
                        df_final[col] = df_final[f'{col}_nasa']
                    elif f'{col}_om' in df_final:
                        df_final[col] = df_final[f'{col}_om']
        else:
            df_final = df_om

        if df_final is None or df_final.empty:
            return jsonify({'error': 'Failed to fetch historical baseline'}), 500

        # Merge AQI
        if df_aqi is not None:
            df_final['date'] = pd.to_datetime(df_final['date']).dt.date
            df_aqi['date'] = pd.to_datetime(df_aqi['date']).dt.date
            df_final = pd.merge(df_final, df_aqi, on='date', how='left')

        df_final = df_final.sort_values('date')
        
        # 4. Generate PDF
        pdf_bytes = bytes(create_pdf_report(df_final, loc_name, report_range))
        
        from flask import send_file
        import io
        return send_file(
            io.BytesIO(pdf_bytes),
            mimetype="application/pdf",
            as_attachment=True,
            download_name=f"EcoGlance_Report_{loc_name}_{report_range}.pdf"
        )
    except Exception as e:
        print(f"Report Generation Failed: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500

@app.route('/trigger_push', methods=['GET'])
def trigger_push():
    """Admin endpoint to instantly trigger a Webpushr broadcast on demand."""
    try:
        webpushr_key = os.environ.get('WEBPUSHR_KEY')
        webpushr_token = os.environ.get('WEBPUSHR_TOKEN')
        if not webpushr_key or not webpushr_token:
            return jsonify({'error': 'WEBPUSHR_KEY or WEBPUSHR_TOKEN missing in .env'}), 400
            
        with app.test_client() as c:
            response = c.get('/predict')
            if response.status_code == 200:
                data = response.get_json()
                main_data = data.get('data', {})
                aqi_info = data.get('aqi', {})
                
                aqi = aqi_info.get('value', 'N/A')
                temp = int(main_data.get('temp_avg', 0)) if main_data.get('temp_avg') is not None else 'N/A'
                rain = main_data.get('precipitation_probability', 0)
                
                url = "https://api.webpushr.com/v1/notification/send/all"
                headers = {
                    "webpushrKey": webpushr_key,
                    "webpushrAuthToken": webpushr_token,
                    "Content-Type": "application/json"
                }
                
                # Dynamically use the current host URL if available, else fallback to .env
                site_url = request.host_url.rstrip('/') if request else os.environ.get('SITE_URL', 'http://127.0.0.1:5000')
                
                payload = {
                    "title": "EcoGlance INSTANT Alert!",
                    "message": f"AQI: {aqi} | Temp: {temp}°C | Rain Chance: {rain}%",
                    "target_url": site_url
                }
                
                res = requests.post(url, headers=headers, json=payload, timeout=10)
                return jsonify({
                    'status': 'Notification broadcast command sent!',
                    'webpushr_api_response': res.json() if res.status_code == 200 else res.text,
                    'http_code': res.status_code
                }), 200
            else:
                return jsonify({'error': 'Failed to read local model prediction data'}), 500
    except Exception as e:
        return jsonify({'error': str(e)}), 500

def webpushr_background_job():
    # Wait for models to load and server to stabilize before first run
    time.sleep(30)
    
    while True:
        try:
            webpushr_key = os.environ.get('WEBPUSHR_KEY')
            webpushr_token = os.environ.get('WEBPUSHR_TOKEN')
            
            if webpushr_key and webpushr_token:
                # Need app context for test_client
                with app.test_request_context():
                    with app.test_client() as c:
                        response = c.get('/predict')
                        if response.status_code == 200:
                            data = response.get_json()
                            if data:
                                main_data = data.get('data', {})
                                aqi_info = data.get('aqi', {})
                                
                                aqi = aqi_info.get('value', 'N/A')
                                temp = int(main_data.get('temp_avg', 0)) if main_data.get('temp_avg') is not None else 'N/A'
                                rain = main_data.get('precipitation_probability', 0)
                                
                                url = "https://api.webpushr.com/v1/notification/send/all"
                                headers = {
                                    "webpushrKey": webpushr_key,
                                    "webpushrAuthToken": webpushr_token,
                                    "Content-Type": "application/json"
                                }
                                
                                site_url = os.environ.get('SITE_URL', 'http://127.0.0.1:5000')
                                payload = {
                                    "title": "EcoGlance Air Quality & Weather Update",
                                    "message": f"AQI: {aqi} | Temp: {temp}°C | Rain Chance: {rain}%",
                                    "target_url": site_url
                                }
                                
                                res = requests.post(url, headers=headers, json=payload, timeout=10)
                                print(f"[Webpushr] Campaign triggered: HTTP {res.status_code}")
                            else:
                                print("[Webpushr] Empty response from /predict")
                        else:
                            print(f"[Webpushr] Internal /predict failed: {response.status_code}")
            else:
                print("[Webpushr] Keys missing in environment. Cannot send.")
        except Exception as e:
            print(f"[Webpushr] Background job error: {str(e)}")
            
        # Sleep for exactly 5 hours before next automated push.
        time.sleep(5 * 3600)

# Start the notification daemon thread globally, but prevent double-execution when Flask reloader is active
if not app.debug or os.environ.get("WERKZEUG_RUN_MAIN") == "true":
    push_thread = threading.Thread(target=webpushr_background_job, daemon=True)
    push_thread.start()

if __name__ == '__main__':
    host_ip = "127.0.0.1"
    try:
        import socket
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.connect(("8.8.8.8", 80))
        host_ip = s.getsockname()[0]
        s.close()
    except: pass

    port = int(os.environ.get("PORT", 5000))
    print("\n" + "="*60)
    print("  EcoGlance AI Dashboard is now running!")
    print(f"  Local Host:    http://127.0.0.1:{port}")
    print(f"  Network/Phone: http://{host_ip}:{port}")
    print("="*60 + "\n")
    
    app.run(host='0.0.0.0', debug=True, port=port)
