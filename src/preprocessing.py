import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from typing import Tuple, List, Dict

class DataPreprocessor:
    def __init__(self, sequence_length: int = 14):
        self.sequence_length = sequence_length
        self.scaler_lstm = MinMaxScaler()
        self.scaler_weather = MinMaxScaler()
        self.scaler_targets = MinMaxScaler()

        # LSTM features (must match trained model - 10 features, NO season)
        # LSTM features (must match trained model - 12 features, NO season)
        self.lstm_features = [
            'temp_max', 'temp_min', 'temp_avg', 'humidity',
            'wind_speed', 'wind_direction', 'pressure', 'rainfall',
            'solar_radiation', 'cloud_cover', 'active_fires_count', 'fire_frp'
        ]

        # XGBoost features (includes season for better predictions)
        self.weather_features = [
            'temp_max', 'temp_min', 'temp_avg', 'humidity',
            'wind_speed', 'wind_direction', 'pressure', 'rainfall',
            'solar_radiation', 'cloud_cover', 'active_fires_count', 'fire_frp', 'season'
        ]

        # Targets (PMs will be log-transformed)
        self.pm_targets = ['pm10', 'pm2_5']

        self.target_columns = [
            'pm10', 'pm2_5',
            'temp_avg', 'temp_min', 'temp_max', 'humidity', 'rainfall', 'wind_speed'
        ]

        # Gas co-predictor feature names (added when hourly AQI data is used)
        self.gas_features = ['carbon_monoxide', 'nitrogen_dioxide', 'sulphur_dioxide', 'ammonia']
        # Monthly climatology of gas features (filled during process_hourly_data)
        self.gas_monthly_means: dict = {}

    def process_data(self, weather_path: str, combined_path: str):
        df_weather = pd.read_csv(weather_path)
        df_combined = pd.read_csv(combined_path)

        df_weather['date'] = pd.to_datetime(df_weather['date'])
        df_combined['date'] = pd.to_datetime(df_combined['date'])

        df_weather = df_weather.sort_values('date').reset_index(drop=True)
        df_combined = df_combined.sort_values('date').reset_index(drop=True)

        if 'humidity' in df_weather.columns:
            h_max = df_weather['humidity'].max()
            if h_max < 1.0:
                print(f"WARNING: Humidity max is {h_max}. May be Specific Humidity.")
            else:
                print(f"Humidity looks like Relative Humidity (Max: {h_max})")

        return df_weather, df_combined

    def process_hourly_data(self, aqi_hourly_path: str, weather_hourly_path: str, fire_data_path=None):
        """
        Build DAILY training frames from two separate hourly datasets:
        - AQI hourly CSV (targets + gas co-predictors source)
        - Weather hourly CSV (feature source)

        The two sources are kept separate and only aligned on date after
        independent aggregation.
        """
        # Load independently
        df_aqi_h = pd.read_csv(aqi_hourly_path)
        df_w_h = pd.read_csv(weather_hourly_path)

        # Parse datetime columns with tolerant matching
        def _pick_col(cols, candidates):
            lower_map = {str(c).strip().lower(): c for c in cols}
            for cand in candidates:
                if cand in lower_map:
                    return lower_map[cand]
            return None

        aqi_dt_col = _pick_col(df_aqi_h.columns, ['datetime', 'date_time', 'timestamp', 'date'])
        w_dt_col = _pick_col(df_w_h.columns, ['datetime', 'date_time', 'timestamp', 'date'])
        if aqi_dt_col is None:
            raise ValueError("AQI hourly dataset must contain a datetime-like column")
        if w_dt_col is None:
            raise ValueError("Weather hourly dataset must contain a datetime-like column")

        df_aqi_h = df_aqi_h.rename(columns={aqi_dt_col: 'datetime'})
        df_w_h = df_w_h.rename(columns={w_dt_col: 'datetime'})

        df_aqi_h['datetime'] = pd.to_datetime(df_aqi_h['datetime'], errors='coerce')
        df_w_h['datetime'] = pd.to_datetime(df_w_h['datetime'], errors='coerce')
        df_aqi_h = df_aqi_h.dropna(subset=['datetime']).sort_values('datetime')
        df_w_h = df_w_h.dropna(subset=['datetime']).sort_values('datetime')

        # --- Impute Nulls (Linear Interpolation for smooth weather/AQI transitions) ---
        # Interpolate numeric columns, then fill remaining edges
        df_aqi_numeric = df_aqi_h.select_dtypes(include=[np.number])
        if not df_aqi_numeric.empty:
            df_aqi_h[df_aqi_numeric.columns] = df_aqi_h[df_aqi_numeric.columns].interpolate(method='linear').bfill().ffill()

        df_w_numeric = df_w_h.select_dtypes(include=[np.number])
        if not df_w_numeric.empty:
            df_w_h[df_w_numeric.columns] = df_w_h[df_w_numeric.columns].interpolate(method='linear').bfill().ffill()

        # --- AQI daily aggregation (separate source) ---
        # Normalize AQI pollutant column names
        aqi_col_map = {str(c).strip().lower(): c for c in df_aqi_h.columns}
        pm25_src = aqi_col_map.get('pm2_5') or aqi_col_map.get('pm25') or aqi_col_map.get('pm2.5')
        pm10_src = aqi_col_map.get('pm10') or aqi_col_map.get('pm_10')
        if pm25_src is None or pm10_src is None:
            raise ValueError("AQI hourly dataset must contain PM2.5 and PM10 columns")
        df_aqi_h = df_aqi_h.rename(columns={pm25_src: 'pm2_5', pm10_src: 'pm10'})

        # Standardise gas co-predictor column names (strip leading/trailing whitespace)
        gas_rename = {}
        for col in df_aqi_h.columns:
            stripped = str(col).strip().lower()
            if stripped in ('carbon_monoxide', 'carbon monoxide', 'co'):
                gas_rename[col] = 'carbon_monoxide'
            elif stripped in ('nitrogen_dioxide', 'nitrogen dioxide', 'no2'):
                gas_rename[col] = 'nitrogen_dioxide'
            elif stripped in ('sulphur_dioxide', 'sulfur_dioxide', 'so2'):
                gas_rename[col] = 'sulphur_dioxide'
            elif stripped in ('ammonia', 'nh3'):
                gas_rename[col] = 'ammonia'
        df_aqi_h = df_aqi_h.rename(columns=gas_rename)

        df_aqi_h['date'] = df_aqi_h['datetime'].dt.floor('D')

        # Coerce pm and gas columns to numeric to avoid AggregationError on whitespace strings
        for col in ['pm10', 'pm2_5'] + [g for g in self.gas_features if g in df_aqi_h.columns]:
            df_aqi_h[col] = pd.to_numeric(df_aqi_h[col], errors='coerce')

        # Build aggregation dict: PM + available gas columns
        agg_dict = {'pm10': 'mean', 'pm2_5': 'mean'}
        for g in self.gas_features:
            if g in df_aqi_h.columns:
                agg_dict[g] = 'mean'
        df_aqi_d = df_aqi_h.groupby('date', as_index=False).agg(agg_dict)

        # Fill any missing gas columns with 0 so downstream code is uniform
        for g in self.gas_features:
            if g not in df_aqi_d.columns:
                df_aqi_d[g] = 0.0

        # --- Weather daily aggregation (separate source) ---
        w_col_map = {str(c).strip().lower(): c for c in df_w_h.columns}
        weather_req = {
            't2m': 'T2M',
            'rh2m': 'RH2M',
            'ws2m': 'WS2M',
            'wd2m': 'WD2M',
            'ps': 'PS',
            'prectotcorr': 'PRECTOTCORR'
        }
        rename_w = {}
        for raw_key, canonical in weather_req.items():
            src = w_col_map.get(raw_key)
            if src is None:
                raise ValueError(f"Weather hourly dataset missing required column: {canonical}")
            rename_w[src] = canonical
        df_w_h = df_w_h.rename(columns=rename_w)

        df_w_h['date'] = df_w_h['datetime'].dt.floor('D')

        # Circular mean for wind direction
        wd_rad = np.deg2rad(df_w_h['WD2M'].astype(float))
        df_w_h['_wd_sin'] = np.sin(wd_rad)
        df_w_h['_wd_cos'] = np.cos(wd_rad)

        w_agg = df_w_h.groupby('date', as_index=False).agg({
            'T2M': ['max', 'min', 'mean'],
            'RH2M': 'mean',
            'WS2M': 'mean',
            'PS': 'mean',
            'PRECTOTCORR': 'sum',
            '_wd_sin': 'mean',
            '_wd_cos': 'mean'
        })
        w_agg.columns = [
            'date',
            'temp_max', 'temp_min', 'temp_avg',
            'humidity', 'wind_speed', 'pressure', 'rainfall',
            '_wd_sin', '_wd_cos'
        ]

        # Convert circular mean back to [0, 360)
        w_agg['wind_direction'] = (
            np.degrees(np.arctan2(w_agg['_wd_sin'], w_agg['_wd_cos'])) + 360.0
        ) % 360.0
        w_agg = w_agg.drop(columns=['_wd_sin', '_wd_cos'])

        # Solar radiation proxy: daily temp range × 0.3 ≈ Hargreaves-style estimate (MJ/m²)
        w_agg['solar_radiation'] = (w_agg['temp_max'] - w_agg['temp_min']).clip(lower=0) * 0.3
        # Cloud cover proxy from humidity
        w_agg['cloud_cover'] = w_agg['humidity'].clip(lower=0, upper=100)

        # Seasonal feature
        w_agg['season'] = pd.to_datetime(w_agg['date']).dt.month.map(
            lambda m: 0 if m in [1, 2] else (1 if m in [3, 4, 5] else (2 if m in [6, 7, 8, 9] else 3))
        )

        # --- NASA FIRMS Fire Data Aggregation ---
        if fire_data_path and pd.io.common.file_exists(fire_data_path):
            df_fire = pd.read_csv(fire_data_path)
            # Acq_date in NASA FIRMS is YYYY-MM-DD
            df_fire['date'] = pd.to_datetime(df_fire['acq_date'])
            # We count rows per day (active_fires_count) and sum the FRP (Fire Radiative Power)
            fire_agg = df_fire.groupby('date', as_index=False).agg({
                'latitude': 'count',
                'frp': 'sum'
            }).rename(columns={'latitude': 'active_fires_count', 'frp': 'fire_frp'})
        else:
            fire_agg = pd.DataFrame(columns=['date', 'active_fires_count', 'fire_frp'])

        # Left merge onto weather, because weather is continuous. Days with no fired detected get 0.
        w_agg = pd.merge(w_agg, fire_agg, on='date', how='left')
        w_agg['active_fires_count'] = w_agg['active_fires_count'].fillna(0.0)
        w_agg['fire_frp'] = w_agg['fire_frp'].fillna(0.0)

        # Separate outputs
        df_weather = w_agg.copy()
        df_combined = pd.merge(w_agg, df_aqi_d, on='date', how='inner').sort_values('date').reset_index(drop=True)

        # Ensure target columns exist for training
        for target in self.target_columns:
            if target not in df_combined.columns:
                raise ValueError(f"Combined daily training frame missing target column: {target}")

        # Build monthly climatology for gas features (used at inference when live gas readings unavailable)
        df_combined['_month'] = pd.to_datetime(df_combined['date']).dt.month
        for g in self.gas_features:
            if g in df_combined.columns:
                self.gas_monthly_means[g] = df_combined.groupby('_month')[g].mean().to_dict()
        # Apply Lag/Rolling Engineered Features
        df_combined = self.add_engineered_features(df_combined)

        print(f"Hourly AQI rows: {len(df_aqi_h)}, hourly weather rows: {len(df_w_h)}")
        print(f"Daily weather rows: {len(df_weather)}, aligned daily training rows: {len(df_combined)}")
        print(f"Gas features available: {[g for g in self.gas_features if g in df_combined.columns]}")
        print(f"Total features after engineering: {len(df_combined.columns)}")

        return df_weather.reset_index(drop=True), df_combined.reset_index(drop=True)

    def add_engineered_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Add explicit historical "memory" features to supercharge tree-based models (XGBoost).
        If PM columns are absent (inference-time with weather-only data), fill them with
        monthly climatological defaults so lag/rolling features can still be computed.
        """
        df_engineered = df.copy()
        df_engineered = df_engineered.sort_values('date')

        # Provide live zero-fill fallbacks for missing fire features during inference
        if 'active_fires_count' not in df_engineered.columns:
            df_engineered['active_fires_count'] = 0.0
        if 'fire_frp' not in df_engineered.columns:
            df_engineered['fire_frp'] = 0.0

        # For inference-time data that doesn't have PM columns, create plausible defaults
        # using monthly climatological means stored from training.
        for col in ['pm2_5', 'pm10']:
            if col not in df_engineered.columns:
                # Use monthly mean if available, otherwise sensible constant
                if self.gas_monthly_means.get(col):
                    month_map = self.gas_monthly_means[col]
                    df_engineered[col] = pd.to_datetime(df_engineered['date']).dt.month.map(
                        lambda m: month_map.get(m, float(np.mean(list(month_map.values()))))
                    )
                else:
                    # Hardcoded Visakhapatnam regional averages as absolute fallback
                    df_engineered[col] = 40.0 if col == 'pm2_5' else 70.0

        # 1. Immediate History (Lags)
        for col in ['pm2_5', 'pm10']:
            if col in df_engineered.columns:
                df_engineered[f'{col}_lag_1'] = df_engineered[col].shift(1)
                df_engineered[f'{col}_lag_2'] = df_engineered[col].shift(2)

        # 2. Rolling Trends (Averages)
        for col in ['pm2_5', 'pm10', 'wind_speed', 'humidity', 'temp_max', 'rainfall']:
            if col in df_engineered.columns:
                df_engineered[f'{col}_rolling_3'] = df_engineered[col].rolling(window=3).mean()
                df_engineered[f'{col}_rolling_7'] = df_engineered[col].rolling(window=7).mean()

        # Handle NaNs created by lagging/rolling at the start of the dataset
        df_engineered = df_engineered.bfill()
        return df_engineered

    def apply_log_transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """Apply np.log1p to PM columns to enforce non-negativity"""
        df_log = df.copy()
        for col in self.pm_targets:
            if col in df_log.columns:
                df_log[col] = df_log[col].clip(lower=0)
                df_log[col] = np.log1p(df_log[col])
        return df_log

    def inverse_log_transform(self, predictions: np.ndarray, feature_indices: List[int]) -> np.ndarray:
        """Apply np.expm1 to PM columns"""
        preds_inv = predictions.copy()
        return np.expm1(preds_inv)

    def fit_scalers(self, df_weather: pd.DataFrame, df_combined: pd.DataFrame):
        """Fit MinMax Scalers on entire weather dataset"""
        df_weather = df_weather.copy()

        # Fit LSTM scaler (10 features without season)
        self.scaler_lstm.fit(df_weather[self.lstm_features])
        # Fit weather scaler (11 features with season)
        self.scaler_weather.fit(df_weather[self.weather_features])

        # Scale targets for LSTM stability
        target_data = df_combined[self.target_columns].values
        self.scaler_targets.fit(target_data)

    def create_sequences(self, df: pd.DataFrame, use_log_targets: bool = True) -> Tuple[np.ndarray, np.ndarray, pd.DataFrame]:
        """
        Returns:
        X_seq: (Samples, Seq_Len, Features) - Scaled Weather
        y_seq: (Samples, Targets) - Scaled Targets (Log-transformed if PM)
        meta_data: DataFrame with dates and unscaled targets for later matching
        """
        if use_log_targets:
            df_proc = self.apply_log_transform(df)
        else:
            df_proc = df.copy()

        X_data = df_proc[self.lstm_features]
        X_scaled = self.scaler_lstm.transform(X_data)

        y_data = df_proc[self.target_columns]
        y_scaled = self.scaler_targets.transform(y_data)

        X_seq, y_seq = [], []
        meta_indices = []

        for i in range(len(df) - self.sequence_length):
            X_seq.append(X_scaled[i : i + self.sequence_length])
            y_seq.append(y_scaled[i + self.sequence_length])
            meta_indices.append(i + self.sequence_length)

        return np.array(X_seq), np.array(y_seq), df.iloc[meta_indices].reset_index(drop=True)

    def prepare_xgb_data(self, lstm_embeddings: np.ndarray, meta_df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Prepare features for XGBoost.
        Input:
          - LSTM Embeddings (Latent features from past)
          - Future Weather (from meta_df / ground truth)
          - Time Features + Gas co-predictors
        Output:
          - X_xgb
          - y_xgb (Log-transformed targets)
        """
        # 1. Embeddings Frame
        emb_cols = [f'emb_{i}' for i in range(lstm_embeddings.shape[1])]
        df_emb = pd.DataFrame(lstm_embeddings, columns=emb_cols)

        df_meta = meta_df.reset_index(drop=True)

        features = pd.DataFrame()
        features = pd.concat([features, df_emb], axis=1)

        # Weather features (10, NO season)
        for col in self.lstm_features:
            features[col] = df_meta[col]

        # Time features
        df_meta['date'] = pd.to_datetime(df_meta['date'])
        features['month'] = df_meta['date'].dt.month
        features['day_of_week'] = df_meta['date'].dt.dayofweek
        features['day'] = df_meta['date'].dt.day
        features['is_weekend'] = features['day_of_week'].isin([5, 6]).astype(int)

        # Wind direction as cyclic features
        if 'wind_direction' in df_meta.columns:
            angle_rad = np.deg2rad(df_meta['wind_direction'])
            features['wind_dir_sin'] = np.sin(angle_rad)
            features['wind_dir_cos'] = np.cos(angle_rad)

        # Pressure tendency
        if 'pressure' in df_meta.columns:
            pressure_delta = df_meta['pressure'].diff().fillna(0.0)
            features['pressure_delta'] = pressure_delta

        # Gas co-predictor features (actual values from training; monthly climatology at inference)
        for g in self.gas_features:
            if g in df_meta.columns:
                features[g] = df_meta[g].values
            elif self.gas_monthly_means.get(g):
                month_means = self.gas_monthly_means[g]
                features[g] = df_meta['date'].dt.month.map(month_means).fillna(
                    np.mean(list(month_means.values()))
                ).values
            else:
                features[g] = 0.0

        # Engineered Memory Features
        engineered_cols = [col for col in df_meta.columns if '_lag_' in col or '_rolling_' in col]
        for col in engineered_cols:
            features[col] = df_meta[col].values

        # Targets (Log-transformed)
        df_log = self.apply_log_transform(df_meta)
        y_xgb = df_log[self.target_columns]

        return features, y_xgb
