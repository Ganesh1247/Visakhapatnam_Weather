import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from typing import Tuple, List, Dict

class DataPreprocessor:
    def __init__(self, sequence_length: int = 14):
        self.sequence_length = sequence_length
        self.scaler_lstm = MinMaxScaler()  # For LSTM features (10 features, no season)
        self.scaler_weather = MinMaxScaler()  # For XGBoost features (11 features, with season)
        self.scaler_targets = MinMaxScaler() # We might not need this for targets if using Log directly, but keeping for non-PM vars if needed? 
        # Actually for Log targets, we don't necessarily scale to 0-1, but XGB handles unscaled fine. LSTM might prefer scaled.
        # Let's scale *after* log transform for LSTM.
        
        # LSTM features (must match trained model - 10 features, NO season)
        self.lstm_features = [
            'temp_max', 'temp_min', 'temp_avg', 'humidity', 
            'wind_speed', 'wind_direction', 'pressure', 'rainfall',
            'solar_radiation', 'cloud_cover'
        ]
        
        # XGBoost features (includes season for better predictions)
        self.weather_features = [
            'temp_max', 'temp_min', 'temp_avg', 'humidity', 
            'wind_speed', 'wind_direction', 'pressure', 'rainfall',
            'solar_radiation', 'cloud_cover', 'season'
        ]
        
        # Targets (PMs will be log-transformed)
        self.pm_targets = ['pm10', 'pm2_5']
        
        self.target_columns = [
            'pm10', 'pm2_5', # Air Quality (Ozone removed)
            'temp_avg', 'temp_min', 'temp_max', 'humidity', 'rainfall', 'wind_speed' # Weather
        ]

    def process_data(self, weather_path: str, combined_path: str):
        df_weather = pd.read_csv(weather_path)
        df_combined = pd.read_csv(combined_path)
        
        df_weather['date'] = pd.to_datetime(df_weather['date'])
        df_combined['date'] = pd.to_datetime(df_combined['date'])
        
        df_weather = df_weather.sort_values('date').reset_index(drop=True)
        df_combined = df_combined.sort_values('date').reset_index(drop=True)
        
        # Humidity Check
        if 'humidity' in df_weather.columns:
            h_max = df_weather['humidity'].max()
            if h_max < 1.0: 
                print(f"WARNING: Humidity max is {h_max}. This might be Specific Humidity (QV2M). converting to % if possible or flagging.")
                # Simple heuristic: if < 1, multiply by 100? No, specific humidity is different.
                # Assuming dataset is already cleaned/prepped as RH based on earlier checks (max was > 1).
            else:
                print(f"Humidity looks like Relative Humidity (Max: {h_max})")

        return df_weather, df_combined

    def apply_log_transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """Apply np.log1p to PM columns to enforce non-negativity"""
        df_log = df.copy()
        for col in self.pm_targets:
            if col in df_log.columns:
                # Ensure no negative values before log (shouldn't be, but clip 0)
                df_log[col] = df_log[col].clip(lower=0)
                df_log[col] = np.log1p(df_log[col])
        return df_log

    def inverse_log_transform(self, predictions: np.ndarray, feature_indices: List[int]) -> np.ndarray:
        """Apply np.expm1 to PM columns"""
        preds_inv = predictions.copy()
        # This requires knowing which column is which. 
        # We'll handle this in the training script usually.
        return np.expm1(preds_inv)

    def fit_scalers(self, df_weather: pd.DataFrame, df_combined: pd.DataFrame):
        """Fit MinMax Scalers on entire weather dataset"""
        # Parse & sort
        df_weather = df_weather.copy()
        
        # Fit LSTM scaler (10 features without season)
        self.scaler_lstm.fit(df_weather[self.lstm_features])
        # Fit weather scaler (11 features with season)
        self.scaler_weather.fit(df_weather[self.weather_features])
        
        # Fit target scaler (Optional, if we want to scale targets for LSTM)
        # For this hybrid approach:
        # LSTM Targets: We can scale them 0-1 for stability.
        # XGB Targets: We can use Raw Log-Transformed values.
        
        # Let's scale everything for LSTM.
        target_data = df_combined[self.target_columns].values
        self.scaler_targets.fit(target_data)
        
    def create_sequences(self, df: pd.DataFrame, use_log_targets: bool = True) -> Tuple[np.ndarray, np.ndarray, pd.DataFrame]:
        """
        Returns:
        X_seq: (Samples, Seq_Len, Features) - Scaled Weather
        y_seq: (Samples, Targets) - Scaled Targets (Log-transformed if PM)
        meta_data: DataFrame with dates and unscaled targets for later matching
        """
        # 1. Log Transform PMs if needed
        if use_log_targets:
            df_proc = self.apply_log_transform(df)
        else:
            df_proc = df.copy()
            
        # 2. Scale Features (Weather)
        # Use LSTM features (10 without season) for consistency throughout pipeline
        X_data = df_proc[self.lstm_features]
        X_scaled = self.scaler_lstm.transform(X_data)
        
        # 3. Scale Targets (for LSTM stability)
        y_data = df_proc[self.target_columns]
        y_scaled = self.scaler_targets.transform(y_data)
        
        X_seq, y_seq = [], []
        meta_indices = []
        
        for i in range(len(df) - self.sequence_length):
            X_seq.append(X_scaled[i : i + self.sequence_length])
            y_seq.append(y_scaled[i + self.sequence_length]) # Next day target
            meta_indices.append(i + self.sequence_length)
            
        return np.array(X_seq), np.array(y_seq), df.iloc[meta_indices].reset_index(drop=True)

    def prepare_xgb_data(self, lstm_embeddings: np.ndarray, meta_df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Prepare features for XGBoost.
        Input: 
          - LSTM Embeddings (Latent features from past)
          - Future Weather (from meta_df / ground truth)
          - Time Features
        Output:
          - X_xgb
          - y_xgb (Log-transformed targets)
        """
        # 1. Embeddings Frame
        emb_cols = [f'emb_{i}' for i in range(lstm_embeddings.shape[1])]
        df_emb = pd.DataFrame(lstm_embeddings, columns=emb_cols)
        
        # 2. Future Weather & Time (Target Day)
        # meta_df contains the Row for the Target Day (t+1)
        # So we take weather from there directly (Assumption: we have good weather forecast)
        # In real inference, we use Open-Meteo forecast.
        
        df_meta = meta_df.reset_index(drop=True)
        
        features = pd.DataFrame()
        # Add Embeddings
        features = pd.concat([features, df_emb], axis=1)
        
        # Add Weather Forecast (Target Day)
        # Use lstm_features (10 features, NO season) to match what models were trained with
        # This ensures 45 total features: 32 embeddings + 10 weather + 3 time
        for col in self.lstm_features:
            features[col] = df_meta[col]
            
        # Add Time
        df_meta['date'] = pd.to_datetime(df_meta['date'])
        features['month'] = df_meta['date'].dt.month
        features['day_of_week'] = df_meta['date'].dt.dayofweek
        features['day'] = df_meta['date'].dt.day
        # Weekday / weekend flag (traffic / human activity proxy)
        features['is_weekend'] = features['day_of_week'].isin([5, 6]).astype(int)

        # Wind direction as cyclic features for XGBoost
        # (raw degrees are less informative because 0° ≈ 360°)
        if 'wind_direction' in df_meta.columns:
            angle_rad = np.deg2rad(df_meta['wind_direction'])
            features['wind_dir_sin'] = np.sin(angle_rad)
            features['wind_dir_cos'] = np.cos(angle_rad)

        # Pressure tendency (today − yesterday) as a simple dynamics feature
        if 'pressure' in df_meta.columns:
            # diff() will use previous row; first row gets 0
            pressure_delta = df_meta['pressure'].diff().fillna(0.0)
            features['pressure_delta'] = pressure_delta

        
        # Targets (Log-transformed)
        # apply log transform to meta_df raw values again to be sure we have the right regression targets
        df_log = self.apply_log_transform(df_meta)
        y_xgb = df_log[self.target_columns]
        
        return features, y_xgb
