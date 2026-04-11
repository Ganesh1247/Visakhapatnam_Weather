# EcoGlance AI System Architecture & Reconstruction Guide

This guide is designed for a senior engineer to rebuild the EcoGlance full-stack web application from the ground up. It covers the exact details from data architecture and AI modeling to API deployment and frontend UI composition.

---

## 1. PROJECT OVERVIEW
*   **Project Title:** EcoGlance – Predictive AI Weather & Air Quality Platform
*   **Problem Statement:** Standard weather APIs rely exclusively on macro-level meteorological modeling and fail to account for hyper-local human-induced urban phenomena like commute hour smog accumulation, thermal trapping (inversions), or cross-wind pollution transport.
*   **Objectives and Goals:** Create a mathematically robust, visually stunning, fault-tolerant platform that merges Neural-Networks with gradient boosted trees to deliver localized, high-accuracy AQI and climate forecasts.
*   **Real-world Use Case:** Public health safeguarding. Providing automated Web-Push and SMS alerts to citizens offering actionable advice before hazardous smog spikes.
*   **System Architecture Diagram:** 
    `[Scheduled Tasks/Sensor Data] -> [Pandas Preprocessing & Feature Engineering] -> [Multi-Task LSTM Emdedder] -> [LightGBM/XGBoost Ensemble] -> [Flask Backend Context] -> [REST APIs] -> [Vanilla JS/Chart JS Frontend] -> [Twilio/Webpushr UI & Alerts]`

---

## 2. DATA COLLECTION
*   **Data Sources:**
    *   **NASA POWER API (Primary 16-Year Dataset):** The historic 16 years of weather data was primarily sourced from NASA, forming the core foundational dataset for long-term climatic modeling.
    *   **Open-Meteo Archive & Forecast APIs:** Acts alongside NASA to provide Hourly/Daily API data for Temperature, Rainfall, Wind, UV, Humidity, and core AQI parameters (PM10, PM2.5, SO2, NO2).
    *   **NASA FIRMS:** Satellite telemetry representing active fire counts and Fire Radiative Power (FRP), crucial for predicting massive regional smoke spikes.
    *   **Simulated Traffic Proxy:** Derived CSV matching local diurnal commute distributions to add time-series variance representing congestion.
*   **Data Schema:** Time-series rows consisting of `date/datetime`, `.temp_max/min/avg`, `humidity`, `wind_speed`, `wind_direction` (later decoupled to `sin`/`cos`), `pressure`, `rainfall`, target variables (`pm2_5`, `pm10`), and trace gases.
*   **Data Frequency:** Hybrid. Weather variables are requested as Daily Aggregates, whilst API fallbacks request Hourly sequences scaled back to daily.
*   **Storage Format:** Persisted locally as `.csv` artifacts within `/data`. User account data resides in `users.db` (SQLite) / Supabase (Postgres).
*   **Challenges Faced:** Dealing with API errors (`-999` flags from NASA), and sensor downtime for trace gases. 

---

## 3. DATA PREPROCESSING
Managed explicitly within `DataPreprocessor` classes (`src/preprocessing.py`).

*   **Data Cleaning steps:** Null fields undergo backward-forward fill logic (`bfill().ffill()`) leveraging linear interpolation. NASA `-999` error codes are cast as `np.nan` before cleaning.
*   **Feature Engineering:**
    *   **Atmospheric Stability Index:** Derived via `wind_speed / ((temp_max - temp_min) + 1)`. Models inversion layers (heat trapping smog on the surface).
    *   **Pollution Transport:** Derived via `pm25_lag_1 * wind_speed` to signify incoming transported debris.
    *   **Lag & Rolling Memories:** Hardcoded rolling averages (3d, 7d, 14d) and lags (up to 30d) implemented natively in Pandas to supply tabular boosted trees with explicit memory context.
    *   **Climatological Monthly Imputation:** A dynamic Python dictionary computes missing trace gas data dynamically based on Historical Monthly Means.
*   **Data Transformation:** Targets are passed through `np.log1p()` (Log Transformation) to squash massive smog spikes variance. Features are scaled using `MinMaxScaler()`.
*   **Sequence Preparation:** The system constructs rolling 3D Tensors of `SEQ_LENGTH = 21` (21 chronological days of history acting as input nodes).
*   **Train-Test Split Strategy:** Ordered `TimeSeriesSplit` 80-20 validation to prevent data leakage from the future.

---

## 4. MACHINE LEARNING MODELS
Trained via `src/train_hybrid_model.py`.

*   **Models Used:** Multi-task LSTM Learner (TensorFlow), LightGBM Regressor Base, XGBoost Regressor Residual.
*   **Why Each Model?** LSTMs are king at extracting sequence semantics but often underperform standalone gradient boosted trees on tabular targets. Therefore, LSTM output gradients are discarded, and an intermediate deep embedding layer is passed natively into Tabular Trees (LGBM) to get the "best of both worlds".
*   **Model Architecture:** 
    *   `Input Shape (Batch_Size, 21, 13 features)` -> 
    *   `LSTM(128)` -> `Dropout(0.2)` -> 
    *   `LSTM(64)` (The Embedding output cut) -> 
    *   Passed laterally to LightGBM + XGBoost.
*   **Training Process:** 
    *   LSTM uses Adam Optimizer, MSE loss, 80 Epochs natively but triggers `EarlyStopping(patience=5)` to prevent overfitting.
    *   LightGBM utilizes an **Optuna** tuning study to auto-search hyperparameter meshes (`n_estimators`, `max_depth`, `learning_rate`).
    *   XGBoost explicitly targets training the *Residual Error Margin* (`y_train - lgbm_predictions_train`).
*   **Evaluation Metrics:** MAE, R² score, RMSE, Mean Absolute Percentage Error (MAPE), evaluated across Summer vs Winter splits.
*   **Model Integration:** Formula: `Prediction Output = np.expm1( LightGBM(Features) + XGBoost(Residuals) )`.

---

## 5. MODEL INTEGRATION (INFERENCE LOOP)
*   Loaded in memory instantly inside `app.py`. Thread locks `threading.Lock()` ensure thread-safe single-instance lazy loading of `.h5` and `.pkl` instances.
*   **Inference Pipeline:**
    1.  User accesses `/predict`.
    2.  `fetch_weather_data()` fetches recent 21 days API context.
    3.  `preprocessor.create_sequences()` embeds logs.
    4.  LSTM slices a 64-dim `X_all_emb` array.
    5.  XGB/LightGBM outputs the JSON schema.
*   **Handling Real-Time:** 
    AI hallucinates immediate 0-hour live data. The backend intercepts the very first index array node and overwrites AI metrics directly with Ground-Truth physical readings from local sensors for absolute reliability. 

---

## 6. BACKEND DEVELOPMENT
*   **Tech Stack:** Python, Flask (Sync router wrapping heavily vectorized Pandas logic). 
*   **API Endpoints:**
    *   `GET /predict`: Executes ML execution block, returns massive schema comprising daily matrices.
    *   `GET /hourly/<date>`: **Crucial Architecture Point**. Computes a localized Sine Wave oscillation constrained between ML Daily minimums and maximums mimicking Diurnal curves natively. Eliminates server crash potential of computing hourly ML inferences matrix dynamically.
    *   `POST /login`, `/signup`: Passes to passlib hashed functions linking to SQLite.
*   **Database Integration:** Hybrid methodology prioritizing SQLite `users.db` locally but utilizing a synchronized `.sql` schema mapping (`supabase_schema.sql`) for scalable Postgres connections (`auth.py`).

---

## 7. FRONTEND DEVELOPMENT
*   **Tech Stack:** Native HTML5, heavily decoupled Vanilla JS, Utility-first Vanilla CSS (mimicking Tailwind).
*   **UI Components:**
    *   Vibrant, dynamic glassmorphic backgrounds scaling natively based on JSON payload weather codes (e.g., dynamically injecting rainy UI classes).
    *   Gauge vectors manipulated by SVG `stroke-dashoffset` computations in Vanilla JS. 
*   **Data Charting:** Employs Chart.js configured for deep dark-mode themes referencing the REST API endpoints arrays directly upon window-load.
*   **Interaction:** DOM relies on asynchronous JSON `fetch()` injections with skeleton-loaded fallbacks while background inferences settle.

---

## 8. NOTIFICATIONS / AUTOMATION
Controlled via `sms_alerts.py`.
*   **Push System Framework:** Twilio REST Protocol (SMS/WhatsApp endpoints) intersecting with Webpushr V1 JSON Payload web hooks.
*   **Trigger Configurations:** Time-locked polling routines via `APScheduler` or Fallback Threads. The notifications are received periodically every 5 hours to keep users consistently updated throughout the day.
*   **Threshold Automation:** Logic structures automatically change string composition flags to append "Avoid activity" or "Stay indoors" based dynamically on whether predictive thresholds map over ~100+ AQI boundaries.

---

## 9. DEPLOYMENT STRUCTURE
*   **Environments Target:** Render natively (utilizing `render.yaml`) / Hugging Face Spaces / Google Cloud Run.
*   **CI/CD Constraints:** Deployed natively via `Dockerfile` or preconfigured PS1 run scripts (`deploy_hf.ps1`). The entire ML pipeline collapses and commits as `Pickle` and `h5` binaries stored statically mapped by `.gitattributes` inside Git LFS constraints.
*   **Scalability Consideration:** Gunicorn threads bind natively to 0.0.0.0. Latency bottlenecks are averted effectively by offloading temporal interpolation computations dynamically to the front-end JS client DOM where possible.

---

## 10. PROJECT DIRECTORY MAPPING
```bash
/c/Users/.../DTI.FINAL
 ├── src/
 │   ├── app.py                  # Core backend Flask interface
 │   ├── preprocessing.py        # Central data normalizer & engineering
 │   ├── train_hybrid_model.py   # Executable pipeline for model retraining
 │   ├── sms_alerts.py           # Twilio & Push routine scheduler
 │   └── auth.py                 # SQLite/Supabase session integration
 ├── models/
 │   ├── lgbm_xgb_ensemble.pkl   # Serialized tree artifacts
 │   └── lstm_hybrid_chain.h5    # Compiled neural embedding matrices
 ├── data/
 │   ├── vizag_aqi_hourly.csv    # Live sensor fallbacks
 │   └── users.db                # Encrypted Local account registry
 ├── templates/
 │   └── index.html              # Core Monolithic frontend view
 └── Dockerfile                  # Application runtime specifications
```

---

## 11. COMPLETE IMPLEMENTATION GUIDE (Rebuild)

### Step 1: Environment Provisioning
Initialize local environment isolation to protect global OS variables.
```bash
python -m venv .venv
source .venv/scripts/activate  # (Windows: .venv\Scripts\activate)
pip install -r requirements.txt
```

### Step 2: Assemble Training Matrices 
To retrain or initialize the system from zero, verify or download localized `.csv` files into `/data`. Check `train_hybrid_model.py` for file dependencies.
```bash
python src/train_hybrid_model.py
```
> *Error Fix: If TensorFlow complains about AVX or DNN nodes on Windows, set `os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'` within runtime environments.*

### Step 3: Local SQL Registration
Ensure `data/` exists, the backend natively populates `users.db` via `auth.py` on first launch automatically.
If utilizing Supabase, execute `supabase_schema.sql` inside your PGAdmin dashboard.

### Step 4: Boot Flask Microservices
Load backend processes synchronously. 
```bash
flask run --app src/app.py --port 5000
```
Visit `http://localhost:5000` to visualize the stack. Twilio notifications will log as failures quietly if `.env` context keys are missing, but will not corrupt application uptime.

---

## 12. IMPROVEMENTS & FUTURE ARCHITECTURE

If planning to present this system to Senior tech panels, highlight these roadmap advancements:

1.  **Migrate the Frontend to Next.js (React):** The current `index.html` structure (over 2k lines) should be abstracted into declarative components (`<GaugeCard/>`, `<DailyCarousel/>`) for testable, uncoupled UI management.
2.  **Streaming WebSockets:** Swap `REST /predict` endpoint for `Socket.io` hooks, enabling instantaneous, partial UI diffing as predictions complete, preventing sequential fetching overhead.
3.  **Real-Time Live-Fire API:** Deprecate static `.csv` tracking for NASA active markers and implement immediate webhook intersections tracing live NASA feeds in local bounds to dynamically shift features dynamically based on hyper-local localized anomalies. 
4.  **Hardware Injections:** Extend the system integration layers toward direct ESP32 IoT localized nodes distributed around the specific city to reduce dependency on sweeping meteorological proxy arrays entirely.
