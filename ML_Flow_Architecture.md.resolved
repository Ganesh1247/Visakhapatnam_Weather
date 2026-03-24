# EcoGlance Machine Learning Flow Architecture (Upgraded V2)

This document explains the end-to-end data processing, model training, and real-time inference pipeline powering the EcoGlance application.

---

## 1. Data Sources & Preprocessing ([src/preprocessing.py](file:///c:/Users/koila/DTI.FINAL/src/preprocessing.py))
The pipeline runs on a dense integration of historical APIs, real-time satellite data, and proxy logic.

### A. Raw Data Ingestion
1. **Weather Data:** Historical (10 years) from Open-Meteo, including parameters like Temperature, Humidity, Wind Speed/Direction, Pressure, Rainfall, and Solar Radiation.
2. **Air Quality (AQI) Data:** Historical PM2.5, PM10, CO, NO2, SO2, and NH3 values from Open-Meteo Air Quality archive.
3. **Satellite Fire Data (NASA FIRMS):** Tracking active fires and Fire Radiative Power (FRP) in the surrounding regional bounding box.
4. **Traffic Proxies:** A simulated traffic congestion index modeled around Vizag's local diurnal commute peaks.

### B. Advanced Feature Engineering
Before the data hits the models, it passes through physical heuristics:
- **Wind Transport Features:** `pm2_5_lag_1 * wind_speed` represents real-time incoming pollution. 
- **Directional Vectors:** Wind direction is converted to `Sin` and `Cos` components.
- **Atmospheric Stability Index:** Uses `wind_speed / (temp_max - temp_min + 1)` to model inversion layers that trap surface smog.
- **Long-Term Memory:** Lags (`_lag_1` up to `_lag_30`) and Rolling Averages (`_rolling_3` up to `_rolling_30`) capture deep seasonal momentum without leaking future data.

*Note: All PM targets (particulate matter) are Log-Transformed (`np.log1p`) to compress the massive variance of severe smog spikes, stabilizing gradient descent during neural network training.*

---

## 2. The Multi-Task Neural-Tree Architecture ([src/train_hybrid_model.py](file:///c:/Users/koila/DTI.FINAL/src/train_hybrid_model.py))
EcoGlance utilizes a heavily upgraded **Multi-Task Neural-Tree Chain** to generate predictions.

### A. Multi-Task LSTM Encoder (The Feature Extractor)
Neural networks are excellent at sequence pattern recognition but struggle with hard Tabular outputs.
1. The model takes a **21-Day rolling sequence window** of raw weather/pollution history.
2. It passes through a multi-task `LSTM (128 -> 64)` structure with 4 independent gradient output heads (predicting PM2.5, PM10, Temp, Humidity simultaneously).
3. **The Trick:** We ignore the literal outputs! We slice the network at the final 64-dimensional layer to extract "Latent Embeddings"—a dense numerical representation of the timeline's "mood".

### B. The Ensemble Regressors (LightGBM + XGBoost Residuals)
Those 64-dimensional embeddings (plus all the engineered stability/transport features) are fed into a dual-model ensemble pipeline, thoroughly optimized via TimeSeries splits and Optuna:

1. **The Base Learner (LightGBM):** Fast, leaf-wise gradient boosting constructs the primary forecast matrix.
2. **The Error Corrector (XGBoost Residuals):** An XGBoost regressor is trained *exclusively* on `y_true - y_lightgbm_pred`. 
3. **Final Formula:** `Prediction = LightGBM(features) + XGBoost(Residuals)`.
4. Values are inverse-log-transformed (`np.expm1`) back into raw micrograms/Celsius.

---

## 3. Real-Time Inference Backend ([src/app.py](file:///c:/Users/koila/DTI.FINAL/src/app.py))
When a user visits EcoGlance, [predict()](file:///c:/Users/koila/DTI.FINAL/src/app.py#784-1066) orchestrates the real-time AI logic.

### A. Dynamic Data Fetching
1. The backend fetches a **35-day historical buffer** ending yesterday (to properly calculate 30-day biological lags online) utilizing NASA+OpenMeteo hybrid merging.
2. It fetches a 7-day realtime Open-Meteo Weather forecast.

### B. "Live Now" Override
For the very first data point (Today/Now), AI hallucination is dangerous for public health. EcoGlance fetches the exact physical sensor readings for the current hour, overriding the AI output structure dynamically inside `render_template`.

### C. Live Inference Pass
The engine builds the 64-dimensional sequence for the upcoming 7 days, executes the LightGBM step, adds the XGBoost Residual correction, and dynamically applies a learned **Seasonal Bias Corrector** matrix to nudge predictions based on the month (Winter heating vs Summer rains).

### D. The Synthesis of the Diurnal Curve
Since tree models return *Daily Aggregates* (Max/Min), asking the backend to run 24 * 7 independent hour-by-hour inferences would crash most standard server topologies. Instead:
- `app.route('/hourly')` wraps a mathematical algorithmic Sine-wave generator tightly around the predicted daily constraints (e.g., oscillating tightly between predicted Min 28c and Max 35c), mimicking sunrise and sunset drops realistically while rendering interactive charts.

---

## 4. Predictions & Accuracy Metrics
The recent Multi-task Ensemble transition massively improved accuracy without over-parameterization. Based on the latest real-world test sets ([metrics_scientific.csv](file:///c:/Users/koila/DTI.FINAL/metrics_scientific.csv)):

- **Weather Forecasting (Temperature, Humidity, Wind):** Near perfect tracking.
  - **Temperature (Avg, Min, Max):** R² score of **~0.99**.
  - **Humidity & Wind Speed:** R² score of **~0.97 - 0.99**.
  - **Rainfall:** Very strong capability with an R² score of **~0.77**.

- **Air Quality (Pollution) Forecasting:** 
  - **PM10:** Achieves an R² score of **~0.70**, tracking heavy dust accurately.
  - **PM2.5:** Improved massively from a baseline of ~0.28 to an R² score of **~0.53** (43% relative variance improvement!). MAE sits roughly at `10.06`, establishing robust environmental tracking suitable for localized health alerts.

---

## 5. Core System Innovations

1. **Multi-Task Sequence Distillation:** Extracting 64-dim embeddings from an independent Multi-task LSTM network directly feeds the tabular regressors chronological understanding that gradient-boosted trees physically cannot calculate themselves.
2. **Residual Error Learning:** LightGBM handles broad-stroke generalization; XGBoost hyper-focuses on the mistakes.
3. **Physical Atmospheric Proxies:** Instead of forcing Deep Learning to discover physics organically, the preprocessor hard-codes `pollution_transport` (Wind * PM2.5 Lag1) and `stability_index` into the matrix.
4. **Dynamic Climatology & Gas Co-predictors:** When live tracking for specific gas components (CO, NO2) drops offline, the system rolls over seamlessly into learned monthly climatological means—preventing inference crashing while keeping forecasts regionally sensible.
5. **Algorithmic Diurnal Curve Synthesizer:** Circumvents massive compute costs by rendering 168-hour timeline charts (24 * 7) mathematically from 7 aggregate node anchors natively in Python.
