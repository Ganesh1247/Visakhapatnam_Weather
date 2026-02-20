# EcoGlance Training Guide: High Precision AI Engine

This guide provides step-by-step instructions on how to re-train the EcoGlance AI models to achieve the highest possible accuracy and how to interpret the results.

## Prerequisites
Ensure you have the required dependencies installed (see `requirements.txt`) and that your datasets are located in the `data/` directory:
- `data/final_dataset.csv`
- `data/final_weather_dataset_2010-2025.csv`

## How to Start Training

To trigger a full re-training of the hybrid LSTM-XGBoost engine, run the following command from the project root:

```bash
python src/train_hybrid_model.py
```

### What happens during training?
1. **Stage 1: LSTM Training** - The "Neural Link" (LSTM) is trained to extract deep temporal patterns from weather and pollution history. We've increased this to 50 epochs for maximum sensitivity.
2. **Stage 2: XGBoost Training** - Specialized regression models are trained for each weather/air variable (PM2.5, PM10, Temp, etc.) using the neural embeddings.
3. **Stage 3: Evaluation** - The system automatically tests the new models against ground-truth data they haven't seen before.

## How to Know the Accuracy

Once training is complete, the script will output a **Summary Metrics Table** and a specific **Percentage Accuracy**.

### 1. Percentage Accuracy (AQI Class Accuracy)
This is the most intuitive metric for checking the model's performance. It measures how often the AI correctly predicts the correct AQI Category (e.g., "Good", "Moderate", "Poor").

Look for this line at the very end of the training log:
> `PM2.5 AQI Class Accuracy (Percentage): XX.XX%`

**What it means:**
- **> 85%:** Industry Standard (Excellent)
- **75% - 85%:** High Precision (Very Good)
- **65% - 75%:** Reliable (Good)

### 2. Scientific Metrics (RMSE, MAE, R2)
- **RMSE/MAE:** Lower is better. These represent the average error in µg/m³ or °C.
- **R2 Score:** Closer to 1.0 is better. It indicates how much of the data volatility the model explains.

## Where to Find Results

All results are automatically saved for your review:
- **Models:** Updated models are saved in the `models/` directory.
- **Accuracy Log:** The latest percentage accuracy is saved in `data/aqi_accuracy.txt`.
- **Metric CSV:** Detailed RMSE/MAE values are in `data/metrics_scientific.csv`.
- **Visual Plots:** Check the `plots/` folder for "True vs Predicted" graphs showing how closely the AI followed real-world trends over the last 60 days.

---
> [!TIP]
> To maintain "Higher Standards", re-train the models monthly as new weather data becomes available in your `data/` folder.
