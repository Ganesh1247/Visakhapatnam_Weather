
---
title: EcoGlance
colorFrom: green
colorTo: blue
sdk: docker
app_port: 7860
pinned: false
---

# EcoGlance - Air Quality & Weather Forecasting System

![Python](https://img.shields.io/badge/Python-3.8%2B-blue)
![Flask](https://img.shields.io/badge/Flask-2.0%2B-green)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.0%2B-orange)
![XGBoost](https://img.shields.io/badge/XGBoost-1.0%2B-red)

EcoGlance is a sophisticated air quality and weather forecasting application that leverages hybrid machine learning models (LSTM + XGBoost) and uncertainty quantification techniques (MC Dropout, Conformal Prediction, Quantile Regression) to provide accurate and reliable environmental predictions.

## 🌟 Key Features

- **Hybrid AI Core**: Combines LSTM for temporal pattern extraction with XGBoost for robust regression.
- **Uncertainty Quantification**: Provides confidence intervals using MC Dropout, Conformal Prediction, and Quantile Regression.
- **Micro-Climate Analysis**: Personalized forecasts for specific coordinates.
- **Interactive Dashboard**: Modern, responsive UI with real-time data visualization.
- **Secure Authentication**: OTP-based email verification system.

## 🛠️ Technology Stack

- **Backend**: Python, Flask, SQLite
- **AI/ML**: TensorFlow/Keras (LSTM), XGBoost, Scikit-learn
- **Frontend**: HTML5, CSS3, JavaScript (Chart.js for visualizations)
- **Data Source**: Open-Meteo API, NASA Power (for historical training data)

## 📂 Project Structure

```bash
EcoGlance/
├── data/               # Datasets and SQLite database
├── models/             # Trained ML models (.h5, .pkl)
├── notebooks/          # Analysis and experiment notebooks
├── src/                # Source code
│   ├── app.py          # Main application entry point
│   ├── auth.py         # Authentication logic
│   ├── preprocessing.py # Data processing pipeline
│   └── backend/        # uncertainty modules
├── static/             # CSS, JS, Images
├── templates/          # HTML templates
└── requirements.txt    # Project dependencies
```

## 🚀 Getting Started

### Prerequisites
- Python 3.8+
- Git

### Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/yourusername/EcoGlance.git
   cd EcoGlance
   ```

2. **Create a virtual environment**
   ```bash
   python -m venv .venv
   # Windows
   .venv\Scripts\activate
   # Linux/Mac
   source .venv/bin/activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Run the Application**
   ```bash
   python src/app.py
   ```
   Access the dashboard at `http://127.0.0.1:5000`.

## 🌐 Deployment

### Deploying to Render

The application is configured for easy deployment on Render's free tier.

**⚠️ Important:** On the free tier, the database is **ephemeral** (wiped on each deployment). Users must re-register after each update.

For detailed deployment instructions and limitations, see [`RENDER_DEPLOYMENT.md`](RENDER_DEPLOYMENT.md).

**Quick Deploy:**
1. Push code to GitHub
2. Connect repository on Render
3. Render auto-detects Python and uses `render.yaml` configuration
4. Access your app at `https://your-app.onrender.com`

**Upgrade to paid plan (~$7/month)** to enable persistent storage and keep user data across deployments.

## 📊 Model Performance

Our hybrid model achieves state-of-the-art performance for local forecasting:

| Model Component | Role | accuracy (R²) |
|-----------------|------|---------------|
| LSTM | Feature Extraction from Time Series | - |
| XGBoost | Regression on LSTM Embeddings | > 0.85 |
| Conformal | Uncertainty Calibration | 90% Coverage |

## 🔮 Future Improvements

- [ ] Integration with more localized sensor networks.
- [ ] Mobile application (React Native).
- [ ] Real-time satellite data ingestion.

---

Developed with ❤️ by the EcoGlance Team.
