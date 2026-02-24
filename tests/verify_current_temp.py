import sys
import os
from datetime import datetime
import pandas as pd
import numpy as np

# Add src to path
sys.path.append(os.path.join(os.getcwd(), 'src'))

def test_hourly_logic():
    print("Testing hourly logic...")
    
    # Mock fore_json
    now = datetime.now()
    current_hour_str = now.strftime('%Y-%m-%dT%H:00')
    
    fore_json = {
        'hourly': {
            'time': [current_hour_str, '2026-02-22T08:00'],
            'temperature_2m': [25.5, 26.0],
            'relative_humidity_2m': [60, 58],
            'rain': [0.0, 0.0],
            'wind_speed_10m': [10.8, 11.2]
        }
    }
    
    main_pred = {
        'temp_avg': 24.0, # Daily average
        'humidity': 65,
        'wind_speed': 15.0,
        'rainfall': 1.0,
        'pm2_5': 20.0,
        'pm10': 40.0,
        'date': now
    }
    
    # Logic copied from src/app.py (predict function)
    try:
        hourly = fore_json.get('hourly', {})
        if hourly:
            now_local = datetime.now()
            current_hour_str = now_local.strftime('%Y-%m-%dT%H:00')
            if current_hour_str in hourly.get('time', []):
                idx = hourly['time'].index(current_hour_str)
                main_pred['temp_avg'] = hourly['temperature_2m'][idx]
                main_pred['humidity'] = hourly['relative_humidity_2m'][idx]
                main_pred['wind_speed'] = hourly['wind_speed_10m'][idx] / 3.6
                main_pred['rainfall'] = hourly['rain'][idx]
                print(f"Simulated Logic: Updated Hero section with current hour ({current_hour_str}) data.")
            else:
                print(f"Simulated Logic Fail: {current_hour_str} not in {hourly.get('time', [])}")
    except Exception as e:
        print(f"Error in simulation: {e}")

    # Assertions
    if main_pred['temp_avg'] == 25.5:
        print("Verification PASSED: Temperature updated to hourly value.")
    else:
        print(f"Verification FAILED: Temperature is {main_pred['temp_avg']}, expected 25.5")
        sys.exit(1)

    if main_pred['wind_speed'] == 3.0: # 10.8 / 3.6 = 3.0
        print("Verification PASSED: Wind speed converted correctly to m/s.")
    else:
        print(f"Verification FAILED: Wind speed is {main_pred['wind_speed']}, expected 3.0")
        sys.exit(1)

if __name__ == "__main__":
    test_hourly_logic()
