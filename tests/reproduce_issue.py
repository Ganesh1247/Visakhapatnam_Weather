import requests
import time
import datetime

BASE_URL = "http://127.0.0.1:5000"

def test_predict():
    print("Testing /predict endpoint...")
    start_time = time.time()
    try:
        response = requests.get(f"{BASE_URL}/predict?method=mc_dropout", timeout=60)
        elapsed = time.time() - start_time
        print(f"Time taken: {elapsed:.2f}s")
        
        if response.status_code == 200:
            data = response.json()
            print("Success! /predict returned valid JSON.")
            return data
        else:
            print(f"Error: Status Code {response.status_code}")
            print(response.text)
            return None
    except Exception as e:
        print(f"Exception: {e}")
        return None

def test_hourly(date_str):
    print(f"Testing /hourly/{date_str} endpoint...")
    start_time = time.time()
    try:
        response = requests.get(f"{BASE_URL}/hourly/{date_str}", timeout=15)
        elapsed = time.time() - start_time
        print(f"Time taken: {elapsed:.2f}s")
        
        if response.status_code == 200:
            data = response.json()
            print(f"Success! /hourly returned {len(data)} data points.")
            return True
        else:
            print(f"Error: Status Code {response.status_code}")
            print(response.text)
            return False
    except Exception as e:
        print(f"Exception: {e}")
        return False

if __name__ == "__main__":
    print("--- Starting Reproduction Test ---")
    prediction_data = test_predict()
    
    if prediction_data:
        try:
            today_date = prediction_data['prediction_date'] # YYYY-MM-DD
            print(f"Prediction date: {today_date}")
            test_hourly(today_date)
        except KeyError:
             print("KeyError: 'prediction_date' not found in response.")
    else:
        print("Skipping /hourly test due to /predict failure.")
    print("--- Test Finished ---")
