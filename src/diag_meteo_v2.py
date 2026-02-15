import requests
import sys
from datetime import datetime, timedelta

LAT = 17.6868
LON = 83.2185
SEQ_LENGTH = 14

def test_api(url):
    print(f"\nTesting URL: {url}")
    try:
        response = requests.get(url, timeout=10)
        print(f"Status Code: {response.status_code}")
        if response.status_code == 200:
            return response.json()
        else:
            print(f"Error Content: {response.text[:200]}")
    except Exception as e:
        print(f"Request failed: {e}")
    return None

end_date = datetime.now().date()
# Open-Meteo Archive usually has a 2-day delay. Today is Feb 11.
# Let's try to get data until Feb 9.
safe_end_date = end_date - timedelta(days=2)
start_date = safe_end_date - timedelta(days=SEQ_LENGTH + 2)

url_hist = f"https://archive-api.open-meteo.com/v1/archive?latitude={LAT}&longitude={LON}&start_date={start_date}&end_date={safe_end_date}&daily=temperature_2m_max,temperature_2m_min,temperature_2m_mean,rain_sum,wind_speed_10m_max,wind_direction_10m_dominant,shortwave_radiation_sum,surface_pressure_mean,relative_humidity_2m_mean,cloud_cover_mean&timezone=auto"
url_fore = f"https://api.open-meteo.com/v1/forecast?latitude={LAT}&longitude={LON}&daily=temperature_2m_max,temperature_2m_min,temperature_2m_mean,rain_sum,wind_speed_10m_max,wind_direction_10m_dominant,shortwave_radiation_sum,surface_pressure_mean,relative_humidity_2m_mean,cloud_cover_mean&forecast_days=8&timezone=auto"

hist_data = test_api(url_hist)
fore_data = test_api(url_fore)

if hist_data:
    dates = hist_data.get('daily', {}).get('time', [])
    print(f"History Dates: {dates[0]} to {dates[-1]} (Count: {len(dates)})")

if fore_data:
    dates = fore_data.get('daily', {}).get('time', [])
    print(f"Forecast Dates: {dates[0]} to {dates[-1]} (Count: {len(dates)})")

if hist_data and fore_data:
    last_hist = hist_data['daily']['time'][-1]
    first_fore = fore_data['daily']['time'][0]
    print(f"\nTransition: {last_hist} -> {first_fore}")
    if last_hist < first_fore:
        gap = (datetime.strptime(first_fore, '%Y-%m-%d') - datetime.strptime(last_hist, '%Y-%m-%d')).days
        print(f"GAP DETECTED: {gap} days")
