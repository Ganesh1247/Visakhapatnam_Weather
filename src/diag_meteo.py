import requests
from datetime import datetime, timedelta

LAT = 17.6868
LON = 83.2185
SEQ_LENGTH = 14

def fetch_weather_data():
    end_date = datetime.now().date()
    start_date = end_date - timedelta(days=SEQ_LENGTH + 2) 
    
    url_hist = f"https://archive-api.open-meteo.com/v1/archive?latitude={LAT}&longitude={LON}&start_date={start_date}&end_date={end_date}&daily=temperature_2m_max,temperature_2m_min,temperature_2m_mean,rain_sum,wind_speed_10m_max,wind_direction_10m_dominant,shortwave_radiation_sum,surface_pressure_mean,relative_humidity_2m_mean,cloud_cover_mean&timezone=auto"
    print(f"Fetching History: {url_hist}")
    r_hist = requests.get(url_hist).json()
    
    url_fore = f"https://api.open-meteo.com/v1/forecast?latitude={LAT}&longitude={LON}&daily=temperature_2m_max,temperature_2m_min,temperature_2m_mean,rain_sum,wind_speed_10m_max,wind_direction_10m_dominant,shortwave_radiation_sum,surface_pressure_mean,relative_humidity_2m_mean,cloud_cover_mean&forecast_days=8&timezone=auto"
    print(f"Fetching Forecast: {url_fore}")
    r_fore = requests.get(url_fore).json()
    
    return r_hist, r_fore

hist, fore = fetch_weather_data()

hist_dates = hist.get('daily', {}).get('time', [])
fore_dates = fore.get('daily', {}).get('time', [])

print(f"Last Archive Date: {hist_dates[-1] if hist_dates else 'None'}")
print(f"First Forecast Date: {fore_dates[0] if fore_dates else 'None'}")
print(f"Archive Dates: {hist_dates}")
print(f"Forecast Dates: {fore_dates}")

if hist_dates and fore_dates:
    last_hist = hist_dates[-1]
    first_fore = fore_dates[0]
    if last_hist < first_fore:
        # Check gap
        lh = datetime.strptime(last_hist, '%Y-%m-%d')
        ff = datetime.strptime(first_fore, '%Y-%m-%d')
        gap = (ff - lh).days
        print(f"GAP DETECTED: {gap} days missing between {last_hist} and {first_fore}")
else:
    print("Error: Missing daily data in response.")
