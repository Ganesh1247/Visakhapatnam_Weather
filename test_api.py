import requests
from datetime import datetime
import pytz

now_local = datetime.now(pytz.timezone('Asia/Kolkata'))
current_hour_str = now_local.strftime('%Y-%m-%dT%H:00')

url = 'https://api.open-meteo.com/v1/forecast?latitude=17.72&longitude=83.28&hourly=temperature_2m,relative_humidity_2m,wind_speed_10m&timezone=auto&wind_speed_unit=kmh'
r = requests.get(url).json()

try:
    hourly = r.get('hourly', {})
    idx = hourly['time'].index(current_hour_str)
    print(f'Time: {current_hour_str}')
    print(f"Temp: {hourly['temperature_2m'][idx]} C")
    print(f"Humidity: {hourly['relative_humidity_2m'][idx]} %")
    print(f"Wind Speed: {hourly['wind_speed_10m'][idx]} km/h")
except Exception as e:
    print('Error:', e)
