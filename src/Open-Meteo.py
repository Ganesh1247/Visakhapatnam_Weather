import requests
import pandas as pd

# LOCATION – Visakhapatnam
LAT = 17.6868
LON = 83.2185

# DATE RANGE
START = "2016-01-01"
END = "2016-12-31"

url = (
    "https://air-quality-api.open-meteo.com/v1/air-quality?"
    f"latitude={LAT}&longitude={LON}"
    f"&start_date={START}&end_date={END}"
    "&hourly=pm10,pm2_5,carbon_monoxide,nitrogen_dioxide,ozone,sulphur_dioxide"
    "&timezone=Asia/Kolkata"
)

print("Downloading data...")
response = requests.get(url)
data = response.json()

# Convert to DataFrame
df = pd.DataFrame(data["hourly"])

# Convert time column
df["time"] = pd.to_datetime(df["time"])

# Create Date column
df["date"] = df["time"].dt.date

# Group by date and take mean
daily_df = df.groupby("date").mean(numeric_only=True)

# Save Final Daily File
daily_df.to_csv("visakhapatnam_air_daily_2016.csv")

print("Final Daily Data Saved Successfully!")
