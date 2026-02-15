import pandas as pd


df_weather = pd.read_csv("visakhapatnam_weather_2010_2025.csv")

# Rename columns from NASA POWER format to expected format
df_weather = df_weather.rename(columns={
    'DATE': 'date',
    'T2M_MAX': 'temp_max',
    'T2M_MIN': 'temp_min',
    'T2M': 'temp_avg',
    'RH2M': 'humidity',
    'WS10M': 'wind_speed',
    'WD10M': 'wind_direction',
    'PS': 'pressure',
    'PRECTOTCORR': 'rainfall',
    'ALLSKY_SFC_SW_DWN': 'solar_radiation',
    'CLOUD_AMT': 'cloud_cover'
})

# Convert DATE format (YYYYMMDD) to proper date format
df_weather['date'] = pd.to_datetime(df_weather['date'].astype(str), format='%Y%m%d')



# feature engineering from date
df_weather['month'] = df_weather['date'].dt.month
df_weather['day'] = df_weather['date'].dt.day
df_weather['day_of_week'] = df_weather['date'].dt.dayofweek
df_weather['year'] = df_weather['date'].dt.year

# Season Mapping
def get_season(month):
    if month in [1, 2]: return 0 # Winter
    elif month in [3, 4, 5]: return 1 # Summer
    elif month in [6, 7, 8, 9]: return 2 # Monsoon
    elif month in [10, 11, 12]: return 3 # Post-Monsoon
    return 0
    
df_weather['season'] = df_weather['month'].apply(get_season)



# save final dataset
df_weather.to_csv("final_weather_dataset_2010-2025.csv", index=False)

print("Final dataset saved successfully")
