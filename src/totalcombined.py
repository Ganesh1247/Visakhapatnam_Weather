import pandas as pd

# Load files
df_air = pd.read_csv("air_quality_clean.csv")
df_weather = pd.read_csv("final_weather_dataset_2010-2025.csv")

# Convert to datetime
df_air['date'] = pd.to_datetime(df_air['date'])
df_weather['date'] = pd.to_datetime(df_weather['date'])

# Merge using INNER JOIN
df_final = pd.merge(df_air, df_weather, on='date', how='inner')

# Save final dataset
df_final.to_csv("final_master_dataset.csv", index=False)

print("Final Dataset Created Successfully!")
# convert date column to datetime once
df_final['date'] = pd.to_datetime(df_final['date'])

# feature engineering from date
df_final['month'] = df_final['date'].dt.month
df_final['day'] = df_final['date'].dt.day
df_final['day_of_week'] = df_final['date'].dt.dayofweek
df_final['year'] = df_final['date'].dt.year

# Season Mapping
def get_season(month):
    if month in [1, 2]: return 0 # Winter
    elif month in [3, 4, 5]: return 1 # Summer
    elif month in [6, 7, 8, 9]: return 2 # Monsoon
    elif month in [10, 11, 12]: return 3 # Post-Monsoon
    return 0

df_final['season'] = df_final['month'].apply(get_season)

# OPTIONAL – unique id column
df_final['id'] = range(1, len(df_final) + 1)

# save final dataset
df_final.to_csv("final_dataset.csv", index=False)

print("Final dataset saved successfully")

