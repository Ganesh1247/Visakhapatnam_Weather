import pandas as pd
import glob

files = glob.glob("visakhapatnam_air_daily_*.csv")

df_air = pd.concat([pd.read_csv(f) for f in files], ignore_index=True)

# Convert date
df_air['date'] = pd.to_datetime(df_air['date'])

# Remove rows where all columns except date are empty
df_air = df_air.dropna(how='all', subset=df_air.columns[1:])

# Sort + remove duplicates
df_air = df_air.sort_values('date')
df_air = df_air.drop_duplicates(subset='date')

# Interpolate numbers
num_cols = df_air.select_dtypes(include='number').columns
df_air[num_cols] = df_air[num_cols].interpolate()

df_air.to_csv("air_quality_clean.csv", index=False)

print("Clean dataset ready!")
