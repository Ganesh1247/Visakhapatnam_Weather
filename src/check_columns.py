import pandas as pd

# Check weather dataset
df_weather = pd.read_csv('final_weather_dataset_2010-2025.csv')
print("=" * 60)
print("WEATHER DATASET COLUMNS:")
print("=" * 60)
for i, col in enumerate(df_weather.columns, 1):
    print(f"{i}. {col}")
print(f"\nTotal: {len(df_weather.columns)} columns")

print("\n" + "=" * 60)
print("FINAL (COMBINED) DATASET COLUMNS:")
print("=" * 60)

# Check final dataset
df_final = pd.read_csv('final_dataset.csv')
for i, col in enumerate(df_final.columns, 1):
    print(f"{i}. {col}")
print(f"\nTotal: {len(df_final.columns)} columns")
