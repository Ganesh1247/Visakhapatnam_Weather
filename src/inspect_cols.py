import pandas as pd

try:
    df_weather = pd.read_csv('final_weather_dataset_2010-2025.csv')
    print("Weather Columns:", df_weather.columns.tolist())
    
    df_final = pd.read_csv('final_dataset.csv')
    print("Final Dataset Columns:", df_final.columns.tolist())
except Exception as e:
    print(e)
