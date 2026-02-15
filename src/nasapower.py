import requests
import pandas as pd

lat = 17.6868
lon = 83.2185

start_date = "20100101"
end_date   = "20251231"

parameters = ",".join([
    "T2M",
    "T2M_MAX",
    "T2M_MIN",
    "RH2M",
    "WS10M",
    "WD10M",
    "PS",
    "PRECTOT",
    "ALLSKY_SFC_SW_DWN",
    "ALLSKY_SFC_LW_DWN",
    "CLRSKY_SFC_SW_DWN",
    "CLOUD_AMT"
])

url = (
    "https://power.larc.nasa.gov/api/temporal/daily/point?"
    f"latitude={lat}&longitude={lon}"
    f"&start={start_date}&end={end_date}"
    f"&parameters={parameters}"
    "&community=RE"
    "&format=JSON"
)

response = requests.get(url)
data = response.json()

records = data["properties"]["parameter"]

df = pd.DataFrame(records)
df.index.name = "DATE"

df.to_csv("visakhapatnam_weather_2010_2025.csv")

print("Downloaded Successfully!")
