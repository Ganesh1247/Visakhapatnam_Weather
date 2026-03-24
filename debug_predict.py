import urllib.request
import urllib.error

try:
    url = 'http://127.0.0.1:5000/predict'
    with urllib.request.urlopen(url) as r:
        print(r.read().decode())
except urllib.error.HTTPError as e:
    print("HTTP Error:", e.code)
    print(e.read().decode())
except Exception as e:
    print("Error:", e)
