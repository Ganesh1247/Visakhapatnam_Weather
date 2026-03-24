import sys
sys.path.append('src')
from app import app
import traceback

with app.app_context():
    client = app.test_client()
    try:
        print("Fetching /predict...")
        response = client.get('/predict')
        print(f"Status: {response.status_code}")
        print(response.get_data(as_text=True))
    except Exception as e:
        traceback.print_exc()
