import requests
import os

def test_pdf_download():
    url = "http://127.0.0.1:5000/download_report?range=week&lat=17.7138&lon=83.275"
    try:
        print(f"Testing download from {url}...")
        resp = requests.get(url, timeout=30)
        print(f"Status Code: {resp.status_code}")
        print(f"Content-Type: {resp.headers.get('Content-Type')}")
        print(f"Content-Length: {resp.headers.get('Content-Length')}")
        
        if resp.status_code == 200:
            content = resp.content
            print(f"Downloaded size: {len(content)} bytes")
            if len(content) > 10:
                print(f"First 10 bytes: {content[:10]}")
                if content.startswith(b'%PDF'):
                    print("SUCCESS: Valid PDF header found!")
                else:
                    print("FAILURE: Content does not start with %PDF. It starts with:")
                    print(content[:100])
            else:
                print("FAILURE: Empty or near-empty response content.")
        else:
            print(f"FAILURE: Server returned error {resp.text}")
            
    except Exception as e:
        print(f"ALARM: Connection failed! {e}")

if __name__ == "__main__":
    test_pdf_download()
