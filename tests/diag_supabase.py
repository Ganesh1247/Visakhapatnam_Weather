import os, sys, base64, json
from dotenv import load_dotenv
load_dotenv()
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from supabase import create_client

url = os.environ.get('SUPABASE_URL', '')
anon_key = os.environ.get('SUPABASE_KEY', '')
svc_key = os.environ.get('SUPABASE_SERVICE_KEY', '')

# Decode JWT to check project ref
def decode_ref(jwt):
    try:
        payload = jwt.split('.')[1]
        payload += '=' * (4 - len(payload) % 4)
        data = json.loads(base64.b64decode(payload))
        return data.get('ref', 'N/A'), data.get('role', 'N/A')
    except:
        return 'N/A (not a JWT)', 'N/A'

url_ref = url.split('//')[1].split('.')[0] if '//' in url else 'N/A'
svc_ref, svc_role = decode_ref(svc_key)

print("=== Supabase Diagnostics ===")
print(f"URL project ref   : {url_ref}")
print(f"Anon key type     : {'JWT' if anon_key.startswith('eyJ') else 'Publishable/Other'}")
print(f"Service key ref   : {svc_ref}")
print(f"Service key role  : {svc_role}")
print(f"Keys match URL    : {url_ref == svc_ref}")
print()

# Test with service key
print("--- Test 1: Service key connection ---")
try:
    admin = create_client(url, svc_key)
    res = admin.table('users').select('*').limit(1).execute()
    print(f"[OK] users table reachable. Rows returned: {len(res.data)}")
except Exception as e:
    print(f"[FAIL] {e}")

# Test with anon key 
print("\n--- Test 2: Anon key connection ---")
try:
    client = create_client(url, anon_key)
    res = client.table('users').select('*').limit(1).execute()
    print(f"[OK] Anon key works. Rows returned: {len(res.data)}")
except Exception as e:
    print(f"[FAIL] {e}")

# Try insert test
print("\n--- Test 3: Insert test (service key) ---")
try:
    admin = create_client(url, svc_key)
    res = admin.table('users').insert({'email': 'test_diag@example.com', 'otp': '999999', 'otp_expiry': '2099-01-01T00:00:00'}).execute()
    print(f"[OK] Insert succeeded: {res.data}")
    # clean up
    admin.table('users').delete().eq('email', 'test_diag@example.com').execute()
    print("[OK] Cleanup done.")
except Exception as e:
    print(f"[FAIL] Insert failed: {e}")
