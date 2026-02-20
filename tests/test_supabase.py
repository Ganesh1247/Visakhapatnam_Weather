import os
import sys
from dotenv import load_dotenv

load_dotenv()

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from auth import get_supabase_client

def test_connection():
    print("Testing Supabase Connection...")
    client = get_supabase_client()
    if not client:
        print("[FAIL] Could not create Supabase client (check credentials).")
        return
    
    print("[OK] Supabase client created.")
    
    try:
        # Try a simple read (even if empty) to verify connectivity
        # users table exists from schema
        # We use limit=1 to avoid fetching too much data, just checking if we can query
        res = client.table('users').select("*").limit(1).execute()
        print(f"[OK] Connection successful! Query executed without error.")
    except Exception as e:
        print(f"[FAIL] Connection error: {e}")
        print("Note: If table 'users' does not exist, run the supabase_schema.sql script in Supabase dashboard.")

if __name__ == "__main__":
    test_connection()
