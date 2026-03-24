"""
One-shot migration: adds phone_number column to the Supabase users table.
Run once from the project root with:  python add_phone_column.py
"""
import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from dotenv import load_dotenv
load_dotenv()

SUPABASE_URL = os.environ.get("SUPABASE_URL", "").strip()
SUPABASE_SERVICE_KEY = os.environ.get("SUPABASE_SERVICE_KEY", "").strip()

if not SUPABASE_URL or not SUPABASE_SERVICE_KEY:
    print("[ERROR] SUPABASE_URL or SUPABASE_SERVICE_KEY not found in environment.")
    sys.exit(1)

try:
    from supabase import create_client
    sb = create_client(SUPABASE_URL, SUPABASE_SERVICE_KEY)
    print(f"[INFO] Connected to Supabase: {SUPABASE_URL}")
except Exception as e:
    print(f"[ERROR] Could not connect to Supabase: {e}")
    sys.exit(1)

# Check if phone_number column already exists
try:
    resp = sb.table('users').select('phone_number').limit(1).execute()
    print("[OK] phone_number column already exists in Supabase 'users' table. Nothing to do.")
    sys.exit(0)
except Exception as e:
    err = str(e)
    if 'pgrst204' in err.lower() or 'phone_number' in err.lower() or 'column' in err.lower():
        print("[INFO] phone_number column not found — will add it via Supabase RPC/SQL.")
    else:
        print(f"[WARN] Unexpected check error: {e}")

# Try using Supabase's postgrest RPC for raw SQL (works if you have service key)
try:
    result = sb.rpc('exec_sql', {
        'sql': 'ALTER TABLE public.users ADD COLUMN IF NOT EXISTS phone_number text;'
    }).execute()
    print("[OK] Migration ran via exec_sql RPC.")
except Exception as rpc_err:
    print(f"[WARN] exec_sql RPC not available: {rpc_err}")
    print()
    print("=" * 60)
    print("ACTION REQUIRED: Run this SQL manually in the Supabase Dashboard:")
    print("  https://supabase.com/dashboard/project/mhzmapbbfaukwppnrahr/editor")
    print()
    print("SQL to run:")
    print("  ALTER TABLE public.users ADD COLUMN IF NOT EXISTS phone_number text;")
    print("=" * 60)
    sys.exit(1)

# Verify the column is now present
try:
    sb.table('users').select('phone_number').limit(1).execute()
    print("[OK] Verified: phone_number column is now accessible.")
except Exception as e:
    print(f"[WARN] Could not verify column after migration: {e}")
