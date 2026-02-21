import sqlite3
import secrets
import smtplib
import os
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from datetime import datetime, timedelta
from functools import wraps
from flask import session, redirect, url_for, request
from werkzeug.security import generate_password_hash, check_password_hash

# Try to import Supabase client
try:
    from supabase import create_client, Client
    SUPABASE_AVAILABLE = True
except ImportError as e:
    SUPABASE_AVAILABLE = False
    print(f"[WARN] supabase library not found. Falling back to SQLite. Error: {e}")

# Define DB Path relative to this file
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DB_PATH = os.path.join(BASE_DIR, 'data', 'users.db')

# Supabase Configuration
SUPABASE_URL = os.environ.get("SUPABASE_URL")
SUPABASE_KEY = os.environ.get("SUPABASE_KEY")
SUPABASE_SERVICE_KEY = os.environ.get("SUPABASE_SERVICE_KEY")

def get_supabase_client():
    """Anon/publishable client (fallback only)."""
    if SUPABASE_AVAILABLE and SUPABASE_URL and SUPABASE_KEY:
        try:
            return create_client(SUPABASE_URL, SUPABASE_KEY)
        except Exception as e:
            print(f"[WARN] Invalid Supabase URL or key: {e}. Falling back to SQLite.")
            return None
    return None

def get_supabase_admin_client():
    """Service role client for privileged writes (bypasses RLS)."""
    if SUPABASE_AVAILABLE and SUPABASE_URL and SUPABASE_SERVICE_KEY:
        try:
            return create_client(SUPABASE_URL, SUPABASE_SERVICE_KEY)
        except Exception as e:
            print(f"[WARN] Invalid Supabase URL or service key: {e}. Falling back to anon client.")
            return get_supabase_client()  # Fallback to anon if no service key
    return get_supabase_client()  # Fallback to anon if no service key

def get_db_client():
    """
    Preferred DB client for backend operations.
    Always prefer service-role key in server-side code to avoid RLS read/write mismatches.
    """
    return get_supabase_admin_client()


# Database initialization
def init_db():
    supabase = get_db_client()
    if supabase:
        print("[INFO] Using Supabase Backend.")
        # Supabase tables are created via SQL editor/dashboard, not here.
        return

    # Fallback to SQLite
    os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute('''CREATE TABLE IF NOT EXISTS users
                 (id INTEGER PRIMARY KEY AUTOINCREMENT,
                  email TEXT UNIQUE NOT NULL,
                  username TEXT,
                  password_hash TEXT,
                  otp TEXT,
                  otp_expiry TIMESTAMP,
                  created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP)''')
    try:
        c.execute('ALTER TABLE users ADD COLUMN username TEXT')
    except sqlite3.OperationalError:
        pass
    try:
        c.execute('ALTER TABLE users ADD COLUMN password_hash TEXT')
    except sqlite3.OperationalError:
        pass
    c.execute('CREATE UNIQUE INDEX IF NOT EXISTS idx_users_username ON users(username)')
    conn.commit()
    conn.close()

def generate_otp():
    """Generate a 6-digit OTP"""
    return str(secrets.randbelow(900000) + 100000)

def send_otp_email(email, otp):
    """Send OTP via SMTP. Returns True only when an actual email was sent."""
    sender_email = os.environ.get("SMTP_EMAIL", "your.email@gmail.com")
    sender_password = os.environ.get("SMTP_PASSWORD", "your_app_password_here")
    is_hf_space = bool(os.environ.get("SPACE_ID") or os.environ.get("HF_SPACE_ID"))

    smtp_configured = (
        sender_email != "your.email@gmail.com" and
        sender_password != "your_app_password_here" and
        bool(sender_email) and bool(sender_password)
    )

    if smtp_configured:
        msg = MIMEMultipart()
        msg["From"] = sender_email
        msg["To"] = email
        msg["Subject"] = "Your OTP for EcoGlance Air Quality App"

        html_body = f"""
        <html>
            <body style="font-family: Arial, sans-serif; padding: 20px; background-color: #f5f5f5;">
                <div style="max-width: 600px; margin: 0 auto; background: white; padding: 30px; border-radius: 10px; box-shadow: 0 2px 10px rgba(0,0,0,0.1);">
                    <h2 style="color: #667eea; margin-bottom: 20px;">EcoGlance Verification</h2>
                    <p style="font-size: 16px; color: #333;">Your One-Time Password (OTP) is:</p>
                    <div style="background: #f0f0f0; padding: 20px; border-radius: 8px; text-align: center; margin: 20px 0;">
                        <h1 style="color: #667eea; font-size: 36px; letter-spacing: 8px; margin: 0;">{otp}</h1>
                    </div>
                    <p style="font-size: 14px; color: #666;">This OTP will expire in <strong>5 minutes</strong>.</p>
                    <p style="font-size: 14px; color: #666;">If you didn't request this, please ignore this email.</p>
                    <hr style="border: none; border-top: 1px solid #eee; margin: 20px 0;">
                    <p style="font-size: 12px; color: #999;">EcoGlance - Air Quality &amp; Weather Forecast System</p>
                </div>
            </body>
        </html>
        """
        msg.attach(MIMEText(html_body, "html"))

        try:
            with smtplib.SMTP_SSL("smtp.gmail.com", 465, timeout=20) as server:
                server.login(sender_email, sender_password)
                server.send_message(msg)
            print(f"[OK] OTP sent successfully to {email} via SMTP_SSL:465")
            return True
        except Exception as err_ssl:
            print(f"[WARN] SMTP_SSL:465 failed: {err_ssl}")

        try:
            with smtplib.SMTP("smtp.gmail.com", 587, timeout=20) as server:
                server.ehlo()
                server.starttls()
                server.ehlo()
                server.login(sender_email, sender_password)
                server.send_message(msg)
            print(f"[OK] OTP sent successfully to {email} via SMTP STARTTLS:587")
            return True
        except Exception as err_tls:
            print(f"[WARN] SMTP STARTTLS:587 failed: {err_tls}")

    if is_hf_space:
        print("[ERROR] OTP email not sent: SMTP is not configured or failed on Hugging Face Space.")
        return False

    print(f"[WARNING] No email configured! OTP for {email}: {otp}")
    print("[INFO] To enable email: Add SMTP_EMAIL and SMTP_PASSWORD in .env")
    print("[INFO] Get Gmail App Password: https://myaccount.google.com/apppasswords")
    return True
def login_required(f):
    """Decorator to protect routes"""
    @wraps(f)
    def decorated_function(*args, **kwargs):
        if 'user_email' not in session and not session.get('guest_access'):
            return redirect(url_for('login_page'))
        return f(*args, **kwargs)
    return decorated_function


def get_user_by_email(email):
    """Get user record by email"""
    supabase = get_db_client()
    if supabase:
        try:
            response = supabase.table('users').select("*").eq('email', email).execute()
            if response.data and len(response.data) > 0:
                user = response.data[0]
                # Map Supabase dict to tuple format for compatibility: (id, email, username, password_hash)
                return (user.get('id'), user.get('email'), user.get('username'), user.get('password_hash'))
            return None
        except Exception as e:
            print(f"Supabase Error: {e}")
            return None

    # Fallback to SQLite
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute('SELECT id, email, username, password_hash FROM users WHERE email = ?', (email,))
    row = c.fetchone()
    conn.close()
    return row

def get_user_by_username(username):
    """Get user record by username"""
    supabase = get_db_client()
    if supabase:
        try:
            response = supabase.table('users').select("*").eq('username', username).execute()
            if response.data and len(response.data) > 0:
                user = response.data[0]
                return (user.get('id'), user.get('email'), user.get('username'), user.get('password_hash'))
            return None
        except Exception as e:
            print(f"Supabase Error: {e}")
            return None

    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute('SELECT id, email, username, password_hash FROM users WHERE username = ?', (username,))
    row = c.fetchone()
    conn.close()
    return row

def user_has_credentials(email):
    """Check if user has set username and password (returning user)"""
    row = get_user_by_email(email)
    # Row is tuple (id, email, username, password_hash)
    return row and row[2] and row[3]

def set_user_credentials(email, username, password):
    """Set username and hashed password for user after OTP verification"""
    password_hash = generate_password_hash(password, method='pbkdf2:sha256')
    
    supabase = get_db_client()  # Admin for reads/writes
    if supabase:
        try:
            # Check unique username first
            existing = supabase.table('users').select("id").eq('username', username).execute()
            if existing.data and len(existing.data) > 0:
                return False  # Username taken
            
            # Update user with admin client
            supabase.table('users').update({
                'username': username,
                'password_hash': password_hash
            }).eq('email', email).execute()
            
            return True
        except Exception as e:
            print(f"Supabase Update Error: {e}")
            return False

    # Fallback SQLite
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    try:
        c.execute('UPDATE users SET username = ?, password_hash = ? WHERE email = ?',
                  (username, password_hash, email))
        conn.commit()
        return True
    except sqlite3.IntegrityError:
        return False
    finally:
        conn.close()

def save_otp(email, otp, expires_at):
    """Save or update OTP for a user"""
    supabase = get_db_client()  # Use admin for writes
    if supabase:
        try:
            # Single atomic write path: update existing row or insert new row by email.
            supabase.table('users').upsert({
                'email': email,
                'otp': otp,
                'otp_expiry': expires_at.isoformat()
            }, on_conflict='email').execute()
            return True
        except Exception as e:
            print(f"Supabase OTP Error: {e}")
            return False

    # SQLite Fallback
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    try:
        c.execute('SELECT id FROM users WHERE email = ?', (email,))
        if c.fetchone():
            c.execute('UPDATE users SET otp = ?, otp_expiry = ? WHERE email = ?',
                      (otp, expires_at, email))
        else:
            c.execute('INSERT INTO users (email, otp, otp_expiry) VALUES (?, ?, ?)',
                      (email, otp, expires_at))
        conn.commit()
        return True
    except Exception as e:
        print(f"SQLite OTP Error: {e}")
        return False
    finally:
        conn.close()

def get_otp(email):
    """Retrieve OTP and expiry for an email"""
    supabase = get_db_client()
    if supabase:
        try:
            response = supabase.table('users').select("otp, otp_expiry").eq('email', email).execute()
            if response.data and len(response.data) > 0:
                user = response.data[0]
                return user.get('otp'), user.get('otp_expiry')
            return None, None
        except Exception as e:
            print(f"Supabase OTP Fetch Error: {e}")
            return None, None

    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute('SELECT otp, otp_expiry FROM users WHERE email = ?', (email,))
    row = c.fetchone()
    conn.close()
    return row if row else (None, None)

def verify_password(username_or_email, password):
    """Verify username/password. Returns (success, email) or (False, None)"""
    row = get_user_by_username(username_or_email)
    if not row:
        row = get_user_by_email(username_or_email)
    
    if not row or not row[3]:  # no password_hash
        return False, None
        
    if check_password_hash(row[3], password):
        return True, row[1]  # return email
        
    return False, None

