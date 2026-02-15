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

# Define DB Path relative to this file
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DB_PATH = os.path.join(BASE_DIR, 'data', 'users.db')

# Database initialization
def init_db():
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
    # Migration: add username and password_hash if table existed before.
    # Note: SQLite ALTER TABLE has limited support for constraints, so we add
    # columns without UNIQUE and create a unique index separately.
    try:
        c.execute('ALTER TABLE users ADD COLUMN username TEXT')
    except sqlite3.OperationalError:
        pass
    try:
        c.execute('ALTER TABLE users ADD COLUMN password_hash TEXT')
    except sqlite3.OperationalError:
        pass
    # Unique constraint for username (works even if column already exists)
    c.execute('CREATE UNIQUE INDEX IF NOT EXISTS idx_users_username ON users(username)')
    conn.commit()
    conn.close()

def generate_otp():
    """Generate a 6-digit OTP"""
    return str(secrets.randbelow(900000) + 100000)

def send_otp_email(email, otp):
    """Send OTP via email using Gmail SMTP"""
    # Email Configuration - UPDATE THESE VALUES
    SENDER_EMAIL = "koiladaganesh43683@gmail.com"  # Replace with your Gmail address
    SENDER_PASSWORD = "ccyh mrps evjb dzlz"  # Replace with Gmail App Password (NOT your regular password)
    
    # You can also use environment variables for security:
    # SENDER_EMAIL = os.environ.get('SMTP_EMAIL', 'your.email@gmail.com')
    # SENDER_PASSWORD = os.environ.get('SMTP_PASSWORD', 'your_app_password_here')
    
    # Check if credentials are configured
    if SENDER_EMAIL == "your.email@gmail.com" or SENDER_PASSWORD == "your_app_password_here":
        print(f"[WARNING] Email not configured! OTP for {email}: {otp}")
        print("[EMAIL] To enable email: Update SENDER_EMAIL and SENDER_PASSWORD in auth.py")
        print("[KEY] Get Gmail App Password: https://myaccount.google.com/apppasswords")
        return True  # Return success so login still works
    
    try:
        # Create message
        msg = MIMEMultipart()
        msg['From'] = SENDER_EMAIL
        msg['To'] = email
        msg['Subject'] = "Your OTP for EcoGlance Air Quality App"
        
        # Email body with HTML styling
        html_body = f"""
        <html>
            <body style="font-family: Arial, sans-serif; padding: 20px; background-color: #f5f5f5;">
                <div style="max-width: 600px; margin: 0 auto; background: white; padding: 30px; border-radius: 10px; box-shadow: 0 2px 10px rgba(0,0,0,0.1);">
                    <h2 style="color: #667eea; margin-bottom: 20px;">🌤️ EcoGlance Verification</h2>
                    <p style="font-size: 16px; color: #333;">Your One-Time Password (OTP) is:</p>
                    <div style="background: #f0f0f0; padding: 20px; border-radius: 8px; text-align: center; margin: 20px 0;">
                        <h1 style="color: #667eea; font-size: 36px; letter-spacing: 8px; margin: 0;">{otp}</h1>
                    </div>
                    <p style="font-size: 14px; color: #666;">This OTP will expire in <strong>5 minutes</strong>.</p>
                    <p style="font-size: 14px; color: #666;">If you didn't request this, please ignore this email.</p>
                    <hr style="border: none; border-top: 1px solid #eee; margin: 20px 0;">
                    <p style="font-size: 12px; color: #999;">EcoGlance - Air Quality & Weather Forecast System</p>
                </div>
            </body>
        </html>
        """
        
        msg.attach(MIMEText(html_body, 'html'))
        
        # Send email via Gmail SMTP
        with smtplib.SMTP_SSL('smtp.gmail.com', 465) as server:
            server.login(SENDER_EMAIL, SENDER_PASSWORD)
            server.send_message(msg)
        
        print(f"[OK] OTP sent successfully to {email}")
        return True
        
    except smtplib.SMTPAuthenticationError:
        print("[ERROR] Email authentication failed. Check your Gmail App Password.")
        print(f"[FALLBACK] OTP for {email}: {otp}")
        return True  # Still return success so login works
        
    except Exception as e:
        print(f"[ERROR] Email error: {e}")
        print(f"[FALLBACK] OTP for {email}: {otp}")
        return True  # Still return success so login works

def login_required(f):
    """Decorator to protect routes"""
    @wraps(f)
    def decorated_function(*args, **kwargs):
        if 'user_email' not in session:
            return redirect(url_for('login_page'))
        return f(*args, **kwargs)
    return decorated_function


def get_user_by_email(email):
    """Get user record by email"""
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute('SELECT id, email, username, password_hash FROM users WHERE email = ?', (email,))
    row = c.fetchone()
    conn.close()
    return row

def get_user_by_username(username):
    """Get user record by username"""
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute('SELECT id, email, username, password_hash FROM users WHERE username = ?', (username,))
    row = c.fetchone()
    conn.close()
    return row

def user_has_credentials(email):
    """Check if user has set username and password (returning user)"""
    row = get_user_by_email(email)
    return row and row[2] and row[3]  # username and password_hash

def set_user_credentials(email, username, password):
    """Set username and hashed password for user after OTP verification"""
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()

    try:
        password_hash = generate_password_hash(password, method='pbkdf2:sha256')
        c.execute('UPDATE users SET username = ?, password_hash = ? WHERE email = ?',
                  (username, password_hash, email))
        conn.commit()
        return True
    except sqlite3.IntegrityError:
        return False  # username already taken
    finally:
        conn.close()

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
