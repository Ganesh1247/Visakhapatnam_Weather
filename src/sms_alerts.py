"""
EcoGlance SMS Alert System
==========================
Sends daily AQI + weather warnings to registered users at 10:00 AM and 18:00 PM (IST).
Uses Twilio SMS/WhatsApp. Falls back to a no-op if Twilio is not configured.

Environment variables required:
  TWILIO_ACCOUNT_SID   – Your Twilio Account SID
  TWILIO_AUTH_TOKEN    – Your Twilio Auth Token
  TWILIO_FROM_NUMBER   – Twilio sending number  e.g. +14155552671
                         For WhatsApp: whatsapp:+14155552671
  TWILIO_USE_WHATSAPP  – Set to "true" to use WhatsApp Sandbox instead of SMS
"""

import os
import sqlite3
import threading
import time
import logging
from datetime import datetime, timedelta

logger = logging.getLogger("sms_alerts")

# ─── Optional Twilio import ─────────────────────────────────────────────────
try:
    from twilio.rest import Client as TwilioClient
    TWILIO_AVAILABLE = True
except ImportError:
    TWILIO_AVAILABLE = False
    logger.warning("[SMS] twilio library not installed. SMS alerts disabled.")

# ─── Optional APScheduler import ────────────────────────────────────────────
try:
    from apscheduler.schedulers.background import BackgroundScheduler
    from apscheduler.triggers.cron import CronTrigger
    import pytz
    SCHEDULER_AVAILABLE = True
except ImportError:
    SCHEDULER_AVAILABLE = False
    logger.warning("[SMS] APScheduler not installed. Using fallback polling thread.")

# ─── Twilio Config ───────────────────────────────────────────────────────────
TWILIO_SID      = os.environ.get("TWILIO_ACCOUNT_SID", "").strip()
TWILIO_TOKEN    = os.environ.get("TWILIO_AUTH_TOKEN", "").strip()
TWILIO_FROM     = os.environ.get("TWILIO_FROM_NUMBER", "").strip()
USE_WHATSAPP    = os.environ.get("TWILIO_USE_WHATSAPP", "false").lower() == "true"

# ─── Webpushr Config ─────────────────────────────────────────────────────────
WEBPUSHR_KEY    = os.environ.get("WEBPUSHR_KEY", "").strip()
WEBPUSHR_TOKEN  = os.environ.get("WEBPUSHR_TOKEN", "").strip()

# DB path (same as auth.py)
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DB_PATH  = os.path.join(BASE_DIR, "data", "users.db")

IST = None
try:
    import pytz
    IST = pytz.timezone("Asia/Kolkata")
except Exception:
    pass


def _twilio_client():
    """Return a Twilio client if configured, else None."""
    if not TWILIO_AVAILABLE:
        return None
    if not (TWILIO_SID and TWILIO_TOKEN):
        logger.debug("[SMS] Twilio credentials not set – skipping.")
        return None
    return TwilioClient(TWILIO_SID, TWILIO_TOKEN)


def _format_number_for_twilio(raw: str, use_whatsapp: bool = False) -> str:
    """Normalize phone number and optionally prefix whatsapp: scheme."""
    num = raw.strip()
    if not num.startswith("+"):
        # Assume India (+91) if no country code
        num = "+91" + num.lstrip("0")
    if use_whatsapp:
        return f"whatsapp:{num}"
    return num


def _format_from_number(use_whatsapp: bool) -> str:
    frm = TWILIO_FROM
    if use_whatsapp and not frm.startswith("whatsapp:"):
        return f"whatsapp:{frm}"
    return frm


# ─── DB helpers ─────────────────────────────────────────────────────────────

def get_all_phone_numbers():
    """Return list of (username, phone_number) for users who opted in."""
    # Try Supabase first
    try:
        from auth import get_supabase_admin_client
        sb = get_supabase_admin_client()
        if sb:
            resp = sb.table("users").select("username,phone_number").neq("phone_number", None).execute()
            result = []
            for row in (resp.data or []):
                ph = (row.get("phone_number") or "").strip()
                if ph:
                    result.append((row.get("username", "User"), ph))
            return result
    except Exception as e:
        logger.warning(f"[SMS] Supabase phone lookup failed: {e}")

    # SQLite fallback
    if not os.path.exists(DB_PATH):
        return []
    try:
        conn = sqlite3.connect(DB_PATH)
        c = conn.cursor()
        c.execute("SELECT username, phone_number FROM users WHERE phone_number IS NOT NULL AND phone_number != ''")
        rows = c.fetchall()
        conn.close()
        return [(r[0] or "User", r[1]) for r in rows if r[1]]
    except Exception as e:
        logger.warning(f"[SMS] SQLite phone lookup failed: {e}")
        return []


# ─── Forecast helper ─────────────────────────────────────────────────────────

def get_current_forecast_summary():
    """
    Import the running app's cached forecast and return a text summary.
    Falls back gracefully if forecast not available yet.
    """
    try:
        # Import lazily to avoid circular imports at module load time
        import sys, os
        sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
        from app import forecast_cache, calculate_india_aqi_from_pm25, get_aqi_status_and_color

        data = None
        for method_key in ["forecast_mc_dropout", "forecast_standard"]:
            cached = forecast_cache.get(method_key)
            if cached:
                data = cached
                break

        if not data:
            return None

        main = data.get("data", {})
        aqi_info = data.get("aqi", {})
        forecast = data.get("forecast", [])

        aqi_val   = aqi_info.get("value", "N/A")
        aqi_status = aqi_info.get("status", "Unknown")
        pm25      = main.get("pm2_5", "N/A")
        temp      = main.get("temp_avg", "N/A")
        humidity  = main.get("humidity", "N/A")
        wind      = main.get("wind_speed", "N/A")
        rainfall  = main.get("rainfall", 0)

        # Health advice based on AQI tier
        advice_map = {
            "Good": "✅ Great day! Safe for all outdoor activities.",
            "Satisfactory": "👍 Generally safe. Sensitive groups take light precautions.",
            "Moderate": "⚠️ Limit heavy outdoor exercise. N95 mask recommended.",
            "Poor": "😷 Stay indoors if possible. Wear N95 mask outside.",
            "Very Poor": "🚨 Avoid outdoor exposure. Use air purifiers indoors.",
            "Severe": "🆘 Emergency conditions! Stay sealed indoors. Follow health advisories.",
        }
        advice = advice_map.get(aqi_status, "ℹ️ Monitor conditions and stay safe.")

        weather_warn = ""
        if isinstance(rainfall, (int, float)) and rainfall > 5:
            weather_warn = f"🌧️ Rain expected ({rainfall:.1f} mm). Carry an umbrella. "
        if isinstance(wind, (int, float)) and wind > 10:
            weather_warn += f"💨 Strong winds ({wind:.1f} m/s). "
        if isinstance(temp, (int, float)) and temp > 38:
            weather_warn += f"🌡️ Extreme heat ({temp:.1f}°C). Stay hydrated. "

        now = datetime.now()
        greeting_tag = "Morning" if now.hour < 14 else "Evening"

        msg = (
            f"🌿 EcoGlance {greeting_tag} Alert – Visakhapatnam\n"
            f"📅 {now.strftime('%d %b %Y, %I:%M %p')}\n\n"
            f"💨 AQI: {aqi_val} ({aqi_status})\n"
            f"   PM2.5: {pm25} µg/m³\n\n"
            f"🌡️ Temp: {temp:.1f}°C | Humidity: {humidity:.0f}%\n"
            f"🌬️ Wind: {wind:.1f} m/s\n"
        )
        if weather_warn:
            msg += f"\n{weather_warn}\n"
        msg += f"\n{advice}\n\nStay well! – EcoGlance AI"
        return msg

    except Exception as e:
        logger.error(f"[SMS] Failed to build forecast summary: {e}")
        return None


# ─── Core send function ───────────────────────────────────────────────────────

def send_webpushr_notification(msg_body: str):
    """Hits the Webpushr REST API to broadcast the message to all registered browsers."""
    if not WEBPUSHR_KEY or not WEBPUSHR_TOKEN:
        logger.debug("[Push] Webpushr credentials not set – skipping background push.")
        return
        
    import requests
    import json
    
    url = "https://api.webpushr.com/v1/notification/send/all"
    headers = {
        "webpushrKey": WEBPUSHR_KEY,
        "webpushrAuthToken": WEBPUSHR_TOKEN,
        "Content-Type": "application/json"
    }
    
    # Parse title out of the message body (usually the first line)
    title = msg_body.split('\n')[0].strip() if msg_body else "EcoGlance AQI & Weather Alert"
    
    payload = {
        "title": title,
        "message": msg_body,
        "target_url": "/", # Relative routing to root
        "icon": "https://ganesh1247-visakhapatnam-weather.hf.space/static/favicon.svg"
    }
    
    try:
        r = requests.post(url, headers=headers, data=json.dumps(payload), timeout=10)
        if r.status_code == 200:
            logger.info("[Push] ✓ Sent Webpushr Push Notification.")
        else:
            logger.warning(f"[Push] Webpushr API returned {r.status_code}: {r.text}")
    except Exception as e:
        logger.error(f"[Push] ✗ Failed to send push: {e}")

def send_alert_to_all_users():
    """
    Main entry-point called by the scheduler (10 AM and 6 PM IST).
    Fetches forecast, formats message, and sends to every opted-in user.
    """
    logger.info("[SMS+Push] Daily alert job triggered.")

    msg_body = get_current_forecast_summary()
    if not msg_body:
        logger.warning("[SMS+Push] Forecast not available yet – skipping alert.")
        return

    # Trigger Webpushr Push broadly
    send_webpushr_notification(msg_body)

    client = _twilio_client()
    if not client:
        logger.info("[SMS] No Twilio client – alert job skipped.")
        return

    if not TWILIO_FROM:
        logger.warning("[SMS] TWILIO_FROM_NUMBER not set – aborting.")
        return

    recipients = get_all_phone_numbers()
    if not recipients:
        logger.info("[SMS] No users with phone numbers – nothing to send.")
        return

    from_num = _format_from_number(USE_WHATSAPP)
    success_count, fail_count = 0, 0

    for username, raw_phone in recipients:
        try:
            to_num = _format_number_for_twilio(raw_phone, USE_WHATSAPP)
            client.messages.create(
                body=msg_body,
                from_=from_num,
                to=to_num
            )
            logger.info(f"[SMS] ✓ Sent to {username} ({to_num})")
            success_count += 1
        except Exception as e:
            logger.error(f"[SMS] ✗ Failed for {username} ({raw_phone}): {e}")
            fail_count += 1

    logger.info(f"[SMS] Alert job done – {success_count} sent, {fail_count} failed.")


# ─── Scheduler setup ─────────────────────────────────────────────────────────

_scheduler = None
_fallback_thread = None
_fallback_stop = threading.Event()


def _fallback_polling_loop():
    """
    Simple thread-based scheduler for environments without APScheduler.
    Fires at 10:00 and 18:00 IST each day.
    """
    logger.info("[SMS] Fallback polling scheduler started.")
    fired_today = {}  # {date_str: set_of_hours_fired}

    while not _fallback_stop.is_set():
        try:
            if IST:
                now = datetime.now(IST)
            else:
                now = datetime.utcnow() + timedelta(hours=5, minutes=30)

            date_key = now.strftime("%Y-%m-%d")
            hour = now.hour

            if date_key not in fired_today:
                fired_today = {date_key: set()}

            if hour in (10, 18) and hour not in fired_today[date_key]:
                fired_today[date_key].add(hour)
                send_alert_to_all_users()

        except Exception as e:
            logger.error(f"[SMS] Fallback loop error: {e}")

        # Sleep 55 seconds (wakes up each minute to check)
        _fallback_stop.wait(55)


def start_alert_scheduler():
    """
    Start the background alert scheduler.
    Prefers APScheduler; falls back to simple polling thread.
    """
    global _scheduler, _fallback_thread

    if SCHEDULER_AVAILABLE:
        try:
            tz = pytz.timezone("Asia/Kolkata")
            _scheduler = BackgroundScheduler(timezone=tz)
            # Fire at 10:00 AM IST every day
            _scheduler.add_job(
                send_alert_to_all_users,
                CronTrigger(hour=10, minute=0, timezone=tz),
                id="alert_morning",
                name="Morning AQI + Weather Alert",
                replace_existing=True
            )
            # Fire at 6:00 PM IST every day
            _scheduler.add_job(
                send_alert_to_all_users,
                CronTrigger(hour=18, minute=0, timezone=tz),
                id="alert_evening",
                name="Evening AQI + Weather Alert",
                replace_existing=True
            )
            _scheduler.start()
            logger.info("[SMS] APScheduler started – alerts at 10:00 and 18:00 IST.")
            return
        except Exception as e:
            logger.error(f"[SMS] APScheduler init failed: {e}. Using fallback.")

    # Fallback: simple polling thread
    _fallback_thread = threading.Thread(
        target=_fallback_polling_loop, daemon=True, name="sms-alert-poller"
    )
    _fallback_thread.start()
    logger.info("[SMS] Fallback polling scheduler started.")


def stop_alert_scheduler():
    """Gracefully stop the scheduler on app shutdown."""
    global _scheduler, _fallback_thread
    if _scheduler:
        try:
            _scheduler.shutdown(wait=False)
        except Exception:
            pass
    _fallback_stop.set()
