import os
import sys
import requests
from datetime import datetime

sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))
from dotenv import load_dotenv
load_dotenv()

from sms_alerts import get_all_phone_numbers, _twilio_client, _format_from_number, _format_number_for_twilio, TWILIO_FROM, USE_WHATSAPP


def force_send():
    print("1. Fetching registered users with phone numbers...")
    recipients = get_all_phone_numbers()
    if not recipients:
        print("   -> No users found with a registered phone number.")
        return
    
    print(f"   -> Found {len(recipients)} recipients: {recipients}")

    print("\n2. Fetching current forecast from running local app to build message...")
    try:
        # Hit the local predictor to get real data just like the frontend does
        resp = requests.get('http://127.0.0.1:5000/predict?method=mc_dropout', timeout=5)
        resp.raise_for_status()
        data = resp.json()
        print("   -> Success: Fetched real-time forecast data.")
    except Exception as e:
        print(f"   -> App prediction failed or unreachable: {e}")
        print("   -> [FALLBACK] Using mock Visakhapatnam data for verification.")
        data = {
            "data": {
                "pm2_5": 38.5,
                "temp_avg": 29.2,
                "humidity": 72,
                "wind_speed": 3.4,
                "rainfall": 0.0
            },
            "aqi": {
                "value": 105,
                "status": "Moderate",
                "color": "#ff7e00"
            }
        }
        
    main = data.get("data", {})
    aqi_info = data.get("aqi", {})
    
    aqi_val = aqi_info.get("value", "N/A")
    aqi_status = aqi_info.get("status", "Unknown")
    pm25 = main.get("pm2_5", "N/A")
    temp = main.get("temp_avg", "N/A")
    humidity = main.get("humidity", "N/A")
    wind = main.get("wind_speed", "N/A")
    rainfall = main.get("rainfall", 0)

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
    if isinstance(wind, (int, float)) and wind > 8:
        weather_warn += f"💨 Strong winds ({wind:.1f} m/s). "
    if isinstance(temp, (int, float)) and temp > 35:
        weather_warn += f"🌡️ Heat warning ({temp:.1f}°C). Stay hydrated. "

    now = datetime.now()
    greeting_tag = "Morning" if now.hour < 14 else "Evening"

    msg_body = (
        f"🌿 EcoGlance {greeting_tag} Alert – Visakhapatnam\n"
        f"📅 {now.strftime('%d %b %Y, %I:%M %p')}\n\n"
        f"💨 AQI: {aqi_val} ({aqi_status})\n"
        f"   PM2.5: {pm25} µg/m³\n\n"
        f"🌡️ Temp: {temp:.1f}°C | Humidity: {humidity:.0f}%\n"
        f"🌬️ Wind: {wind:.1f} m/s\n"
    )
    if weather_warn:
        msg_body += f"\n{weather_warn}\n"
    msg_body += f"\n{advice}\n\nStay well! – EcoGlance AI"
    
    # Safe print for Windows console with emojis
    def safe_print(text):
        try:
            print(text)
        except UnicodeEncodeError:
            print(text.encode('ascii', 'ignore').decode('ascii'))

    print("\n--- MESSAGE PREVIEW ---")
    safe_print(msg_body)
    print("-----------------------\n")

    print("3. Connecting to Twilio...")
    client = _twilio_client()
    if not client:
        print("   -> [ERROR] Twilio credentials (TWILIO_ACCOUNT_SID, TWILIO_AUTH_TOKEN) not found in .env.")
        print("      Cannot send actual SMS without Twilio configuration.")
        return
        
    if not TWILIO_FROM:
        print("   -> [ERROR] TWILIO_FROM_NUMBER is not set in .env.")
        return

    from_num = _format_from_number(USE_WHATSAPP)
    print(f"   -> Using sender number: {from_num}")
    
    for username, raw_phone in recipients:
        to_num = _format_number_for_twilio(raw_phone, USE_WHATSAPP)
        print(f"   -> Sending to {username} ({to_num})...")
        try:
            message = client.messages.create(
                body=msg_body,
                from_=from_num,
                to=to_num
            )
            print(f"      [SUCCESS] Message sent! SID: {message.sid}")
        except Exception as e:
            print(f"      [FAILED] Error sending: {e}")

if __name__ == "__main__":
    force_send()
