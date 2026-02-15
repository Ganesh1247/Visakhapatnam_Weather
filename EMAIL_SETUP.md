# Email Configuration Guide for EcoGlance OTP System

## 🔧 Setup Instructions

### Option 1: Using Gmail (Recommended)

1. **Enable 2-Factor Authentication on your Gmail account**
   - Go to: https://myaccount.google.com/security
   - Enable "2-Step Verification"

2. **Generate an App Password**
   - Go to: https://myaccount.google.com/apppasswords
   - Select "Mail" and your device
   - Click "Generate"
   - Copy the 16-character password (e.g., `abcd efgh ijkl mnop`)

3. **Update auth.py**
   - Open `auth.py`
   - Line 28: Replace `"your.email@gmail.com"` with your Gmail address
   - Line 29: Replace `"your_app_password_here"` with the App Password (remove spaces)

   Example:
   ```python
   SENDER_EMAIL = "myemail@gmail.com"
   SENDER_PASSWORD = "abcdefghijklmnop"  # 16 chars, no spaces
   ```

4. **Restart the Flask app**
   ```bash
   # Stop current server (Ctrl+C in terminal)
   python app.py
   ```

### Option 2: Using Environment Variables (More Secure)

1. Create a `.env` file in your project directory:
   ```
   SMTP_EMAIL=your.email@gmail.com
   SMTP_PASSWORD=your_app_password_here
   ```

2. Install python-dotenv:
   ```bash
   pip install python-dotenv
   ```

3. Update auth.py (lines 28-29):
   ```python
   import os
   from dotenv import load_dotenv
   
   load_dotenv()
   SENDER_EMAIL = os.environ.get('SMTP_EMAIL')
   SENDER_PASSWORD = os.environ.get('SMTP_PASSWORD')
   ```

### Option 3: Using Other Email Providers

#### Outlook/Hotmail
```python
# In auth.py, change the SMTP server (line 56):
with smtplib.SMTP_SSL('smtp-mail.outlook.com', 587) as server:
```

#### Yahoo Mail
```python
# In auth.py, change the SMTP server (line 56):
with smtplib.SMTP_SSL('smtp.mail.yahoo.com', 465) as server:
```

## ⚠️ Troubleshooting

### "SMTPAuthenticationError"
- ✅ Make sure you're using an **App Password**, not your regular Gmail password
- ✅ Check that 2-Factor Authentication is enabled
- ✅ Verify email and password are correct (no extra spaces)

### OTP still showing in console
- ✅ Verify you updated both SENDER_EMAIL and SENDER_PASSWORD
- ✅ Restart the Flask server after making changes
- ✅ Check console for error messages

### Email not received
- ✅ Check spam/junk folder
- ✅ Verify the recipient email is correct
- ✅ Check console for "✅ OTP sent successfully" message

## 🔒 Security Notes

- **Never commit** your App Password to Git
- Use environment variables for production
- App Passwords are specific to apps - rotate them if needed
- If compromised, revoke the App Password in Google Account settings

## 📧 Current Status

The system will:
1. Try to send OTP via email
2. If email fails → Show OTP in console as fallback
3. Login still works even if email fails
