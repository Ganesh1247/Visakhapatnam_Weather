# Testing on Mobile Phone - Quick Fix Guide

## The Problem
When you see **"click F12"** error on your phone while testing the OTP login, it means the phone **cannot connect to the server.**

This happens because:
- The app runs on `http://127.0.0.1:5000` (localhost - only accessible from the same computer)
- Your phone is a different device and cannot access `127.0.0.1`

## Solutions

### Option 1: Use Your Render Deployment URL (RECOMMENDED)
Since you're deploying to Render, simply use your **Render app URL** on your phone instead of localhost:

```
https://your-app-name.onrender.com
```

This is the best way to test on mobile devices!

### Option 2: Test on Your Computer's Local Network IP

If you want to test locally before deploying:

#### Step 1: Find Your Computer's IP Address

**Windows:**
```powershell
ipconfig
```
Look for "IPv4 Address" under your active network (WiFi/Ethernet)
Example: `192.168.1.100`

#### Step 2: Run the App with Network Access

Edit `run_app.bat` or run directly:
```powershell
cd src
python app.py
```

Then modify the last line of `src/app.py` from:
```python
app.run(host='127.0.0.1', debug=True, port=5000)
```

To:
```python
app.run(host='0.0.0.0', debug=True, port=5000)
```

#### Step 3: Access from Phone

Make sure both computer and phone are on the **same WiFi network**, then open:
```
http://192.168.1.100:5000
```
(Replace with your actual IP address)

### Option 3: Test Locally on Same Computer
Open the app in your computer's browser:
```
http://127.0.0.1:5000
```

## For Render Deployment

Your Render app should work on phone automatically once deployed. Just use the Render URL!

**No network configuration needed** - Render handles everything for you.
