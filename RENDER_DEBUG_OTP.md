# Debugging OTP Issues on Render

## Current Issue
OTP emails work on localhost but not on Render deployment.

## Step 1: Push Latest Changes
You have uncommitted changes that fix the SMTP timeout issue:

```bash
# Check what changed
git diff src/auth.py

# Add and commit the changes
git add src/auth.py README.md RENDER_DEPLOYMENT.md MOBILE_TESTING.md run_app.bat render.yaml requirements.txt
git commit -m "Fix database paths, add SMTP timeout, and document deployment"

# Push to trigger Render deployment
git push
```

## Step 2: Check Render Logs

After pushing, Render will automatically redeploy. **Check the logs** to see what's happening:

### How to View Render Logs:

1. Go to [Render Dashboard](https://dashboard.render.com)
2. Click on your **weather-app** service
3. Click on **Logs** tab (left sidebar)
4. Try signing up with OTP on your Render URL
5. Watch the logs in real-time

### What to Look For in Logs:

**✅ Success - Email sent:**
```
[OK] OTP sent successfully to koiladaganesh43683@gmail.com
```

**⚠️ Fallback - Email failed, but OTP shown in logs:**
```
[ERROR] SMTP error: timeout
[FALLBACK] OTP for koiladaganesh43683@gmail.com: 123456
```
→ **If you see this**, copy the OTP from the logs and paste it in the verification field

**❌ Error - Authentication failed:**
```
[ERROR] Email authentication failed. Check your Gmail App Password.
[FALLBACK] OTP for koiladaganesh43683@gmail.com: 123456
```
→ Gmail credentials might not be working from Render's servers

## Step 3: Common Issues & Solutions

### Issue 1: Gmail SMTP Blocked by Firewall
**Symptom:** Timeout errors in Render logs  
**Cause:** Render's servers might be blocked by Gmail  
**Solution:** OTP will be printed in logs - copy it from there

### Issue 2: Gmail App Password Wrong
**Symptom:** Authentication errors in logs  
**Cause:** App password in `src/auth.py` line 55 is incorrect  
**Solution:** 
- Go to https://myaccount.google.com/apppasswords
- Generate new app password
- Update line 55 in `src/auth.py`
- Commit and push

### Issue 3: Email in Spam
**Symptom:** No errors in logs, but email doesn't arrive  
**Cause:** Gmail filters it as spam  
**Solution:** Check your spam/junk folder

### Issue 4: Database Still Empty
**Symptom:** OTP request fails completely  
**Cause:** Database not initialized  
**Fix:** Already applied - `init_db()` creates directory now

## Step 4: Temporary Workaround

If email continues to fail on Render, you can use the **terminal logs workaround**:

1. Request OTP on Render website
2. Open Render logs immediately
3. Look for: `[FALLBACK] OTP for your@email.com: 123456`
4. Copy the 6-digit OTP from logs
5. Paste into verification field
6. Complete signup

## Step 5: Environment Variables (Advanced)

For better security, you can set email credentials as Render environment variables:

1. In Render Dashboard → Your Service → Environment
2. Add variables:
   - `SMTP_EMAIL` = `koiladaganesh43683@gmail.com`
   - `SMTP_PASSWORD` = `ccyh mrps evjb dzlz`
3. Update `src/auth.py` lines 58-59 to use these instead of hardcoded values

## Quick Test Checklist

After deploying:
- [ ] Render deployment succeeded (no build errors)
- [ ] Open Render logs tab
- [ ] Visit your Render URL
- [ ] Click "Sign up" and enter email
- [ ] Click "Send OTP"
- [ ] Watch logs for OTP code or error message
- [ ] Either:
  - ✅ Email arrives in inbox, or
  - ⚠️ OTP shown in logs (copy and paste it)
