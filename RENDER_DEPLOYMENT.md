# Deploying to Render (Free Tier)

## ⚠️ Important: Ephemeral Database Limitation

On Render's **free tier**, the filesystem is **ephemeral** - meaning it gets completely wiped on every deployment. This includes your SQLite database.

### What This Means

**After each deployment:**
- ❌ All user accounts are deleted
- ❌ All login credentials are wiped
- ❌ Users must sign up again with OTP

**What persists:**
- ✅ Your code and application logic
- ✅ Weather/air quality data (fetched fresh)
- ✅ Machine learning models

### User Experience

1. **First time user:**
   - Visit your Render URL: `https://your-app.onrender.com`
   - Click "Sign up" → Enter email → Get OTP
   - Verify OTP → Set username/password
   - Login and use the app

2. **After you redeploy:**
   - Previous users **no longer exist**
   - Everyone must sign up again (fresh database)
   - This is normal behavior on free tier

### When Does Redeployment Happen?

Render redeploys your app when:
- You push new code to GitHub
- You manually trigger a redeploy from Render Dashboard
- Render restarts the service (can happen automatically)

### Solutions

#### Option 1: Upgrade to Paid Plan (~$7/month)
Add persistent disk to keep user data:
- See commented section in `render.yaml`
- Costs ~$7/month for service + $0.25/month for 1GB storage
- User data survives deployments

#### Option 2: Continue with Free Tier (Current)
- Accept that users must re-register after deployments
- Suitable for:
  - Development and testing
  - Demos and portfolios
  - Personal use
  - Projects where user accounts aren't critical

## Testing Your Deployment

### On Computer
Open: `https://your-app.onrender.com`

### On Mobile Phone
1. Make sure you use the **Render URL**, not `127.0.0.1:5000`
2. The Render URL works from any device
3. No network configuration needed

### Troubleshooting

**Problem:** "Wrong username/password" on Render
- **Cause:** Database was wiped in last deployment
- **Solution:** Sign up again with OTP

**Problem:** OTP not arriving
- **Cause:** Database doesn't exist (shouldn't happen with current code)
- **Solution:** Check Render logs for errors

**Problem:** Works locally but not on Render
- **Cause:** You're testing different databases (local vs Render)
- **Solution:** Create account separately on each platform

## Deployment Checklist

- [x] Fixed database path to use `DB_PATH` constant
- [x] Updated `render.yaml` to create data directory
- [x] Configured gunicorn to run from `src/` directory
- [ ] Push code to GitHub
- [ ] Verify deployment succeeds on Render
- [ ] Test signup with OTP on Render URL
- [ ] Test login with username/password
- [ ] (Optional) Test on mobile phone

## Files Modified for Render

- [`src/app.py`](file:///c:/Users/koila/DTI.FINAL/src/app.py) - Fixed database paths
- [`src/auth.py`](file:///c:/Users/koila/DTI.FINAL/src/auth.py) - Added directory creation
- [`render.yaml`](file:///c:/Users/koila/DTI.FINAL/render.yaml) - Build/start commands
- [`requirements.txt`](file:///c:/Users/koila/DTI.FINAL/requirements.txt) - Fixed pandas/tensorflow versions
