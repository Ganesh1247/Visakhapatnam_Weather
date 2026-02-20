# Google Cloud Run Deployment Guide
# EcoGlance Weather & Air Quality App

# ================================================
# STEP 1: Install Google Cloud SDK
# Download from: https://cloud.google.com/sdk/docs/install
# After install, run: gcloud init
# ================================================

# ================================================
# STEP 2: Set your project ID
# ================================================
gcloud config set project YOUR_PROJECT_ID

# ================================================
# STEP 3: Enable required APIs
# ================================================
gcloud services enable run.googleapis.com
gcloud services enable cloudbuild.googleapis.com
gcloud services enable containerregistry.googleapis.com

# ================================================
# STEP 4: Build and push Docker image
# ================================================
gcloud builds submit --tag gcr.io/YOUR_PROJECT_ID/ecoglance-app

# ================================================
# STEP 5: Deploy to Cloud Run with all env vars
# ================================================
gcloud run deploy ecoglance-app \
  --image gcr.io/YOUR_PROJECT_ID/ecoglance-app \
  --platform managed \
  --region asia-south1 \
  --allow-unauthenticated \
  --memory 2Gi \
  --cpu 2 \
  --timeout 300 \
  --set-env-vars "SUPABASE_URL=https://mhzmapbbfaukwppnrahr.supabase.co" \
  --set-env-vars "SUPABASE_KEY=eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJzdXBhYmFzZSIsInJlZiI6Im1oem1hcGJiZmF1a3dwcG5yYWhyIiwicm9sZSI6ImFub24iLCJpYXQiOjE3NzE1NTU3MTgsImV4cCI6MjA4NzEzMTcxOH0.o6UFA4l5ec6eFYHQrctmqYzWkglYMGKb-51_meceEaw" \
  --set-env-vars "SUPABASE_SERVICE_KEY=eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJzdXBhYmFzZSIsInJlZiI6Im1oem1hcGJiZmF1a3dwcG5yYWhyIiwicm9sZSI6InNlcnZpY2Vfcm9sZSIsImlhdCI6MTc3MTU1NTcxOCwiZXhwIjoyMDg3MTMxNzE4fQ.nbQwG32NroucPR24qhyeadJintvYQdmiNB5buwZKY1Y" \
  --set-env-vars "SMTP_EMAIL=ganeshkoilada1247@gmail.com" \
  --set-env-vars "SMTP_PASSWORD=ifsy uivk jzpt jjce" \
  --set-env-vars "SECRET_KEY=ecoglance-super-secret-key-2024"

# ================================================
# STEP 6: Get your deployed URL
# ================================================
gcloud run services describe ecoglance-app --region asia-south1 --format "value(status.url)"

# ================================================
# NOTES:
# - Replace YOUR_PROJECT_ID with your actual GCP project ID
# - Region asia-south1 = Mumbai (closest to Visakhapatnam)
# - Free tier: 2 million requests/month, 360,000 GB-seconds/month
# - First deployment takes ~5-10 minutes (TensorFlow is large)
# ================================================
