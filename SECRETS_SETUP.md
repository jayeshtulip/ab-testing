# GitHub Secrets Setup Guide

## Required Secrets

Go to: https://github.com/YOUR_USERNAME/mlops_production/settings/secrets/actions

### Critical Secrets (Required)

1. **AWS_ACCESS_KEY_ID**
   - Your AWS Access Key ID
   - Used for: DVC data access, MLflow artifacts, S3 operations

2. **AWS_SECRET_ACCESS_KEY**
   - Your AWS Secret Access Key
   - Used for: DVC data access, MLflow artifacts, S3 operations

### Optional Secrets

3. **SLACK_WEBHOOK_URL** (Optional)
   - Slack webhook for notifications
   - Used for: Pipeline notifications, alerts

## Setup Steps

1. Go to your GitHub repository
2. Click **Settings** tab
3. Click **Secrets and variables** → **Actions** in left sidebar
4. Click **New repository secret**
5. Add each secret name and value
6. Click **Add secret**

## Test Your Setup

After adding secrets, test the pipeline:

```bash
# Make a small change
echo "# Test CI/CD" >> README.md
git add README.md
git commit -m "Test CI/CD pipeline"
git push origin main
```

Then check the **Actions** tab to see workflows running.

## Security Notes

- 🔒 Never commit secrets to your repository
- 🔒 Rotate secrets regularly
- 🔒 Monitor secret usage in workflow logs
- 🔒 Use minimum required AWS permissions
