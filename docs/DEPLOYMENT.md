# Deployment Guide

This guide covers deploying FootballAnalytics to the cloud.

## Quick Start: Supabase + Streamlit Cloud (Free)

**Cost: $0/month** for low-traffic use.

### Step 1: Create Supabase Database

1. Go to [supabase.com](https://supabase.com) and create a free account
2. Create a new project (choose region closest to you, e.g., `eu-west-2` for UK)
3. Wait for the project to be provisioned (~2 minutes)
4. Go to **Project Settings** → **Database** → **Connection string**
5. Copy the **URI** connection string (starts with `postgresql://`)
6. Replace `[YOUR-PASSWORD]` with your database password

Your connection string should look like:
```
postgresql://postgres.[PROJECT-ID]:[PASSWORD]@aws-0-eu-west-2.pooler.supabase.com:6543/postgres
```

### Step 2: Import Your Data

Export your local database and import to Supabase:

```bash
# Export from local Docker database
docker exec footballanalytics-db-1 pg_dump -U football -d football > backup.sql

# Import to Supabase (use connection string from Step 1)
psql "postgresql://postgres.[PROJECT-ID]:[PASSWORD]@aws-0-eu-west-2.pooler.supabase.com:6543/postgres" < backup.sql
```

Alternatively, use Supabase's SQL Editor to run the migrations manually.

### Step 3: Deploy to Streamlit Cloud

1. Push your code to GitHub (if not already):
   ```bash
   git add .
   git commit -m "Prepare for Streamlit Cloud deployment"
   git push origin main
   ```

2. Go to [share.streamlit.io](https://share.streamlit.io) and sign in with GitHub

3. Click **New app** and configure:
   - **Repository:** `your-username/FootballAnalytics`
   - **Branch:** `main`
   - **Main file path:** `scripts/Home.py`
   - **Python version:** `3.11`

4. Click **Advanced settings** and add secrets:
   ```toml
   DATABASE_URL_SYNC = "postgresql://postgres.[PROJECT-ID]:[PASSWORD]@aws-0-eu-west-2.pooler.supabase.com:6543/postgres"
   ```

5. Click **Deploy**

Your app will be live at `https://your-app-name.streamlit.app` in ~5 minutes.

### Step 4: Configure Streamlit Cloud

In your Streamlit Cloud dashboard, go to **Settings**:

1. **Requirements file:** Set to `requirements-streamlit.txt`
2. **Python version:** `3.11`

---

## Migration to Azure (When Ready to Scale)

When you outgrow the free tier or need more reliability, migrate to Azure.

### Cost Estimate (Azure)
| Resource | SKU | Monthly Cost |
|----------|-----|--------------|
| PostgreSQL Flexible Server | B1ms (1 vCore, 2GB) | ~$15-20 |
| Container Apps (Dashboard) | Consumption | Free tier* |
| Container Apps (API) | Consumption | Free tier* |
| Container Registry | Basic | ~$5 |
| **Total** | | **~$20-25** |

*Free tier includes 180,000 vCPU-seconds/month

### Prerequisites

1. **Azure CLI** installed:
   ```bash
   curl -sL https://aka.ms/InstallAzureCLIDeb | sudo bash
   ```

2. **Azure subscription** (free tier available with $200 credit)

3. **Login to Azure:**
   ```bash
   az login
   ```

### Step 1: Create Resource Group

```bash
az group create --name footballanalytics-rg --location uksouth
```

### Step 2: Deploy Infrastructure

```bash
cd infra

# Deploy (you'll be prompted for passwords/API keys)
az deployment group create \
  --resource-group footballanalytics-rg \
  --template-file main.bicep \
  --parameters environment=prod \
  --parameters postgresPassword='YourSecurePassword123!' \
  --parameters footballDataApiKey='your-key' \
  --parameters oddsApiKey='your-key' \
  --parameters anthropicApiKey='your-key'
```

This deploys:
- PostgreSQL Flexible Server
- Container Registry
- Key Vault (for secrets)
- Container Apps Environment
- Container Apps (API + Dashboard)
- Azure Functions (scheduled jobs)

### Step 3: Migrate Database

```bash
# Export from Supabase
pg_dump "postgresql://postgres.[PROJECT-ID]:[PASSWORD]@aws-0-eu-west-2.pooler.supabase.com:6543/postgres" > backup.sql

# Get Azure PostgreSQL hostname from deployment output
POSTGRES_HOST=$(az deployment group show \
  --resource-group footballanalytics-rg \
  --name main \
  --query properties.outputs.postgresHost.value -o tsv)

# Import to Azure
psql "postgresql://faadmin:YourSecurePassword123!@${POSTGRES_HOST}:5432/football?sslmode=require" < backup.sql
```

### Step 4: Build and Push Container Images

```bash
# Get registry name
ACR_NAME=$(az deployment group show \
  --resource-group footballanalytics-rg \
  --name main \
  --query properties.outputs.containerRegistryLoginServer.value -o tsv)

# Login to registry
az acr login --name ${ACR_NAME%%.*}

# Build and push API image
docker build -t ${ACR_NAME}/footballanalytics-api:latest -f Dockerfile .
docker push ${ACR_NAME}/footballanalytics-api:latest

# Build and push Streamlit image
docker build -t ${ACR_NAME}/footballanalytics-streamlit:latest -f Dockerfile.streamlit .
docker push ${ACR_NAME}/footballanalytics-streamlit:latest
```

### Step 5: Update Container Apps

```bash
# Update API container
az containerapp update \
  --name fa-api-prod \
  --resource-group footballanalytics-rg \
  --image ${ACR_NAME}/footballanalytics-api:latest

# Update Dashboard container
az containerapp update \
  --name fa-dashboard-prod \
  --resource-group footballanalytics-rg \
  --image ${ACR_NAME}/footballanalytics-streamlit:latest
```

### Step 6: Get Your URLs

```bash
# Dashboard URL
az containerapp show \
  --name fa-dashboard-prod \
  --resource-group footballanalytics-rg \
  --query properties.configuration.ingress.fqdn -o tsv

# API URL
az containerapp show \
  --name fa-api-prod \
  --resource-group footballanalytics-rg \
  --query properties.configuration.ingress.fqdn -o tsv
```

---

## Rollback Plan

If Azure deployment fails, your Supabase + Streamlit Cloud setup remains intact.

To rollback:
1. Update Streamlit Cloud secrets back to Supabase connection string
2. Redeploy on Streamlit Cloud
3. Delete Azure resource group: `az group delete --name footballanalytics-rg`

---

## Environment Variables Reference

| Variable | Required | Description |
|----------|----------|-------------|
| `DATABASE_URL_SYNC` | Yes | PostgreSQL connection string |
| `DATABASE_URL` | API only | Async PostgreSQL connection (asyncpg) |
| `FOOTBALL_DATA_API_KEY` | Optional | For live data fetching |
| `ODDS_API_KEY` | Optional | For live odds fetching |
| `ANTHROPIC_API_KEY` | Optional | For AI narratives |
| `ENVIRONMENT` | No | `development`, `staging`, `production` |
| `CURRENT_SEASON` | No | Default: `2025-26` |

---

## Troubleshooting

### "Connection refused" on Streamlit Cloud
- Check your Supabase project is not paused (free tier pauses after 1 week of inactivity)
- Verify connection string in secrets includes `?sslmode=require` for Supabase
- Check Supabase dashboard for connection pooler settings

### "Module not found" on Streamlit Cloud
- Ensure `requirements-streamlit.txt` is specified in Streamlit Cloud settings
- Check Python version is set to 3.11

### Azure Container App not starting
- Check container logs: `az containerapp logs show --name fa-dashboard-prod --resource-group footballanalytics-rg`
- Verify image was pushed successfully to ACR
- Check Key Vault access permissions

### Database migration issues
- Ensure `sslmode=require` for Azure PostgreSQL
- Check firewall rules allow your IP (Azure Portal → PostgreSQL → Networking)
