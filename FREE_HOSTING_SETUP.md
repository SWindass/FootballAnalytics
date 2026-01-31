# Free Hosting Setup Guide

Host FootballAnalytics for £0/month using Streamlit Cloud + Neon + GitHub Actions.

## Overview

| Component | Service | Cost |
|-----------|---------|------|
| Dashboard | Streamlit Community Cloud | Free |
| Database | Neon PostgreSQL | Free (500MB) |
| Batch Jobs | GitHub Actions | Free (2000 mins/month) |

## Step 1: Set Up Neon Database

1. Go to [neon.tech](https://neon.tech) and sign up
2. Create a new project:
   - Name: `footballanalytics`
   - Region: `eu-west-2` (London) or nearest to you
   - PostgreSQL version: 16
3. Copy the connection string from the dashboard
4. Run migrations:
   ```bash
   # Set the connection string
   export DATABASE_URL_SYNC="postgresql://..."

   # Run Alembic migrations
   alembic upgrade head
   ```
5. Seed initial data:
   ```bash
   python -m batch.jobs.seed_data
   python -m batch.jobs.seed_strategies
   ```

## Step 2: Deploy to Streamlit Cloud

1. Push your code to GitHub (if not already)
2. Go to [share.streamlit.io](https://share.streamlit.io)
3. Click "New app"
4. Select:
   - Repository: `your-username/FootballAnalytics`
   - Branch: `main`
   - Main file: `scripts/Home.py`
5. Click "Advanced settings" > "Secrets"
6. Add your secrets:
   ```toml
   DATABASE_URL_SYNC = "postgresql://..."
   DATABASE_URL = "postgresql+asyncpg://..."
   CURRENT_SEASON = "2025/26"
   ```
7. Click "Deploy"

Your app will be live at: `https://your-app-name.streamlit.app`

## Step 3: Configure GitHub Secrets

For batch jobs to run, add secrets to your GitHub repo:

1. Go to: Repository > Settings > Secrets and variables > Actions
2. Add these secrets:

| Secret Name | Value |
|-------------|-------|
| `DATABASE_URL` | Your Neon async connection string |
| `DATABASE_URL_SYNC` | Your Neon sync connection string |
| `FOOTBALL_DATA_API_KEY` | From football-data.org |
| `ODDS_API_KEY` | From the-odds-api.com |
| `ANTHROPIC_API_KEY` | From Anthropic console |
| `STREAMLIT_APP_URL` | Your Streamlit app URL (for keep-alive) |

## Step 4: Enable GitHub Actions

1. Go to: Repository > Actions
2. Enable workflows if prompted
3. The following schedules are set up:

| Workflow | Schedule | Purpose |
|----------|----------|---------|
| Results Update | Every 6 hours | Update scores, settle bets |
| Weekly Analysis | Tuesday 5PM | Generate predictions |
| Odds Refresh | Fri 6PM, Sat 8AM | Get latest odds |
| Injury Update | Friday 3PM | Update team news |
| Keep Alive | Every 5 mins (7am-11pm) | Prevent cold starts |

## Step 5: Migrate Existing Data (Optional)

If you have data in an existing database:

```bash
# Export from old database
pg_dump -h old-host -U user -d football > backup.sql

# Import to Neon
psql "postgresql://...@neon.tech/football" < backup.sql
```

## Troubleshooting

### Cold starts
The keep-alive workflow pings your app during UK daytime. First load outside these hours may take 10-30 seconds.

### Database connection errors
Neon free tier has connection limits. If you see connection errors:
- Check Neon dashboard for connection count
- The app uses connection pooling, but GitHub Actions may hit limits if multiple workflows run simultaneously

### GitHub Actions failures
Check the Actions tab for logs. Common issues:
- Missing secrets
- API rate limits (especially football-data.org free tier: 10 req/min)

## Costs

| Service | Free Tier Limits |
|---------|------------------|
| Neon | 500MB storage, 0.25 vCPU |
| Streamlit Cloud | Unlimited public apps |
| GitHub Actions | 2000 mins/month (private), unlimited (public) |

For 2-3 users with EPL data only, you'll stay well within these limits.
