# FootballAnalytics - EPL Value Bet Finder

> **Important:** Keep this documentation in sync with code changes. When modifying prediction models, betting strategies, database schema, or API endpoints, update the relevant sections of this file.

## Project Overview

Backend system for analyzing English Premier League matches to identify value betting opportunities.

**Tech Stack:**
- FastAPI REST API
- PostgreSQL database
- SQLAlchemy ORM with async support
- XGBoost, ELO ratings, Poisson models for predictions
- Claude API for AI-generated match narratives
- Azure Container Apps deployment

## Project Structure

```
/home/steve/code/FootballAnalytics/
├── app/                    # FastAPI application
│   ├── main.py            # Application entry point
│   ├── core/              # Config, security
│   ├── db/                # Database models, migrations
│   ├── api/v1/            # API endpoints
│   ├── schemas/           # Pydantic models
│   └── services/          # Business logic
├── batch/                  # Batch processing jobs
│   ├── jobs/              # Scheduled tasks
│   ├── models/            # Prediction models (ELO, Poisson, XGBoost)
│   ├── ai/                # Claude narrative generation
│   ├── data_sources/      # API clients for external data
│   └── betting/           # Kelly criterion, value detection
├── infra/                  # Azure Bicep templates
├── functions/              # Azure Functions (timer triggers)
└── tests/                  # Unit and integration tests
```

## Key APIs

### REST Endpoints
- `GET /api/v1/matchweek/current` - Current matchweek with all analyses
- `GET /api/v1/match/{id}` - Single match details + prediction
- `GET /api/v1/value-bets` - Current value betting opportunities
- `GET /api/v1/team/{name}/stats` - Team statistics and form

### Data Sources
- football-data.org - Fixtures, results (free, 10 req/min)
- The Odds API - Betting odds (free 500 req/month)
- Understat - xG data (scraped)

## Database Schema

Key tables:
- `teams` - 20 EPL teams
- `matches` - Fixtures and results
- `elo_ratings` - Team ELO by matchweek
- `team_stats` - Form, injuries, xG averages
- `match_analyses` - Predictions + AI narratives
- `value_bets` - Detected value opportunities
- `odds_history` - Historical odds tracking

## Prediction Models

1. **ELO Rating System** (`batch/models/elo.py`) - Team strength rankings via rating difference → logistic function
2. **Poisson Distribution** (`batch/models/poisson.py`) - Expected goals → goal distribution modeling
3. **Dixon-Coles** (`batch/jobs/backfill_new_models.py:32`) - Poisson with goal correlation adjustment (rho=-0.13)
4. **Pi Rating** (`batch/jobs/backfill_new_models.py:96`) - ELO → expected score conversion
5. **XGBoost Classifier** (`batch/models/xgb_model.py`) - Feature-based ML classifier

### Prediction Pipeline: Models → Bets

```
┌─────────────┐   ┌─────────────┐   ┌─────────────┐
│    ELO      │   │   Poisson   │   │  Dixon-Coles│
│  Ratings    │   │    Model    │   │    Model    │
└──────┬──────┘   └──────┬──────┘   └──────┬──────┘
       │                 │                 │
       └────────┬────────┴────────┬────────┘
                │                 │
                ▼                 ▼
        ┌───────────────────────────────┐
        │      Neural Stacker /         │
        │      Weighted Average         │
        └───────────────┬───────────────┘
                        │
                        ▼
        ┌───────────────────────────────┐
        │    CONSENSUS PROBABILITIES    │
        │  (stored in match_analyses)   │
        └───────────────┬───────────────┘
                        │
        ┌───────────────┴───────────────┐
        │                               │
        ▼                               ▼
┌───────────────┐               ┌───────────────┐
│  Compare to   │               │  Compare to   │
│  AWAY Odds    │               │  HOME Odds    │
└───────┬───────┘               └───────┬───────┘
        │                               │
        ▼                               ▼
┌───────────────┐               ┌───────────────┐
│ Edge 5-12%?   │               │ Form >= 12?   │
│ Exclude form  │               │ Edge < 0?     │
│    4-6?       │               │               │
└───────┬───────┘               └───────┬───────┘
        │                               │
        ▼                               ▼
   VALUE BET:                      VALUE BET:
   Away Win                        Home Win
```

**Step 1: Individual Models** - Each model generates probabilities (home win, draw, away win)

**Step 2: Consensus Calculation** (`weekly_analysis.py:140-180`)
- Neural stacker combines model outputs when available
- Fallback: weighted average (40% market, 60% model average)

**Step 3: Value Detection** (`backfill_value_bets.py`)
```python
market_implied_prob = 1 / bookmaker_odds
edge = consensus_prob - market_implied_prob
```

**Step 4: Strategy Filters**
- **Away Win**: Trust model over market (positive edge 5-12%, exclude home form 4-6)
- **Home Win**: Trust market over model (negative edge, form 12+ momentum)

## Batch Job Schedule

| Job | Schedule | Purpose |
|-----|----------|---------|
| Weekly Analysis | Tuesday 5PM | Generate matchweek predictions |
| Injury Update | Friday 3PM | Update team news |
| Odds Refresh | Saturday 8AM | Final odds capture |
| Results Update | Every 6 hours | Update scores, recalculate ELO |

## Local Development (WSL2 + Docker Desktop)

**Prerequisites:**
1. Docker Desktop running on Windows with WSL2 integration enabled
2. Local PostgreSQL must be stopped (conflicts with Docker on port 5432)

**Quick Start:**
```bash
./start.sh    # Starts database + Streamlit app
./stop.sh     # Stops Streamlit (database keeps running)
```

**Manual Start:**
```bash
# 1. Ensure Docker Desktop is running on Windows
# 2. Stop local PostgreSQL if installed
sudo service postgresql stop

# 3. Start database container
docker-compose up -d db

# 4. Start Streamlit app
source venv/bin/activate
PYTHONPATH=/home/steve/code/FootballAnalytics streamlit run scripts/Home.py --server.port 8501
```

**App URL:** http://localhost:8501

**Note:** The `.env` uses `host.docker.internal` for database connections - this is required for WSL2 to connect to Docker Desktop containers.

## Database Configuration

This project uses **two PostgreSQL databases** that must be kept in sync:

| Database | Purpose | Connection |
|----------|---------|------------|
| **Local (Docker)** | Development & testing | `host.docker.internal:5432` |
| **Neon (Remote)** | Production (Streamlit Cloud) | `ep-weathered-flower-abk2q9k9.eu-west-2.aws.neon.tech` |

### Environment Files

- `.env` - Local Docker database (default for development)
- `.env.neon` - Neon production database credentials (git-ignored)

### Keeping Databases in Sync

When running batch jobs that modify data (backfill, odds refresh, etc.), run against **BOTH** databases:

```bash
# 1. Run against local database (default)
PYTHONPATH=/home/steve/code/FootballAnalytics python3 batch/jobs/backfill_value_bets.py --force

# 2. Run against Neon database
source .env.neon
PYTHONPATH=/home/steve/code/FootballAnalytics python3 batch/jobs/backfill_value_bets.py --force
```

Or use this helper pattern:
```bash
# Run a command against Neon
export DATABASE_URL_SYNC="$(grep DATABASE_URL_SYNC .env.neon | cut -d= -f2-)"
python3 batch/jobs/backfill_value_bets.py --force
```

### Critical Jobs to Sync

These jobs modify database state and should be run on both databases:
- `batch/jobs/backfill_value_bets.py` - Regenerates historical value bets
- `batch/jobs/results_update.py` - Updates match scores
- `batch/jobs/odds_refresh.py` - Captures live odds
- `batch/jobs/weekly_analysis.py` - Generates predictions

## Other Commands

```bash
# Run migrations
alembic upgrade head

# Run API locally
uvicorn app.main:app --reload --port 8000

# Run tests
pytest tests/ -v

# Type checking
mypy app/ batch/

# Linting
ruff check app/ batch/
```

## Environment Variables

Required in `.env`:
```
DATABASE_URL=postgresql+asyncpg://user:pass@localhost:5432/football
FOOTBALL_DATA_API_KEY=your_key
ODDS_API_KEY=your_key
ANTHROPIC_API_KEY=your_key
```

## Value Bet Detection

### STATIC 5% + ERA-BASED FORM OVERRIDE STRATEGY

**2020s Era Performance: ~140 bets, 56% win rate, +30% ROI**

This strategy uses era-based detection rather than regime detection because:
1. Regime detection has a bootstrap problem - can't build ROI history without betting
2. Era-based approach is simpler and validated across 30+ years of data

#### Strategy 1: Away Wins with 5%+ Edge (Always Active)

Trust the model when it identifies value in away teams.

| Metric | 2020s Era |
|--------|-----------|
| Bets/Season | ~23 |
| Win Rate | 55.6% |
| ROI | **+30.2%** |
| Edge Threshold | 5% minimum |
| Odds Range | 1.50 to 8.00 |
| Form Filter | Exclude home form 4-6 |

**Criteria:**
```python
edge = consensus_away_prob - (1 / away_odds)
value_bet = (edge >= 0.05) and (1.50 <= away_odds <= 8.00)
            and home_form not in [4, 5, 6]
```

#### Why 5% Instead of 12%?

Analysis revealed that **lower thresholds are MORE profitable in the 2020s era**:

| Edge Threshold | Bets/Yr | Win% | ROI | Profit/Yr |
|----------------|---------|------|-----|-----------|
| **5%** | **23.4** | **55.6%** | **+30.2%** | **+35.4** |
| 7% | 17.2 | 55.8% | +27.8% | +23.9 |
| 9% | 11.4 | 54.4% | +11.8% | +6.7 |
| 12% (old) | 6.4 | 59.4% | +23.0% | +7.4 |

**Key insight:** The 5% threshold produces **5x more profit** than 12% because:

1. **Model is well-calibrated in 2020s** - The consensus model accurately identifies value
2. **Even small edges are real** - 55.6% win rate at 5% edge proves the model finds true value
3. **Volume compounds** - More bets × decent ROI = significantly more profit
4. **Historical data was different** - Pre-2020s required 12% because model was less calibrated

The 5% threshold is optimal for the modern era where our prediction models have access to better data (xG, advanced stats) and the consensus stacker is well-trained.

#### Strategy 2: Home Form Override (2020s Era Only)

**Only active from season 2020-21 onwards.** When home team is on a hot streak AND the market values them MORE than our model, back the home team.

| Metric | 2020s Era |
|--------|-----------|
| Bets/Season | ~16 |
| Win Rate | 67.0% |
| ROI | **+30.4%** |
| Form Requirement | 12+ points (from last 5 games) |
| Edge Requirement | Negative (model_prob < market_prob) |

**Criteria:**
```python
# Only in 2020s era (season >= "2020-21")
form_points = sum of points from last 5 games (W=3, D=1, L=0), max 15
edge = consensus_home_prob - (1 / home_odds)
value_bet = (form_points >= 12) and (edge < 0) and is_2020s_era()
```

Why this is era-restricted: The home form override was unprofitable pre-2020 (-9.4% ROI) but highly profitable in the 2020s. Teams on hot streaks have momentum that statistical models undervalue.

### Combined Strategy Performance (2020s Era)

| Strategy | Bets/Yr | Win% | ROI | Profit/Yr |
|----------|---------|------|-----|-----------|
| Away 5% Edge | ~23 | 55.6% | +30.2% | +35 units |
| Home Form Override | ~16 | 67.0% | +30.4% | +29 units |
| **COMBINED** | **~39** | **~60%** | **~30%** | **~64 units** |

### Expected Annual Results

| Unit Stake | Bets/Yr | Expected Profit |
|------------|---------|-----------------|
| £10 | ~39 | **~£640/season** |
| £25 | ~39 | **~£1,600/season** |
| £50 | ~39 | **~£3,200/season** |

### Stop-Loss Rules

- **Hard stop if 12-month rolling ROI < -20%**
- **Hard stop if away success rate < 30% over 50+ bets**

### Strategies That DON'T Work

| Strategy | Result |
|----------|--------|
| Home wins with positive edge | -19.1% ROI (model overvalues favorites) |
| Draws (any edge) | -5% to -14% ROI |
| Away wins with edge > 15% | Model overconfident at extreme edges |
| Home form override pre-2020 | -9.4% ROI |

### Implementation

- Value bets generated by `batch/jobs/backfill_value_bets.py`
- Era detection in `batch/betting/era_detection.py`
- Backtest script at `batch/jobs/strategy_backtest.py`
- Kelly Criterion for stake sizing (25% Kelly, max 5% stake)

## AI Narratives

Claude generates match previews including:
- Form analysis
- Key player matchups
- Historical head-to-head
- Tactical insights
- Betting angle summary

Prompts stored in `batch/ai/prompts/` directory.
