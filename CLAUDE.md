# FootballAnalytics - EPL Value Bet Finder

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

1. **ELO Rating System** - Team strength rankings
2. **Poisson Distribution** - Goal probability modeling
3. **XGBoost Classifier** - Match outcome prediction

Consensus probabilities are weighted averages of all models.

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

### Proven Profitable Strategies (Backtested 2020-2025)

#### Strategy 1: Away Wins with 5-12% Edge
When consensus model probability exceeds market implied probability by 5-12%, away wins are profitable.

| Metric | Value |
|--------|-------|
| Total Bets (2020-25) | 185 |
| Win Rate | 56.8% |
| ROI | **+16.0%** |
| Edge Range | 5% to 12% |
| Odds Range | 1.50 to 8.00 |

**Criteria:**
```python
edge = consensus_away_prob - (1 / away_odds)
value_bet = (edge >= 0.05) and (edge <= 0.12) and (1.50 <= away_odds <= 8.00)
```

**Season-by-season results:**
| Season | Bets | Win Rate | ROI |
|--------|------|----------|-----|
| 2020-21 | 37 | 54.1% | +9.1% |
| 2021-22 | 46 | 67.4% | +25.5% |
| 2022-23 | 44 | 50.0% | -1.3% |
| 2023-24 | 43 | 58.1% | +20.3% |
| 2024-25 | 14 | 50.0% | +52.4% |

**Enhanced Variant: Exclude Home Form 4-6**
Adding a filter to exclude bets when home team has "poor but not terrible" form (4-6 points) significantly improves ROI:

| Metric | Base | Enhanced |
|--------|------|----------|
| Total Bets | 132 | 85 |
| Win Rate | 51.5% | 54.1% |
| ROI | +18.5% | **+32.9%** |

Why it works: When home team form is 4-6, they're in a slump but haven't fully collapsed. The market may have already adjusted odds, removing the edge. At extremes (<=3 or >=7), the edge remains.

```python
# Enhanced filter
value_bet = (edge >= 0.05) and (edge <= 0.12) and (1.50 <= away_odds <= 8.00)
            and (home_form <= 3 or home_form >= 7)  # Exclude "poor" form 4-6
```

Enhanced variant season results (5/5 profitable):
| Season | Bets | Win Rate | ROI |
|--------|------|----------|-----|
| 2020-21 | 17 | 52.9% | +36.0% |
| 2021-22 | 21 | 61.9% | +53.5% |
| 2022-23 | 22 | 50.0% | +18.5% |
| 2023-24 | 17 | 47.1% | +27.1% |
| 2024-25 | 8 | 62.5% | +24.4% |

#### Strategy 2: Home Wins - Hot Streaks with Negative Edge
**Counterintuitive but profitable.** When home team is on a hot streak AND the market values them MORE than our model, back the home team. Our ELO/Poisson model undervalues momentum; the market prices it better.

| Metric | Value |
|--------|-------|
| Total Bets (2020-25) | 94 |
| Win Rate | 67.0% |
| ROI | **+30.4%** |
| Form Requirement | 12+ points (from last 5 games) |
| Edge Requirement | Negative (model_prob < market_prob) |

**Criteria:**
```python
form_points = sum of points from last 5 games (W=3, D=1, L=0), max 15
edge = consensus_home_prob - (1 / home_odds)
value_bet = (form_points >= 12) and (edge < 0)  # Market sees MORE value than model
```

**Season-by-season results:**
| Season | Bets | Win Rate | ROI |
|--------|------|----------|-----|
| 2020-21 | 11 | 36.4% | -53.2% (Covid season) |
| 2021-22 | 20 | 65.0% | +30.1% |
| 2022-23 | 20 | 75.0% | +71.5% |
| 2023-24 | 17 | 64.7% | +29.8% |
| 2024-25 | 14 | 85.7% | +42.8% |
| 2025-26 | 12 | 66.7% | +25.7% |

**Why it works:** Teams on exceptional form (4+ wins in last 5) have momentum that our statistical models don't capture. When the market prices them as stronger favorites than our model suggests, trust the market - they're pricing intangible factors like confidence and team cohesion.

### Combined Strategy Performance

Both strategies are now implemented in `batch/jobs/backfill_value_bets.py` and stored in the database.

| Strategy | Bets | Win% | ROI |
|----------|------|------|-----|
| Away Win (enhanced) | 85 | 54.1% | +32.9% |
| Home Win (form 12+) | 94 | 67.0% | +30.4% |
| **COMBINED** | **179** | **60.9%** | **+31.6%** |

### Strategies That DON'T Work

| Strategy | Result |
|----------|--------|
| Home wins with positive edge | -19.1% ROI (model overvalues favorites) |
| Draws (any edge) | -5% to -14% ROI |
| Over 2.5 goals | Marginal at best (+2.7% at 5-8% edge, inconsistent) |
| Away wins with edge < 5% | Negative ROI (-9.3%) |
| Away wins with edge > 12% | Lower ROI (+12.4%) - model overconfident |

### Implementation

Value bets are generated by `batch/jobs/backfill_value_bets.py` using the proven strategy.

Kelly Criterion used for stake sizing with fractional Kelly (0.25) for risk management.

## AI Narratives

Claude generates match previews including:
- Form analysis
- Key player matchups
- Historical head-to-head
- Tactical insights
- Betting angle summary

Prompts stored in `batch/ai/prompts/` directory.
