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

## Development Commands

```bash
# Start local development
docker-compose up -d

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

A value bet is identified when:
```
model_probability > implied_odds_probability * (1 + edge_threshold)
```

Edge threshold default: 5% (configurable)

Kelly Criterion used for stake sizing with fractional Kelly (0.25) for risk management.

## AI Narratives

Claude generates match previews including:
- Form analysis
- Key player matchups
- Historical head-to-head
- Tactical insights
- Betting angle summary

Prompts stored in `batch/ai/prompts/` directory.
