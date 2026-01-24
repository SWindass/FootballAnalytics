# FootballAnalytics - EPL Value Bet Finder

A backend system for analyzing English Premier League matches to identify value betting opportunities using statistical models and AI-generated narratives.

## Features

- **Match Predictions**: Multiple prediction models (ELO, Poisson, XGBoost) combined for consensus probabilities
- **Value Bet Detection**: Automated identification of betting opportunities using Kelly Criterion
- **AI Narratives**: Claude-powered match previews with form analysis and tactical insights
- **REST API**: FastAPI endpoints for accessing predictions and value bets
- **Scheduled Jobs**: Automated weekly analysis, odds refresh, and results updates

## Tech Stack

- **API**: FastAPI with async support
- **Database**: PostgreSQL with SQLAlchemy ORM
- **ML Models**: XGBoost, ELO ratings, Poisson distributions
- **AI**: Anthropic Claude API for narrative generation
- **Infrastructure**: Azure Container Apps, Azure Functions, Bicep IaC

## Quick Start

### Prerequisites

- Python 3.11+
- Docker and Docker Compose
- API keys (see Configuration)

### Local Development

1. Clone the repository:
```bash
cd /home/steve/code/FootballAnalytics
```

2. Copy environment file and add your API keys:
```bash
cp .env.example .env
# Edit .env with your API keys
```

3. Start services:
```bash
docker-compose up -d
```

4. Run database migrations:
```bash
docker-compose exec api alembic upgrade head
```

5. Access the API:
- API: http://localhost:8000
- Docs: http://localhost:8000/docs

### Running Tests

```bash
# Install dev dependencies
pip install -e ".[dev]"

# Run tests
pytest tests/ -v

# With coverage
pytest tests/ --cov=app --cov=batch
```

## API Endpoints

### Matchweek
- `GET /api/v1/matchweek/current` - Current matchweek with predictions
- `GET /api/v1/matchweek/{matchweek}` - Specific matchweek data

### Matches
- `GET /api/v1/match/{id}` - Match details with analysis
- `GET /api/v1/matches` - List matches with filters
- `GET /api/v1/matches/upcoming` - Upcoming matches

### Value Bets
- `GET /api/v1/value-bets` - Current value betting opportunities
- `GET /api/v1/value-bets/{id}` - Specific value bet details

### Teams
- `GET /api/v1/teams` - All EPL teams
- `GET /api/v1/team/{id}/stats` - Team statistics

## Configuration

Required environment variables:

| Variable | Description |
|----------|-------------|
| `DATABASE_URL` | PostgreSQL connection string (async) |
| `DATABASE_URL_SYNC` | PostgreSQL connection string (sync) |
| `FOOTBALL_DATA_API_KEY` | API key from football-data.org |
| `ODDS_API_KEY` | API key from the-odds-api.com |
| `ANTHROPIC_API_KEY` | API key from Anthropic |

Optional settings:

| Variable | Default | Description |
|----------|---------|-------------|
| `EDGE_THRESHOLD` | 0.05 | Minimum edge for value bets (5%) |
| `KELLY_FRACTION` | 0.25 | Fractional Kelly for risk management |
| `MIN_ODDS` | 1.5 | Minimum odds to consider |
| `MAX_ODDS` | 10.0 | Maximum odds to consider |

## Batch Jobs

| Job | Schedule | Description |
|-----|----------|-------------|
| Weekly Analysis | Tuesday 5PM | Generate matchweek predictions |
| Injury Update | Friday 3PM | Update team news |
| Odds Refresh | Saturday 8AM | Capture final odds, detect value |
| Results Update | Every 6 hours | Update scores, recalculate stats |

## Prediction Models

### ELO Rating System
Team strength rankings using modified ELO with:
- Home advantage adjustment
- Goal difference multipliers
- Season regression

### Poisson Distribution
Goal probability modeling for:
- Match outcome probabilities
- Over/Under lines
- Both Teams to Score

### XGBoost Classifier
Machine learning model using features:
- ELO ratings
- Form statistics
- Goal averages
- xG (Expected Goals)
- Home/Away performance

## Project Structure

```
FootballAnalytics/
├── app/                    # FastAPI application
│   ├── api/v1/            # API endpoints
│   ├── core/              # Configuration
│   ├── db/                # Database models & migrations
│   ├── schemas/           # Pydantic models
│   └── services/          # Business logic
├── batch/                  # Batch processing
│   ├── jobs/              # Scheduled jobs
│   ├── models/            # Prediction models
│   ├── ai/                # AI narrative generation
│   ├── data_sources/      # External API clients
│   └── betting/           # Kelly criterion, value detection
├── infra/                  # Azure Bicep templates
├── functions/              # Azure Functions
└── tests/                  # Test suite
```

## Azure Deployment

Estimated monthly cost: ~$30-45

1. Deploy infrastructure:
```bash
az deployment group create \
  --resource-group rg-footballanalytics-prod \
  --template-file infra/main.bicep \
  --parameters postgresPassword=<password> \
               footballDataApiKey=<key> \
               oddsApiKey=<key> \
               anthropicApiKey=<key>
```

2. Build and push container:
```bash
docker build -t footballanalytics-api .
docker tag footballanalytics-api <registry>.azurecr.io/footballanalytics-api:latest
docker push <registry>.azurecr.io/footballanalytics-api:latest
```

## Data Sources

| Source | Data | Cost |
|--------|------|------|
| football-data.org | Fixtures, results | Free (10 req/min) |
| The Odds API | Betting odds | Free (500 req/month) |
| Understat | xG data | Free (scraped) |

## License

MIT License

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Run tests
5. Submit a pull request
