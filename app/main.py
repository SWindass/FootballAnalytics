from contextlib import asynccontextmanager
from datetime import datetime

import structlog
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.api.v1 import matches, matchweek, teams, value_bets
from app.core.config import get_settings

settings = get_settings()

# Configure structured logging
structlog.configure(
    processors=[
        structlog.stdlib.filter_by_level,
        structlog.stdlib.add_logger_name,
        structlog.stdlib.add_log_level,
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.processors.StackInfoRenderer(),
        structlog.processors.format_exc_info,
        structlog.processors.JSONRenderer() if settings.is_production else structlog.dev.ConsoleRenderer(),
    ],
    wrapper_class=structlog.stdlib.BoundLogger,
    context_class=dict,
    logger_factory=structlog.stdlib.LoggerFactory(),
    cache_logger_on_first_use=True,
)

logger = structlog.get_logger()


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan handler."""
    logger.info("Starting FootballAnalytics API", environment=settings.environment)
    yield
    logger.info("Shutting down FootballAnalytics API")


app = FastAPI(
    title="FootballAnalytics API",
    description="EPL Value Bet Finder - Match predictions and value betting opportunities",
    version="0.1.0",
    lifespan=lifespan,
    docs_url="/docs" if not settings.is_production else None,
    redoc_url="/redoc" if not settings.is_production else None,
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"] if not settings.is_production else ["https://footballanalytics.com"],
    allow_credentials=True,
    allow_methods=["GET"],
    allow_headers=["*"],
)


@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {
        "status": "healthy",
        "timestamp": datetime.utcnow().isoformat(),
        "version": "0.1.0",
        "environment": settings.environment,
    }


@app.get("/")
async def root():
    """Root endpoint with API information."""
    return {
        "name": "FootballAnalytics API",
        "version": "0.1.0",
        "description": "EPL Value Bet Finder",
        "docs": "/docs" if not settings.is_production else None,
    }


# Include API routers
app.include_router(matchweek.router, prefix="/api/v1", tags=["Matchweek"])
app.include_router(matches.router, prefix="/api/v1", tags=["Matches"])
app.include_router(value_bets.router, prefix="/api/v1", tags=["Value Bets"])
app.include_router(teams.router, prefix="/api/v1", tags=["Teams"])
