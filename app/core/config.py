from functools import lru_cache
import os
from typing import Literal

from pydantic import Field, PostgresDsn, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


def _load_streamlit_secrets():
    """Load secrets from Streamlit Cloud if available."""
    try:
        import streamlit as st
        if hasattr(st, 'secrets') and len(st.secrets) > 0:
            # Export Streamlit secrets as environment variables
            for key, value in st.secrets.items():
                if isinstance(value, str):
                    os.environ.setdefault(key.upper(), value)
    except Exception:
        pass  # Not running in Streamlit context


# Load Streamlit secrets before Settings initialization
_load_streamlit_secrets()


class Settings(BaseSettings):
    """Application settings loaded from environment variables."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # Environment
    environment: Literal["development", "staging", "production"] = "development"
    debug: bool = False
    log_level: str = "INFO"

    # Database
    database_url: PostgresDsn = Field(
        default="postgresql+asyncpg://football:football@localhost:5432/football"
    )
    database_url_sync: str = Field(
        default="postgresql://football:football@localhost:5432/football"
    )

    # API Keys
    football_data_api_key: str = Field(default="")
    odds_api_key: str = Field(default="")
    anthropic_api_key: str = Field(default="")

    # Value Bet Settings
    edge_threshold: float = Field(default=0.05, ge=0.0, le=0.5)
    kelly_fraction: float = Field(default=0.25, ge=0.0, le=1.0)
    min_odds: float = Field(default=1.5, ge=1.01)
    max_odds: float = Field(default=10.0, le=100.0)

    # Rate Limiting
    football_data_rate_limit: int = Field(default=10, ge=1)
    odds_api_daily_limit: int = Field(default=500, ge=1)

    # EPL Season
    current_season: str = Field(default="2024-25")
    epl_competition_code: str = Field(default="PL")

    @field_validator("database_url", mode="before")
    @classmethod
    def validate_database_url(cls, v: str) -> str:
        if v and not v.startswith("postgresql"):
            raise ValueError("Database URL must be a PostgreSQL connection string")
        return v

    @property
    def is_production(self) -> bool:
        return self.environment == "production"


@lru_cache
def get_settings() -> Settings:
    """Get cached settings instance."""
    return Settings()
