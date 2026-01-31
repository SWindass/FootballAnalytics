from collections.abc import AsyncGenerator
import os

from sqlalchemy import create_engine
from sqlalchemy.orm import DeclarativeBase, sessionmaker

from app.core.config import get_settings


class Base(DeclarativeBase):
    """Base class for all SQLAlchemy models."""

    pass


settings = get_settings()

# Sync engine for Alembic migrations, batch jobs, and Streamlit
sync_engine = create_engine(
    settings.database_url_sync,
    echo=settings.debug,
    pool_pre_ping=True,
)

SyncSessionLocal = sessionmaker(
    bind=sync_engine,
    autocommit=False,
    autoflush=False,
)

# Async engine for FastAPI (only create if not in Streamlit context)
# This avoids import errors when asyncpg isn't configured
async_engine = None
AsyncSessionLocal = None

def _init_async_engine():
    """Lazily initialize async engine when needed."""
    global async_engine, AsyncSessionLocal
    if async_engine is None:
        from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine
        async_engine = create_async_engine(
            str(settings.database_url),
            echo=settings.debug,
            pool_pre_ping=True,
            pool_size=5,
            max_overflow=10,
        )
        AsyncSessionLocal = async_sessionmaker(
            bind=async_engine,
            class_=AsyncSession,
            expire_on_commit=False,
            autocommit=False,
            autoflush=False,
        )


async def get_async_session():
    """Dependency for FastAPI routes to get async database session."""
    _init_async_engine()
    from sqlalchemy.ext.asyncio import AsyncSession
    async with AsyncSessionLocal() as session:
        try:
            yield session
        finally:
            await session.close()


def get_sync_session():
    """Get sync session for batch jobs."""
    session = SyncSessionLocal()
    try:
        yield session
    finally:
        session.close()
