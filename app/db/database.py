
from sqlalchemy import create_engine
from sqlalchemy.orm import DeclarativeBase, sessionmaker

from app.core.config import get_settings


class Base(DeclarativeBase):
    """Base class for all SQLAlchemy models."""

    pass


# Lazy initialization for sync engine (allows Streamlit secrets to load first)
_sync_engine = None
_sync_session_local = None


def reset_sync_engine():
    """Reset the sync engine and session factory to force reinitialization."""
    global _sync_engine, _sync_session_local
    if _sync_engine is not None:
        _sync_engine.dispose()
    _sync_engine = None
    _sync_session_local = None


def _get_sync_engine():
    """Lazily initialize sync engine when needed."""
    global _sync_engine
    if _sync_engine is None:
        settings = get_settings()
        db_url = settings.database_url_sync

        # Ensure SSL mode for Neon (and other cloud databases)
        if "neon.tech" in db_url and "sslmode" not in db_url:
            separator = "&" if "?" in db_url else "?"
            db_url = f"{db_url}{separator}sslmode=require"

        _sync_engine = create_engine(
            db_url,
            echo=settings.debug,
            pool_pre_ping=True,
        )
    return _sync_engine


def _get_sync_session_local():
    """Lazily initialize sync session factory when needed."""
    global _sync_session_local
    if _sync_session_local is None:
        _sync_session_local = sessionmaker(
            bind=_get_sync_engine(),
            autocommit=False,
            autoflush=False,
        )
    return _sync_session_local


class SyncSessionLocal:
    """Wrapper that lazily initializes the session factory."""

    def __new__(cls):
        return _get_sync_session_local()()


# Async engine for FastAPI (only create if not in Streamlit context)
# This avoids import errors when asyncpg isn't configured
async_engine = None
AsyncSessionLocal = None


def _init_async_engine():
    """Lazily initialize async engine when needed."""
    global async_engine, AsyncSessionLocal
    if async_engine is None:
        settings = get_settings()
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
