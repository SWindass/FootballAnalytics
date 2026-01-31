"""Database initialization helper for Streamlit pages.

Import this module FIRST in any Streamlit page that needs database access.
It ensures Streamlit secrets are loaded before the database engine is created.
"""
import os
import streamlit as st

# Load Streamlit secrets into environment variables
if hasattr(st, 'secrets') and len(st.secrets) > 0:
    for key, value in st.secrets.items():
        if isinstance(value, str):
            os.environ[key.upper()] = value

# Clear settings cache and reset database engine
from app.core.config import get_settings
get_settings.cache_clear()

from app.db.database import reset_sync_engine
reset_sync_engine()
