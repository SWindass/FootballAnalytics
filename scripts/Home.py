"""Football Analytics Dashboard - Home Page."""

import streamlit as st

st.set_page_config(page_title="Football Analytics", page_icon="⚽", layout="wide")
st.title("⚽ Football Analytics Dashboard")
st.write("Testing basic Streamlit startup...")

# Test imports one by one
import sys
st.write(f"Python version: {sys.version}")

try:
    from datetime import datetime, timezone
    st.success("✓ datetime imported")
except Exception as e:
    st.error(f"✗ datetime: {e}")

try:
    from sqlalchemy import select
    st.success("✓ sqlalchemy imported")
except Exception as e:
    st.error(f"✗ sqlalchemy: {e}")

try:
    import os
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    st.write(f"Path: {sys.path[:3]}")
except Exception as e:
    st.error(f"✗ path setup: {e}")

try:
    from app.core.config import get_settings
    settings = get_settings()
    st.success("✓ config loaded")
    st.write(f"DB URL: {settings.database_url_sync[:50]}...")
except Exception as e:
    st.error(f"✗ config: {e}")
    import traceback
    st.code(traceback.format_exc())

try:
    from app.db.database import SyncSessionLocal
    st.success("✓ database imported")
except Exception as e:
    st.error(f"✗ database: {e}")
    import traceback
    st.code(traceback.format_exc())

try:
    from app.db.models import Team
    st.success("✓ models imported")
except Exception as e:
    st.error(f"✗ models: {e}")
    import traceback
    st.code(traceback.format_exc())

try:
    with SyncSessionLocal() as session:
        from sqlalchemy import select
        count = session.execute(select(Team)).scalars().all()
        st.success(f"✓ Database connected! Found {len(count)} teams")
except Exception as e:
    st.error(f"✗ database connection: {e}")
    import traceback
    st.code(traceback.format_exc())

st.write("Done!")
