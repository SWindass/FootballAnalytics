"""Football Analytics Dashboard - Login Page."""

import streamlit as st

st.set_page_config(
    page_title="Football Analytics - Login",
    page_icon="⚽",
    layout="wide",
)

st.title("⚽ Football Analytics")
st.caption("Testing deployment...")

# Debug info
import sys
from pathlib import Path
st.write(f"Python version: {sys.version}")
st.write(f"Working directory: {Path.cwd()}")
st.write(f"Script location: {Path(__file__).parent}")

# Try importing
try:
    project_root = Path(__file__).parent.parent
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))
    st.write(f"Added to path: {project_root}")

    from app.core.config import get_settings
    settings = get_settings()
    st.success(f"Config loaded! Season: {settings.current_season}")
    st.write(f"DB URL: {settings.database_url_sync[:50]}...")
except Exception as e:
    st.error(f"Error: {type(e).__name__}: {e}")
    import traceback
    st.code(traceback.format_exc())
