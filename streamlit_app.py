import streamlit as st
import sys
from pathlib import Path

st.write("Step 1: Basic imports work")

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))
st.write(f"Step 2: Added {project_root} to path")

# Try config import
try:
    from app.core.config import get_settings
    settings = get_settings()
    st.write(f"Step 3: Config loaded - Season: {settings.current_season}")
    st.write(f"Step 4: DB URL starts with: {settings.database_url_sync[:30]}...")
except Exception as e:
    st.error(f"Config error: {e}")
    import traceback
    st.code(traceback.format_exc())
