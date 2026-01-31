import streamlit as st
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(project_root / "scripts"))

st.set_page_config(page_title="Football Analytics", page_icon="⚽", layout="wide")

# Test imports step by step
try:
    from app.core.config import get_settings
    settings = get_settings()
    st.write(f"✅ Config loaded - Season: {settings.current_season}")
except Exception as e:
    st.error(f"Config error: {e}")
    st.stop()

try:
    from app.db.database import SyncSessionLocal
    st.write("✅ Database module imported")
except Exception as e:
    st.error(f"Database import error: {e}")
    import traceback
    st.code(traceback.format_exc())
    st.stop()

try:
    from auth import login_form, get_current_user, logout
    st.write("✅ Auth module imported")
except Exception as e:
    st.error(f"Auth import error: {e}")
    import traceback
    st.code(traceback.format_exc())
    st.stop()

try:
    from pwa import inject_pwa_tags, show_install_prompt
    st.write("✅ PWA module imported")
except Exception as e:
    st.error(f"PWA import error: {e}")
    import traceback
    st.code(traceback.format_exc())
    st.stop()

# Test database connection
try:
    with SyncSessionLocal() as session:
        result = session.execute("SELECT COUNT(*) FROM teams").scalar()
        st.write(f"✅ Database connected - {result} teams found")
except Exception as e:
    st.error(f"Database connection error: {e}")
    import traceback
    st.code(traceback.format_exc())
    st.stop()

st.success("All systems working! Ready to build full app.")
