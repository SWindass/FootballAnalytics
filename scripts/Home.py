"""Football Analytics Dashboard - Login Page."""

import sys
from pathlib import Path

# Add project root to path for imports
project_root = Path(__file__).parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

import streamlit as st

# Debug: Show any import errors
try:
    from auth import login_form, get_current_user, logout
    from pwa import inject_pwa_tags, show_install_prompt
except Exception as e:
    st.error(f"Import error: {e}")
    st.code(f"sys.path: {sys.path}")
    st.code(f"project_root: {project_root}")
    st.stop()

st.set_page_config(
    page_title="Football Analytics - Login",
    page_icon="⚽",
    layout="wide",
)

# PWA support
inject_pwa_tags()

# Check for logout request
if st.query_params.get("logout"):
    logout()

# Check if user is logged in
user = get_current_user()

if not user:
    # Show login form
    st.title("⚽ Football Analytics")
    st.caption("EPL Value Bet Finder")
    login_form()
    # Show install prompt on login page
    show_install_prompt()
else:
    # User is logged in - redirect to Fixtures
    st.switch_page("pages/1_📅_Fixtures.py")
