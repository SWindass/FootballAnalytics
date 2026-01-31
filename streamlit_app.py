"""Entry point for Streamlit Cloud deployment."""

import sys
from pathlib import Path

# Add project root to Python path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(project_root / "scripts"))

import streamlit as st

st.set_page_config(
    page_title="Football Analytics",
    page_icon="⚽",
    layout="wide",
)

# Now import and run the app
try:
    from auth import login_form, get_current_user, logout
    from pwa import inject_pwa_tags, show_install_prompt

    inject_pwa_tags()

    if st.query_params.get("logout"):
        logout()

    user = get_current_user()

    if not user:
        st.title("⚽ Football Analytics")
        st.caption("EPL Value Bet Finder")
        login_form()
        show_install_prompt()
    else:
        st.switch_page("scripts/pages/1_📅_Fixtures.py")

except Exception as e:
    st.error(f"Startup error: {e}")
    import traceback
    st.code(traceback.format_exc())
    st.write(f"sys.path: {sys.path}")
