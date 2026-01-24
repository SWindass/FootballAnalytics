"""Unified Football Analytics Dashboard."""

import streamlit as st

st.set_page_config(
    page_title="Football Analytics",
    page_icon="",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Page routing
pages = {
    "Upcoming Fixtures": "scripts/fixtures_dashboard.py",
    "Poisson Model": "scripts/poisson_dashboard.py",
    "ELO Ratings": "scripts/elo_dashboard.py",
}

st.sidebar.title("Football Analytics")
st.sidebar.markdown("EPL Value Bet Finder")

selection = st.sidebar.radio("Navigate", list(pages.keys()))

st.sidebar.markdown("---")
st.sidebar.markdown("""
**Quick Commands:**
```bash
# Refresh odds & find value bets
python batch/jobs/odds_refresh.py

# Update predictions
python batch/jobs/weekly_analysis.py
```
""")

# Run selected page
if selection == "Upcoming Fixtures":
    from scripts.fixtures_dashboard import main
    main()
elif selection == "Poisson Model":
    from scripts.poisson_dashboard import main
    main()
elif selection == "ELO Ratings":
    from scripts.elo_dashboard import main
    main()
