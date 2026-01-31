#!/bin/bash
# FootballAnalytics Stop Script

echo "=== Stopping FootballAnalytics ==="

# Stop Streamlit
if pkill -f "streamlit run scripts/Home.py" 2>/dev/null; then
    echo "[OK] Streamlit stopped"
else
    echo "[--] Streamlit was not running"
fi

# Optionally stop database (uncomment if you want to stop it too)
# echo "[..] Stopping database..."
# docker-compose down

echo ""
echo "=== Stopped ==="
echo "Note: Database container is still running. Run 'docker-compose down' to stop it."
