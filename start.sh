#!/bin/bash
# FootballAnalytics Startup Script
#
# Prerequisites:
#   - Docker Desktop must be running (start it from Windows first)
#   - Local PostgreSQL must be stopped (this script handles it)

set -e

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$PROJECT_DIR"

echo "=== FootballAnalytics Startup ==="

# Check Docker is available
if ! docker info > /dev/null 2>&1; then
    echo "ERROR: Docker is not running."
    echo "Please start Docker Desktop on Windows first."
    exit 1
fi
echo "[OK] Docker is running"

# Stop local PostgreSQL if running (it conflicts with Docker on port 5432)
if pgrep -x "postgres" > /dev/null 2>&1; then
    echo "[..] Stopping local PostgreSQL (conflicts with Docker)..."
    sudo service postgresql stop 2>/dev/null || true
fi
echo "[OK] Local PostgreSQL not conflicting"

# Start the database container
echo "[..] Starting PostgreSQL container..."
docker-compose up -d db
sleep 2

# Wait for database to be ready
echo "[..] Waiting for database..."
for i in {1..30}; do
    if docker exec footballanalytics-db-1 pg_isready -U football -d football > /dev/null 2>&1; then
        echo "[OK] Database is ready"
        break
    fi
    if [ $i -eq 30 ]; then
        echo "ERROR: Database failed to start"
        exit 1
    fi
    sleep 1
done

# Verify database has data
TEAM_COUNT=$(docker exec footballanalytics-db-1 psql -U football -d football -t -c "SELECT count(*) FROM teams;" 2>/dev/null | tr -d ' ')
echo "[OK] Database connected - $TEAM_COUNT teams found"

# Kill any existing Streamlit process
pkill -f "streamlit run scripts/Home.py" 2>/dev/null || true

# Activate venv and start Streamlit
echo "[..] Starting Streamlit app..."
source venv/bin/activate
PYTHONPATH="$PROJECT_DIR" streamlit run scripts/Home.py --server.port 8501 &

# Wait for app to be ready
sleep 3
for i in {1..20}; do
    if curl -s -o /dev/null -w "%{http_code}" http://localhost:8501 | grep -q "200"; then
        echo "[OK] App is running"
        break
    fi
    if [ $i -eq 20 ]; then
        echo "WARNING: App may still be starting..."
    fi
    sleep 1
done

echo ""
echo "=== Ready ==="
echo "App: http://localhost:8501"
echo ""
echo "To stop: ./stop.sh"
