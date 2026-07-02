#!/bin/sh
# Lightweight start for the dev/tracing-verification image.
# Starts the internal postgres, waits for it, then runs the backend only
# (no g2-ssr / mcp — not needed for tracing verification).
set -e

APP_PATH=/opt/sqlbot/app

echo "[dev_start] starting internal postgres..."
/usr/local/bin/docker-entrypoint.sh postgres &

echo "[dev_start] waiting for postgres..."
for i in $(seq 1 120); do
    if pg_isready -h 127.0.0.1 -p 5432 >/dev/null 2>&1; then
        echo "[dev_start] postgres ready"
        break
    fi
    sleep 1
done

cd "$APP_PATH"
echo "[dev_start] launching uvicorn..."
exec uvicorn main:app --host 0.0.0.0 --port 8000 --workers 1 --proxy-headers
