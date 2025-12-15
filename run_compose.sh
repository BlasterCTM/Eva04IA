#!/usr/bin/env bash
set -euo pipefail
# Build and run with docker compose, stop/remove existing container name if present
echo "Stopping and removing existing container 'eva02_app' (if any)..."
docker rm -f eva02_app || true
echo "Starting docker compose (build) in detached mode..."
docker compose up --build -d
echo "Showing container status:"
docker compose ps
echo "Tailing logs (press Ctrl+C to exit)..."
docker compose logs -f
