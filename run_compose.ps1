# PowerShell helper to stop existing container and run docker compose
Write-Host "Stopping and removing existing container 'eva02_app' (if any)..."
try { docker rm -f eva02_app -ErrorAction Stop } catch { }
Write-Host "Starting docker compose (build) in detached mode..."
docker compose up --build -d
Write-Host "Container status:"
docker compose ps
Write-Host "Tailing logs (press Ctrl+C to exit)..."
docker compose logs -f
