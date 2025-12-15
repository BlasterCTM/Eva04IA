#!/usr/bin/env bash
set -euo pipefail
BASE_URL=${BASE_URL:-http://127.0.0.1:8000}
TIMEOUT=${TIMEOUT:-60}
INTERVAL=2

echo "Esperando servicio en ${BASE_URL}/health (timeout ${TIMEOUT}s)..."
started=0
for i in $(seq 1 $((TIMEOUT/INTERVAL))); do
  if curl -sS "${BASE_URL}/health" | grep -q '"status":'; then
    echo "Servicio disponible"
    started=1
    break
  fi
  sleep ${INTERVAL}
done

if [ "$started" -ne 1 ]; then
  echo "ERROR: el servicio no respondió en ${BASE_URL}/health dentro de ${TIMEOUT}s"
  echo "Salida de 'docker compose ps':"
  docker compose ps
  echo "Últimas 200 líneas de logs:" 
  docker compose logs --tail 200
  exit 2
fi

echo
for path in "/health" "/reports/detail?rows=1" "/reports/season" "/model/version" "/plots"; do
  echo "GET ${BASE_URL}${path}"
  curl -sS "${BASE_URL}${path}" | jq . || curl -sS "${BASE_URL}${path}"
  echo
done

echo "Pruebas completadas" 
