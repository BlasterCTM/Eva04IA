Uso rápido de Docker Compose

1) Desde la raíz del repositorio (donde está `docker-compose.yml`), ejecuta:

Bash / Git Bash / WSL:

```bash
./run_compose.sh
```

PowerShell:

```powershell
./run_compose.ps1
```

2) Si prefieres manualmente:

```bash
docker compose up --build -d
docker compose ps
docker compose logs -f
```

3) Probar endpoints (en otra terminal):

```bash
curl -sS http://127.0.0.1:8000/reports/detail?rows=1 | jq .
curl -sS http://127.0.0.1:8000/reports/season | jq .
```

4) Si ves `{"detail":"not found"}` o errores, copia las últimas líneas de `docker compose logs --tail 200` y pégalas aquí.

Notas:
- Los volúmenes montan `./outputs` y `./models` del directorio donde ejecutes `docker compose up`.
- Asegúrate de que `modelo_demanda_v1.joblib` esté en `./models` y los CSV en `./outputs/data`.
