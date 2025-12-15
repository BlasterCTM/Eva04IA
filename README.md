# Proyecto Pronóstico de Demanda — Runbook mínimo

Instrucciones rápidas para reproducir el modelo y la API localmente.

Requisitos
- Python 3.8+ (en este workspace se detectó Python 3.14).
- Dependencias en `requirements.txt`.

Instalación

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Entrenar y guardar el modelo

```bash
# Ejecuta el pipeline y genera models/modelo_demanda_v1.joblib
python -m src.train.save_model
```

Levantar la API

```bash
# En un terminal (mantenga abierto)
uvicorn src.serving.api:app --reload --port 8000
```

Probar la API

```bash
curl -X POST "http://127.0.0.1:8000/predict" -H "Content-Type: application/json" \
  -d '{"records":[{"Date":"2022-01-01","Store ID":"S001","Product ID":"P0001","Category":"Groceries","Region":"North","Weather Condition":"Sunny","Holiday/Promotion":0,"Seasonality":"Summer","Inventory Level":100,"Units Ordered":10,"Demand Forecast":90.0,"Price":9.99,"Discount":0,"Competitor Pricing":9.5}] }'
```

Ejecutar tests

```bash
pytest -q
```

Notas
- La API aplica internamente la ingeniería de características (`agregar_caracteristicas`) y rellena columnas faltantes cuando es posible.
- Los logs de la API se guardan en `outputs/logs/api.log`.

Minimal endpoints (solo JSON)
-----------------------------
Si prefieres una versión mínima (sin UI ni `/docs`), hay un módulo que expone únicamente endpoints JSON:

- Levantar la versión mínima:

```bash
# Levanta la API mínima (solo endpoints JSON)
uvicorn src.serving.api_endpoints_only:app --reload --port 8000
```

Nota: el comando `uvicorn src.serving.api:app` también arranca la versión mínima por defecto (wrapper que invoca `api_endpoints_only`).

Despliegue con Docker
---------------------
Se incluye un `Dockerfile` para contenerizar la API mínima.

Construir la imagen localmente:

```bash
# desde la raíz del repo
docker build -t eva02ialsd:latest .
```

Ejecutar contenedor (puerto 8000 por defecto):

```bash
docker run --rm -p 8000:8000 \
  -e MODEL_PATH=models/modelo_demanda_v1.joblib \
  -e LOG_LEVEL=info \
  eva02ialsd:latest
```

CI (GitHub Actions)
--------------------
Se añadió un workflow `CI` que instala dependencias, ejecuta `pytest` y construye la imagen Docker (no hace push).

Variables de entorno
--------------------
Se añadió `.env.example` con variables recomendadas: `PORT`, `MODEL_PATH`, `LOG_LEVEL`.

