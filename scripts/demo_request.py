"""Demo de uso de la API de inferencia.

Este script intenta llamar al endpoint `/predict` en http://127.0.0.1:8000.
- Si el servidor está levantado, hace la petición por HTTP (requiere `requests`).
- Si no, usa `TestClient` para llamar a la app FastAPI directamente.

Ejecutar:
    python scripts/demo_request.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

# Ensure project root is importable (so `import src` works when running the script)
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, ROOT.as_posix())


payload = {
    "records": [
        {
            "Date": "2022-01-01",
            "Store ID": "S001",
            "Product ID": "P0001",
            "Category": "Groceries",
            "Region": "North",
            "Weather Condition": "Sunny",
            "Holiday/Promotion": 0,
            "Seasonality": "Summer",
            "Inventory Level": 100,
            "Units Ordered": 10,
            "Demand Forecast": 90.0,
            "Price": 9.99,
            "Discount": 0,
            "Competitor Pricing": 9.5,
        }
    ]
}


def call_http():
    try:
        import requests

        url = "http://127.0.0.1:8000/predict"
        r = requests.post(url, json=payload, timeout=10)
        print("HTTP status:", r.status_code)
        print(json.dumps(r.json(), indent=2, ensure_ascii=False))
        return True
    except Exception as exc:
        print("HTTP call failed:", exc)
        return False


def call_testclient():
    try:
        from fastapi.testclient import TestClient
        from src.serving.api import app

        client = TestClient(app)
        r = client.post("/predict", json=payload)
        print("TestClient status:", r.status_code)
        print(json.dumps(r.json(), indent=2, ensure_ascii=False))
        return True
    except Exception as exc:
        print("TestClient call failed:", exc)
        return False


if __name__ == "__main__":
    print("Intentando llamar por HTTP a http://127.0.0.1:8000/predict...")
    ok = call_http()
    if not ok:
        print("Servidor no disponible por HTTP; usando TestClient internamente...")
        ok2 = call_testclient()
        if not ok2:
            print("No fue posible ejecutar la demo.")
            sys.exit(2)
    sys.exit(0)
