"""Prueba todos los endpoints de la API mínima usando TestClient.

Ejecutar desde la raíz del repo con `PYTHONPATH=.` para asegurar
que `src` sea importable.
"""
import json
import traceback

from fastapi.testclient import TestClient

import src.serving.api_endpoints_only as api_mod


def main():
    # Evitar que retrain ejecute el pipeline real durante la prueba
    try:
        api_mod._retrain_task = lambda: None  # type: ignore
    except Exception:
        pass

    client = TestClient(api_mod.app)
    results = {}

    def _safe_json(r):
        try:
            return r.json()
        except Exception:
            return r.text

    # Endpoints a probar
    endpoints = [
        ("GET", "/", None),
        ("GET", "/health", None),
        (
            "POST",
            "/predict",
            {
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
            },
        ),
        ("GET", "/model/version", None),
        ("GET", "/reports/detail", {"rows": 1}),
        ("GET", "/reports/season", None),
        ("GET", "/plots", None),
        ("POST", "/retrain", None),
    ]

    for method, path, payload in endpoints:
        try:
            if method == "GET":
                if isinstance(payload, dict):
                    r = client.get(path, params=payload)
                else:
                    r = client.get(path)
            else:
                r = client.post(path, json=payload)
            results[path] = {"status_code": r.status_code, "body": _safe_json(r)}
        except Exception as e:
            results[path] = {"error": str(e)}

    print(json.dumps(results, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    try:
        main()
    except Exception:
        traceback.print_exc()
        raise

