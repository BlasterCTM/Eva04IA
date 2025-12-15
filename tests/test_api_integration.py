from fastapi.testclient import TestClient
from src.serving.api import app


def test_predict_endpoint():
    client = TestClient(app)
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

    r = client.post("/predict", json=payload)
    assert r.status_code == 200, r.text
    data = r.json()
    assert "predictions" in data and data["n"] == 1


def test_reports_and_version():
    client = TestClient(app)
    r = client.get("/reports/detail")
    assert r.status_code in (200, 404)
    r2 = client.get("/reports/season")
    assert r2.status_code in (200, 404)
    r3 = client.get("/model/version")
    assert r3.status_code in (200, 404)
