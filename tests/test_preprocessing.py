import pandas as pd

from src.ingenieria_caracteristicas import agregar_caracteristicas


def test_agregar_caracteristicas_basic():
    df = pd.DataFrame(
        {
            "Date": ["2022-01-01", "2022-01-02"],
            "Store ID": ["S1", "S1"],
            "Product ID": ["P1", "P1"],
            "Units Sold": [10, 12],
        }
    )
    df["Date"] = pd.to_datetime(df["Date"])
    out = agregar_caracteristicas(df)
    assert "mes" in out.columns
    assert "ventas_lag_1" in out.columns
    # The second row's lag 1 should be the first row's Units Sold
    assert out.loc[1, "ventas_lag_1"] == 10
