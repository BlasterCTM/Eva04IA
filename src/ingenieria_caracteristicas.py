"""Modulo de ingenieria de caracteristicas para demanda minorista."""

from __future__ import annotations

import pandas as pd


def agregar_caracteristicas(df: pd.DataFrame) -> pd.DataFrame:
    """Agrega variables de calendario, rezagos y medias moviles."""
    df["mes"] = df["Date"].dt.month
    df["dia_semana"] = df["Date"].dt.dayofweek
    df["dia_mes"] = df["Date"].dt.day
    df["dia_anio"] = df["Date"].dt.dayofyear

    columnas_grupo = ["Store ID", "Product ID"]
    ventas_series = df.groupby(columnas_grupo)["Units Sold"]

    df["ventas_lag_1"] = ventas_series.shift(1)
    df["ventas_lag_7"] = ventas_series.shift(7)
    df["ventas_lag_14"] = ventas_series.shift(14)

    df["media_movil_7d"] = (
        df.groupby(columnas_grupo)["Units Sold"]
        .transform(lambda serie: serie.shift(1).rolling(window=7, min_periods=1).mean())
    )

    columnas_relleno = ["ventas_lag_1", "ventas_lag_7", "ventas_lag_14", "media_movil_7d"]
    df[columnas_relleno] = df[columnas_relleno].fillna(0)

    return df
