"""Modulo de procesamiento de datos para pronostico de demanda."""

from __future__ import annotations

from pathlib import Path

import pandas as pd


def cargar_y_explorar(ruta_csv: Path) -> pd.DataFrame:
    """Carga el conjunto de datos y muestra diagnosticos basicos."""
    df = pd.read_csv(ruta_csv)

    print("\n[Exploracion] Resumen del conjunto de datos:")
    print(df.info())
    print("\n[Exploracion] Filas de ejemplo:")
    print(df.head())

    return df


def limpiar_datos(df: pd.DataFrame) -> pd.DataFrame:
    """Convierte columnas numericas, maneja valores faltantes y ordena por fecha."""
    columnas_numericas = [
        "Inventory Level",
        "Units Sold",
        "Units Ordered",
        "Demand Forecast",
        "Price",
        "Discount",
        "Competitor Pricing",
    ]

    for columna in columnas_numericas:
        df[columna] = pd.to_numeric(df[columna], errors="coerce")

    print("\n[Limpieza] Conteo de NaN tras convertir a numerico:")
    print(df[columnas_numericas].isna().sum())

    df = df.dropna(subset=["Units Sold"]).copy()

    columnas_imputacion = [
        "Inventory Level",
        "Units Ordered",
        "Demand Forecast",
        "Price",
        "Discount",
        "Competitor Pricing",
    ]
    for columna in columnas_imputacion:
        df[columna] = df[columna].fillna(df[columna].median())

    df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
    df = df.dropna(subset=["Date"]).copy()
    df = df.sort_values(by=["Store ID", "Product ID", "Date"]).reset_index(drop=True)

    return df
