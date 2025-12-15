"""Logica de calculo para los KPIs del dashboard."""

from __future__ import annotations

from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from .visualizacion import _asegurar_directorio


def generar_kpis_resumen(results_df: pd.DataFrame) -> dict:
    """Calcula los KPIs diarios de la seccion de resumen."""
    if results_df.empty or "Date" not in results_df.columns:
        return {
            "fecha_hoy": "",
            "productos_en_riesgo": 0,
            "ventas_pronosticadas": 0.0,
            "capital_inmovilizado": 0.0,
        }

    resultados = results_df.copy()
    resultados["Date"] = pd.to_datetime(resultados["Date"], errors="coerce")
    resultados = resultados.dropna(subset=["Date"])

    if resultados.empty:
        return {
            "fecha_hoy": "",
            "productos_en_riesgo": 0,
            "ventas_pronosticadas": 0.0,
            "capital_inmovilizado": 0.0,
        }

    fecha_hoy = resultados["Date"].max()
    today_df = resultados[resultados["Date"] == fecha_hoy].copy()

    df_riesgo = today_df[today_df["Demanda Modelo"] > today_df["Nivel Inventario"]]
    kpi_productos_riesgo = int(len(df_riesgo))
    kpi_ventas_pronosticadas = float(today_df["Demanda Modelo"].sum())

    today_df["sobrestock_unidades"] = np.clip(
        today_df["Nivel Inventario"] - today_df["Demanda Modelo"],
        a_min=0,
        a_max=None,
    )
    today_df["sobrestock_valor"] = today_df["sobrestock_unidades"] * today_df["Price"]
    kpi_capital_inmovilizado = float(today_df["sobrestock_valor"].sum())

    return {
        "fecha_hoy": fecha_hoy.strftime("%Y-%m-%d") if isinstance(fecha_hoy, datetime) else "",
        "productos_en_riesgo": kpi_productos_riesgo,
        "ventas_pronosticadas": kpi_ventas_pronosticadas,
        "capital_inmovilizado": kpi_capital_inmovilizado,
    }


def generar_lista_riesgo_quiebre(results_df: pd.DataFrame) -> pd.DataFrame:
    """Genera la tabla de productos con alto riesgo de quiebre para el último día."""
    if "Date" not in results_df.columns:
        return pd.DataFrame()

    resultados = results_df.copy()
    if not pd.api.types.is_datetime64_any_dtype(resultados["Date"]):
        resultados["Date"] = pd.to_datetime(resultados["Date"], errors="coerce")

    resultados = resultados.dropna(subset=["Date"])
    if resultados.empty:
        return pd.DataFrame()

    fecha_hoy = resultados["Date"].max()
    today_df = resultados[resultados["Date"] == fecha_hoy].copy()

    df_riesgo = today_df[today_df["Demanda Modelo"] > today_df["Nivel Inventario"]].copy()
    if df_riesgo.empty:
        return pd.DataFrame(columns=[
            "Producto (SKU)",
            "Categoría",
            "Nivel Inventario",
            "Pronóstico (Hoy)",
            "Quiebre (Unidades)",
        ])

    df_riesgo["Quiebre (Unidades)"] = df_riesgo["Demanda Modelo"] - df_riesgo["Nivel Inventario"]

    columnas_tabla = [
        "Product ID",
        "Category",
        "Nivel Inventario",
        "Demanda Modelo",
        "Quiebre (Unidades)",
    ]
    df_final = df_riesgo[columnas_tabla].rename(
        columns={
            "Product ID": "Producto (SKU)",
            "Category": "Categoría",
            "Demanda Modelo": "Pronóstico (Hoy)",
        }
    )

    return df_final.sort_values(by="Quiebre (Unidades)", ascending=False).reset_index(drop=True)


def generar_lista_sobrestock(results_df: pd.DataFrame) -> pd.DataFrame:
    """Genera la tabla de productos con mayor capital inmovilizado para el último día."""
    if "Date" not in results_df.columns:
        return pd.DataFrame()

    resultados = results_df.copy()
    if not pd.api.types.is_datetime64_any_dtype(resultados["Date"]):
        resultados["Date"] = pd.to_datetime(resultados["Date"], errors="coerce")

    resultados = resultados.dropna(subset=["Date"])
    if resultados.empty:
        return pd.DataFrame()

    fecha_hoy = resultados["Date"].max()
    today_df = resultados[resultados["Date"] == fecha_hoy].copy()

    today_df["Sobrestock (Unidades)"] = np.clip(
        today_df["Nivel Inventario"] - today_df["Demanda Modelo"],
        a_min=0,
        a_max=None,
    )
    today_df["Capital Inmovilizado"] = today_df["Sobrestock (Unidades)"] * today_df["Price"]

    df_sobrestock = today_df[today_df["Capital Inmovilizado"] > 0].copy()
    if df_sobrestock.empty:
        return pd.DataFrame(columns=[
            "Producto (SKU)",
            "Categoría",
            "Nivel Inventario",
            "Pronóstico (Hoy)",
            "Capital Inmovilizado",
        ])

    columnas_tabla = [
        "Product ID",
        "Category",
        "Nivel Inventario",
        "Demanda Modelo",
        "Capital Inmovilizado",
    ]
    df_final = df_sobrestock[columnas_tabla].rename(
        columns={
            "Product ID": "Producto (SKU)",
            "Category": "Categoría",
            "Demanda Modelo": "Pronóstico (Hoy)",
        }
    )

    return df_final.sort_values(by="Capital Inmovilizado", ascending=False).reset_index(drop=True)


def generar_predicciones_por_producto(results_df: pd.DataFrame) -> pd.DataFrame:
    """Devuelve las predicciones del ultimo dia por tienda y producto."""
    if "Date" not in results_df.columns:
        return pd.DataFrame()

    resultados = results_df.copy()
    if not pd.api.types.is_datetime64_any_dtype(resultados["Date"]):
        resultados["Date"] = pd.to_datetime(resultados["Date"], errors="coerce")

    resultados = resultados.dropna(subset=["Date"])
    if resultados.empty:
        return pd.DataFrame()

    fecha_hoy = resultados["Date"].max()
    today_df = resultados[resultados["Date"] == fecha_hoy].copy()

    if today_df.empty:
        return pd.DataFrame()

    today_df["Quiebre (Unidades)"] = today_df["Demanda Modelo"] - today_df["Nivel Inventario"]
    today_df["Quiebre (Unidades)"] = today_df["Quiebre (Unidades)"].clip(lower=0)
    today_df["Etiqueta Quiebre"] = np.where(
        today_df["Demanda Modelo"] > today_df["Nivel Inventario"],
        "Sí",
        "No",
    )

    columnas_base = [
        "Store ID",
        "Product ID",
        "Category",
        "Nivel Inventario",
        "Demanda Modelo",
        "Demanda Historica",
        "Unidades Vendidas",
        "Price",
        "Etiqueta Quiebre",
        "Quiebre (Unidades)",
    ]

    columnas_existentes = [col for col in columnas_base if col in today_df.columns]
    if not columnas_existentes:
        return pd.DataFrame()

    tabla = today_df[columnas_existentes].rename(
        columns={
            "Store ID": "Supermercado",
            "Product ID": "Producto (SKU)",
            "Category": "Categoría",
            "Nivel Inventario": "Inventario",
            "Demanda Modelo": "Pronóstico Modelo",
            "Demanda Historica": "Pronóstico Histórico",
            "Unidades Vendidas": "Ventas Reales",
            "Price": "Precio",
            "Etiqueta Quiebre": "Riesgo Quiebre",
        }
    )

    orden = [
        "Supermercado",
        "Producto (SKU)",
        "Categoría",
        "Inventario",
        "Pronóstico Modelo",
        "Pronóstico Histórico",
        "Ventas Reales",
        "Precio",
        "Riesgo Quiebre",
        "Quiebre (Unidades)",
    ]
    columnas_presentes = [col for col in orden if col in tabla.columns]

    tabla = tabla[columnas_presentes].sort_values(
        by=["Supermercado", "Pronóstico Modelo"], ascending=[True, False]
    )

    if "Supermercado" in tabla.columns:
        tabla = tabla.groupby("Supermercado", group_keys=False).head(20)

    return tabla.reset_index(drop=True)


def generar_grafico_ventas_categoria(
    results_df: pd.DataFrame,
    ruta_salida: str = "outputs/plots/dashboard/ventas_por_categoria.png",
) -> None:
    """Genera un grafico de barras con el pronostico de ventas por categoria para el ultimo dia."""
    if "Date" not in results_df.columns:
        print("[Dashboard] No se encontro la columna Date para graficar.")
        return

    resultados = results_df.copy()
    if not pd.api.types.is_datetime64_any_dtype(resultados["Date"]):
        resultados["Date"] = pd.to_datetime(resultados["Date"], errors="coerce")

    resultados = resultados.dropna(subset=["Date"])
    if resultados.empty:
        print("[Dashboard] No hay datos validos para graficar ventas por categoria.")
        return

    fecha_hoy = resultados["Date"].max()
    today_df = resultados[resultados["Date"] == fecha_hoy].copy()

    ventas_categoria = today_df.groupby("Category")["Demanda Modelo"].sum().sort_values(ascending=False)

    if ventas_categoria.empty:
        print("[Dashboard] No hay datos de ventas por categoria para graficar.")
        return

    fig, ax = plt.subplots(figsize=(10, 6))
    ventas_categoria.plot(kind="bar", ax=ax, color="#1f77b4")

    ax.set_title(f"Pronostico de Ventas por Categoria (Hoy: {fecha_hoy.strftime('%Y-%m-%d')})")
    ax.set_ylabel("Unidades pronosticadas")
    ax.set_xlabel("Categoria")
    ax.grid(axis="y", linestyle="--", alpha=0.7)
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()

    ruta_path = Path(ruta_salida)
    _asegurar_directorio(ruta_path)
    fig.savefig(ruta_path, dpi=150)
    plt.close(fig)

    print(f"[Dashboard] Grafico de ventas por categoria guardado en {ruta_path.as_posix()}")
