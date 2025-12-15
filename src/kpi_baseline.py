"""Modulo para generar analisis de KPIs base a partir del dataset original."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def _asegurar_directorio(ruta: Path) -> None:
    """Crea el directorio padre si no existe."""
    ruta.parent.mkdir(parents=True, exist_ok=True)


def generar_kpis_linea_base(
    df: pd.DataFrame,
    ruta_salida: Path | None = Path("outputs/plots/baseline_kpi_dashboard.png"),
    mostrar: bool = True,
) -> dict[str, dict[str, float] | dict[str, Path]]:
    """Construye graficos y metricas base usando solo el dataset original."""
    if df.empty:
        raise ValueError("El DataFrame de entrada esta vacio; no se pueden generar KPIs.")

    data = df.copy()
    if not np.issubdtype(data["Date"].dtype, np.datetime64):
        data["Date"] = pd.to_datetime(data["Date"], errors="coerce")
    data = data.dropna(subset=["Date"])

    agrupado = (
        data.groupby(["Date", "Product ID"], as_index=False)
        .agg(
            {
                "Units Sold": "sum",
                "Demand Forecast": "sum",
                "Inventory Level": "sum",
            }
        )
        .sort_values("Date")
    )

    agrupado["APE"] = np.where(
        agrupado["Units Sold"] == 0,
        np.nan,
        np.abs(agrupado["Units Sold"] - agrupado["Demand Forecast"]) / agrupado["Units Sold"],
    )
    diario_mape = agrupado.groupby("Date")["APE"].mean()
    rolling_mape = diario_mape.rolling(30, min_periods=1).mean() * 100
    mape_general = diario_mape.mean() * 100

    fig_mape, ax_mape = plt.subplots(figsize=(12, 5))
    ax_mape.plot(rolling_mape.index, rolling_mape.values, color="#1f77b4")
    ax_mape.set_title("Precisión del Pronóstico (MAPE Móvil 30 Días) - Línea Base")
    ax_mape.set_xlabel("Fecha")
    ax_mape.set_ylabel("MAPE (%)")
    ax_mape.grid(True, linestyle="--", linewidth=0.5, alpha=0.7)

    fig_scatter, ax_scatter = plt.subplots(figsize=(6, 6))
    ax_scatter.scatter(
        agrupado["Demand Forecast"],
        agrupado["Units Sold"],
        alpha=0.4,
        edgecolor="none",
    )
    limite_superior = max(
        agrupado["Demand Forecast"].max(),
        agrupado["Units Sold"].max(),
    )
    limites = [0, limite_superior]
    ax_scatter.plot(limites, limites, linestyle="--", color="red", linewidth=1)
    ax_scatter.set_title("Dispersión: Demanda Real vs. Demanda Pronosticada (Línea Base)")
    ax_scatter.set_xlabel("Demanda Pronosticada (Demand Forecast)")
    ax_scatter.set_ylabel("Demanda Real (Units Sold)")
    ax_scatter.set_xlim(limites)
    ax_scatter.set_ylim(limites)
    ax_scatter.grid(True, linestyle=":", linewidth=0.5, alpha=0.6)

    totales_diarios = agrupado.groupby("Date").agg({"Units Sold": "sum", "Inventory Level": "sum"})
    totales_diarios = totales_diarios.rename(
        columns={"Units Sold": "unidades", "Inventory Level": "inventario"}
    )
    totales_diarios["DII"] = totales_diarios["inventario"] / totales_diarios["unidades"].replace(0, np.nan)
    rolling_dii = totales_diarios["DII"].rolling(30, min_periods=1).mean()
    inventario_promedio = totales_diarios["inventario"].mean()
    ventas_promedio = totales_diarios["unidades"].mean()
    if ventas_promedio == 0 or np.isnan(ventas_promedio):
        dii_promedio = np.nan
    else:
        dii_promedio = inventario_promedio / ventas_promedio

    fig_dii, ax_dii = plt.subplots(figsize=(12, 5))
    ax_dii.plot(rolling_dii.index, rolling_dii.values, color="#2ca02c")
    ax_dii.set_title("Tendencia de Días de Inventario (DII Móvil 30 Días) - Línea Base")
    ax_dii.set_xlabel("Fecha")
    ax_dii.set_ylabel("DII (días)")
    ax_dii.grid(True, linestyle="--", linewidth=0.5, alpha=0.7)

    # Rutas de salida específicas por KPI
    png_mape = Path("outputs/plots/1_Precision/baseline_tendencia_mape.png")
    png_scatter = Path("outputs/plots/1_Precision/baseline_scatter_real_vs_forecast.png")
    png_dii = Path("outputs/plots/2_DII/baseline_tendencia_dii.png")
    # Asegurar directorios antes de guardar
    for ruta in (png_mape, png_scatter, png_dii):
        _asegurar_directorio(ruta)

    # Guardar PNGs
    fig_mape.savefig(png_mape, dpi=200)
    fig_scatter.savefig(png_scatter, dpi=200)
    fig_dii.savefig(png_dii, dpi=200)

    print(
        "[Baseline KPI] Figuras guardadas: "
        f"{png_mape.as_posix()}, {png_scatter.as_posix()}, {png_dii.as_posix()}"
    )

    if mostrar:
        plt.show()

    plt.close("all")

    print(f"[Baseline KPI] MAPE general de la línea base: {mape_general:.2f}%")
    print(f"[Baseline KPI] DII promedio general: {dii_promedio:.2f} días")

    metricas = {
        "mape_general": mape_general,
        "dii_promedio": dii_promedio,
    }

    figuras = {
        "baseline_mape": png_mape,
        "baseline_scatter": png_scatter,
        "baseline_dii": png_dii,
    }

    return {
        "metrics": metricas,
        "figures": figuras,
    }
