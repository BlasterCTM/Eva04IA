"""Analisis de indicadores de sobreinventario y precision de predicciones."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def _asegurar_directorio(ruta: Path) -> None:
    """Garantiza que el directorio padre exista."""
    ruta.parent.mkdir(parents=True, exist_ok=True)


def analizar_sobrestock_y_precision(
    resultados: pd.DataFrame,
    mostrar: bool = True,
) -> dict[str, dict[str, Path] | dict[str, float]]:
    """Genera graficas y metricas de sobreinventario, OOS, MAE y RMSE."""
    if resultados.empty:
        raise ValueError("No hay resultados para calcular KPIs de sobreinventario.")

    datos = resultados.copy()

    # --- Sobreinventario base y modelo
    datos["sobre_base"] = np.clip(
        datos["Nivel Inventario"] - datos["Demanda Historica"], a_min=0, a_max=None
    )
    datos["sobre_modelo"] = np.clip(
        datos["Nivel Inventario"] - datos["Demanda Modelo"], a_min=0, a_max=None
    )

    agregados = datos.groupby("Date").agg(
        sobre_base=("sobre_base", "sum"),
        sobre_modelo=("sobre_modelo", "sum"),
        inventario=("Nivel Inventario", "sum"),
    ).sort_index()

    agregados["pct_sobre_base"] = np.where(
        agregados["inventario"] > 0,
        (agregados["sobre_base"] / agregados["inventario"]) * 100,
        np.nan,
    )
    agregados["pct_sobre_modelo"] = np.where(
        agregados["inventario"] > 0,
        (agregados["sobre_modelo"] / agregados["inventario"]) * 100,
        np.nan,
    )

    promedio_pct_base = (
        (agregados["sobre_base"].sum() / agregados["inventario"].sum()) * 100
        if agregados["inventario"].sum() > 0
        else np.nan
    )
    promedio_pct_modelo = (
        (agregados["sobre_modelo"].sum() / agregados["inventario"].sum()) * 100
        if agregados["inventario"].sum() > 0
        else np.nan
    )

    promedio_pred_base = datos["Demanda Historica"].mean()
    promedio_pred_modelo = datos["Demanda Modelo"].mean()

    # --- Metricas por fila y diarias: OOS, MAE y RMSE (base vs modelo)
    datos["residuo_base"] = datos["Unidades Vendidas"] - datos["Demanda Historica"]
    datos["residuo_modelo"] = datos["Unidades Vendidas"] - datos["Demanda Modelo"]
    datos["ae_base"] = np.abs(datos["residuo_base"])
    datos["ae_modelo"] = np.abs(datos["residuo_modelo"])
    datos["se_base"] = datos["residuo_base"] ** 2
    datos["se_modelo"] = datos["residuo_modelo"] ** 2
    datos["oos_base"] = datos["Nivel Inventario"] < datos["Demanda Historica"]
    datos["oos_modelo"] = datos["Nivel Inventario"] < datos["Demanda Modelo"]

    metricas = datos.groupby("Date").agg(
        tasa_oos_base=("oos_base", "mean"),
        tasa_oos_modelo=("oos_modelo", "mean"),
        mae_base=("ae_base", "mean"),
        mae_modelo=("ae_modelo", "mean"),
        rmse_base=("se_base", lambda s: float(np.sqrt(np.mean(s)))),
        rmse_modelo=("se_modelo", lambda s: float(np.sqrt(np.mean(s)))),
    )

    # --- Grafico 1: Sobrestock Base (valores absolutos)
    fig_sobrestock_base, ax_base = plt.subplots(figsize=(12, 5))
    ax_base.plot(agregados.index, agregados["sobre_base"], color="#1f77b4")
    ax_base.set_title("Tendencia Sobrestock - Línea Base")
    ax_base.set_ylabel("Sobrestock (unidades)")
    ax_base.set_xlabel("Fecha")
    ax_base.grid(True, linestyle="--", linewidth=0.5, alpha=0.7)

    # --- Grafico 2: Sobrestock Modelo (valores absolutos)
    fig_sobrestock_modelo, ax_modelo = plt.subplots(figsize=(12, 5))
    ax_modelo.plot(agregados.index, agregados["sobre_modelo"], color="#ff7f0e")
    ax_modelo.set_title("Tendencia Sobrestock - Modelo XGBoost")
    ax_modelo.set_ylabel("Sobrestock (unidades)")
    ax_modelo.set_xlabel("Fecha")
    ax_modelo.grid(True, linestyle="--", linewidth=0.5, alpha=0.7)

    # --- Grafico 3: Sobrestock Comparativo (valores absolutos)
    fig_sobrestock_comparativo, ax_sobre_comp = plt.subplots(figsize=(12, 5))
    ax_sobre_comp.plot(agregados.index, agregados["sobre_base"], label="Línea Base", color="#1f77b4", linewidth=1.5)
    ax_sobre_comp.plot(agregados.index, agregados["sobre_modelo"], label="Modelo XGBoost", color="#ff7f0e", linewidth=1.5, linestyle="--")
    ax_sobre_comp.set_title("Comparativa Tendencia Sobrestock")
    ax_sobre_comp.set_ylabel("Sobrestock (unidades)")
    ax_sobre_comp.set_xlabel("Fecha")
    ax_sobre_comp.grid(True, linestyle="--", linewidth=0.5, alpha=0.7)
    ax_sobre_comp.legend(loc="upper left")

    # --- Grafico 4: OOS comparativo
    fig_oos, ax_oos = plt.subplots(figsize=(12, 5))
    ax_oos.plot(metricas.index, metricas["tasa_oos_base"] * 100, label="OOS Base", color="#1f77b4", linewidth=1.5)
    ax_oos.plot(metricas.index, metricas["tasa_oos_modelo"] * 100, label="OOS Modelo", color="#ff7f0e", linewidth=1.5, linestyle="--")
    ax_oos.set_title("Tendencia Tasa de OOS (Quiebre de Stock)")
    ax_oos.set_ylabel("Tasa OOS (%)")
    ax_oos.set_xlabel("Fecha")
    ax_oos.grid(True, linestyle="--", linewidth=0.5, alpha=0.7)
    ax_oos.legend(loc="upper left")

    # --- Grafico 5: MAE comparativo
    fig_mae, ax_mae = plt.subplots(figsize=(12, 5))
    ax_mae.plot(metricas.index, metricas["mae_base"], label="MAE Base", color="#1f77b4", linewidth=1.5)
    ax_mae.plot(metricas.index, metricas["mae_modelo"], label="MAE Modelo", color="#ff7f0e", linewidth=1.5, linestyle="--")
    ax_mae.set_title("Tendencia MAE (Error Absoluto Medio)")
    ax_mae.set_ylabel("MAE (unidades)")
    ax_mae.set_xlabel("Fecha")
    ax_mae.grid(True, linestyle="--", linewidth=0.5, alpha=0.7)
    ax_mae.legend(loc="upper left")

    # --- Grafico 6: RMSE comparativo
    fig_rmse, ax_rmse = plt.subplots(figsize=(12, 5))
    ax_rmse.plot(metricas.index, metricas["rmse_base"], label="RMSE Base", color="#1f77b4", linewidth=1.5)
    ax_rmse.plot(metricas.index, metricas["rmse_modelo"], label="RMSE Modelo", color="#ff7f0e", linewidth=1.5, linestyle="--")
    ax_rmse.set_title("Tendencia RMSE (Error Cuadratico Medio)")
    ax_rmse.set_ylabel("RMSE (unidades)")
    ax_rmse.set_xlabel("Fecha")
    ax_rmse.grid(True, linestyle="--", linewidth=0.5, alpha=0.7)
    ax_rmse.legend(loc="upper left")

    # --- Rutas de guardado por KPI (6 rutas)
    png_sobre_base = Path("outputs/plots/4_Sobrestock/sobrestock_tendencia_baseline.png")
    png_sobre_modelo = Path("outputs/plots/4_Sobrestock/sobrestock_tendencia_modelo.png")
    png_sobre_comp = Path("outputs/plots/4_Sobrestock/comparativa_tendencia_sobrestock.png")
    png_oos = Path("outputs/plots/3_OOS/comparativa_tendencia_oos.png")
    png_mae = Path("outputs/plots/1_Precision/comparativa_tendencia_mae.png")
    png_rmse = Path("outputs/plots/1_Precision/comparativa_tendencia_rmse.png")

    for ruta in (png_sobre_base, png_sobre_modelo, png_sobre_comp, png_oos, png_mae, png_rmse):
        _asegurar_directorio(ruta)

    # --- Guardado
    fig_sobrestock_base.savefig(png_sobre_base, dpi=200)
    fig_sobrestock_modelo.savefig(png_sobre_modelo, dpi=200)
    fig_sobrestock_comparativo.savefig(png_sobre_comp, dpi=200)
    fig_oos.savefig(png_oos, dpi=200)
    fig_mae.savefig(png_mae, dpi=200)
    fig_rmse.savefig(png_rmse, dpi=200)

    print(
        "[KPI Sobrestock] Figuras guardadas: "
        f"{png_sobre_base.as_posix()}, {png_sobre_modelo.as_posix()}, {png_sobre_comp.as_posix()}, "
        f"{png_oos.as_posix()}, {png_mae.as_posix()}, {png_rmse.as_posix()}"
    )

    if mostrar:
        plt.show()

    plt.close("all")

    figuras = {
        "sobrestock_baseline": png_sobre_base,
        "sobrestock_modelo": png_sobre_modelo,
        "sobrestock_comparativo": png_sobre_comp,
        "oos_comparativo": png_oos,
        "mae_comparativo": png_mae,
        "rmse_comparativo": png_rmse,
    }

    # --- Resumen imprimible y retorno (se mantiene)
    resumen = pd.DataFrame(
        {
            "Indicador": [
                "Sobreinventario promedio línea base (%)",
                "Sobreinventario promedio modelo (%)",
                "Promedio predicciones línea base",
                "Promedio predicciones modelo",
            ],
            "Valor": [
                round(float(promedio_pct_base), 2) if not np.isnan(promedio_pct_base) else np.nan,
                round(float(promedio_pct_modelo), 2) if not np.isnan(promedio_pct_modelo) else np.nan,
                round(float(promedio_pred_base), 2),
                round(float(promedio_pred_modelo), 2),
            ],
        }
    )
    print("\n[KPI Sobrestock] Resumen de indicadores:")
    print(resumen.to_string(index=False))

    metricas = {
        "sobre_base_pct": float(promedio_pct_base)
        if not np.isnan(promedio_pct_base)
        else float("nan"),
        "sobre_modelo_pct": float(promedio_pct_modelo)
        if not np.isnan(promedio_pct_modelo)
        else float("nan"),
        "promedio_pred_base": float(promedio_pred_base),
        "promedio_pred_modelo": float(promedio_pred_modelo),
    }

    return {
        "metrics": metricas,
        "figures": figuras,
    }
