"""Modulo de diagnosticos y graficos del pronostico de demanda."""

from __future__ import annotations

import importlib
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def _asegurar_directorio(ruta: Path) -> None:
    """Crea el directorio padre si no existe."""
    ruta.parent.mkdir(parents=True, exist_ok=True)


def compilar_resultados(
    df: pd.DataFrame,
    y_pred_modelo: np.ndarray,
) -> pd.DataFrame:
    """Construye un DataFrame completo con metricas y banderas de agotamiento."""

    if len(df) != len(y_pred_modelo):
        raise ValueError(
            "La longitud de las predicciones no coincide con el DataFrame de entrada."
        )

    columnas_diagnostico = [
        "Date",
        "Store ID",
        "Product ID",
        "Inventory Level",
        "Units Sold",
        "Price",
        "Category",
        "Seasonality",
        "Demand Forecast",
    ]

    faltantes_df = [col for col in columnas_diagnostico if col not in df.columns]
    if faltantes_df:
        raise KeyError(f"Columnas faltantes en los datos originales: {faltantes_df}")

    resultados = df[columnas_diagnostico].copy()
    resultados = resultados.rename(
        columns={
            "Inventory Level": "Nivel Inventario",
            "Units Sold": "Unidades Vendidas",
            "Seasonality": "Temporada",
            "Demand Forecast": "Demanda Historica",
        }
    )

    resultados["Date"] = pd.to_datetime(resultados["Date"], errors="coerce")
    resultados = resultados.dropna(subset=["Date"]).copy()

    resultados["Demanda Modelo"] = np.asarray(y_pred_modelo)[: len(resultados)]

    resultados["Error Modelo"] = resultados["Unidades Vendidas"] - resultados["Demanda Modelo"]
    resultados["Error Absoluto Modelo"] = np.abs(resultados["Error Modelo"])
    resultados["Error Cuadratico Modelo"] = resultados["Error Modelo"] ** 2

    resultados["Error Historico"] = resultados["Unidades Vendidas"] - resultados["Demanda Historica"]
    resultados["Error Absoluto Historico"] = np.abs(resultados["Error Historico"])
    resultados["Error Cuadratico Historico"] = resultados["Error Historico"] ** 2

    resultados["agotamiento_real"] = resultados["Unidades Vendidas"] > resultados["Nivel Inventario"]
    resultados["agotamiento_modelo"] = resultados["Demanda Modelo"] > resultados["Nivel Inventario"]
    resultados["agotamiento_historico"] = resultados["Demanda Historica"] > resultados["Nivel Inventario"]

    columnas_orden = [
        "Date",
        "Store ID",
        "Product ID",
        "Category",
        "Price",
        "Temporada",
        "Nivel Inventario",
        "Unidades Vendidas",
        "Demanda Historica",
        "Demanda Modelo",
        "Error Modelo",
        "Error Absoluto Modelo",
        "Error Cuadratico Modelo",
        "Error Historico",
        "Error Absoluto Historico",
        "Error Cuadratico Historico",
        "agotamiento_real",
        "agotamiento_modelo",
        "agotamiento_historico",
    ]

    resultados = resultados[columnas_orden].sort_values(["Date", "Store ID", "Product ID"]).reset_index(drop=True)

    return resultados


def graficar_indicadores(
    resultados: pd.DataFrame,
    mostrar: bool = True,
) -> dict[str, Path]:
    """Genera comparativas diarias entre ventas reales, pronostico historico y modelo."""

    if resultados.empty:
        print("[Visualizacion] DataFrame de resultados vacio; no se generan indicadores.")
        return {}

    datos = resultados.copy()
    if "Date" not in datos.columns:
        raise KeyError("El DataFrame de resultados no contiene la columna 'Date'.")

    datos["Date"] = pd.to_datetime(datos["Date"], errors="coerce")
    datos = datos.dropna(subset=["Date"]).sort_values("Date")
    if datos.empty:
        print("[Visualizacion] No hay registros con fecha valida para graficar.")
        return {}

    diarios = (
        datos.groupby("Date")[["Unidades Vendidas", "Demanda Historica", "Demanda Modelo"]]
        .sum()
        .sort_index()
    )

    fig_baseline, ax_baseline = plt.subplots(figsize=(12, 5))
    ax_baseline.plot(
        diarios.index,
        diarios["Unidades Vendidas"],
        label="Ventas reales",
        color="#2ca02c",
        linewidth=1.4,
    )
    ax_baseline.plot(
        diarios.index,
        diarios["Demanda Historica"],
        label="Pronostico historico",
        color="#1f77b4",
        linewidth=1.4,
        linestyle="--",
    )
    ax_baseline.set_title("Demanda diaria: Ventas reales vs. pronostico historico")
    ax_baseline.set_xlabel("Fecha")
    ax_baseline.set_ylabel("Unidades")
    ax_baseline.tick_params(axis="x", rotation=20)
    ax_baseline.grid(True, linestyle="--", linewidth=0.5, alpha=0.6)
    ax_baseline.legend(loc="upper left")

    fig_modelo, ax_modelo = plt.subplots(figsize=(12, 5))
    ax_modelo.plot(
        diarios.index,
        diarios["Unidades Vendidas"],
        label="Ventas reales",
        color="#2ca02c",
        linewidth=1.4,
    )
    ax_modelo.plot(
        diarios.index,
        diarios["Demanda Modelo"],
        label="Pronostico modelo XGBoost",
        color="#ff7f0e",
        linewidth=1.4,
        linestyle="--",
    )
    ax_modelo.set_title("Demanda diaria: Ventas reales vs. pronostico del modelo")
    ax_modelo.set_xlabel("Fecha")
    ax_modelo.set_ylabel("Unidades")
    ax_modelo.tick_params(axis="x", rotation=20)
    ax_modelo.grid(True, linestyle="--", linewidth=0.5, alpha=0.6)
    ax_modelo.legend(loc="upper left")

    ruta_baseline = Path("outputs/plots/1_Precision/comparativa_demanda_baseline.png")
    ruta_modelo = Path("outputs/plots/1_Precision/comparativa_demanda_modelo.png")
    for ruta in (ruta_baseline, ruta_modelo):
        _asegurar_directorio(ruta)

    fig_baseline.savefig(ruta_baseline, dpi=200)
    fig_modelo.savefig(ruta_modelo, dpi=200)

    print(
        "[Visualizacion] Figuras guardadas: "
        f"{ruta_baseline.as_posix()}, {ruta_modelo.as_posix()}"
    )

    if mostrar:
        plt.show()

    plt.close(fig_baseline)
    plt.close(fig_modelo)

    return {
        "comparativa_baseline": ruta_baseline,
        "comparativa_modelo": ruta_modelo,
    }


def resumen_por_temporada(resultados: pd.DataFrame) -> pd.DataFrame:
    """Calcula indicadores agregados por temporada."""
    resumen = resultados.groupby("Temporada").agg(
        unidades_reales=("Unidades Vendidas", "sum"),
        unidades_modelo=("Demanda Modelo", "sum"),
        unidades_historicas=("Demanda Historica", "sum"),
        mae_modelo=("Error Absoluto Modelo", "mean"),
        mse_modelo=("Error Cuadratico Modelo", "mean"),
        mae_historico=("Error Absoluto Historico", "mean"),
        mse_historico=("Error Cuadratico Historico", "mean"),
        tasa_agotamiento_real=("agotamiento_real", "mean"),
        tasa_agotamiento_modelo=("agotamiento_modelo", "mean"),
        tasa_agotamiento_historico=("agotamiento_historico", "mean"),
    )
    resumen["rmse_modelo"] = np.sqrt(resumen["mse_modelo"])
    resumen["rmse_historico"] = np.sqrt(resumen["mse_historico"])
    resumen = resumen.reset_index().sort_values("Temporada")
    columnas_redondeo = [
        "unidades_reales",
        "unidades_modelo",
        "unidades_historicas",
        "mae_modelo",
        "mse_modelo",
        "rmse_modelo",
        "mae_historico",
        "mse_historico",
        "rmse_historico",
        "tasa_agotamiento_real",
        "tasa_agotamiento_modelo",
        "tasa_agotamiento_historico",
    ]
    resumen[columnas_redondeo] = resumen[columnas_redondeo].round(2)

    return resumen


def resumen_global_modelos(resultados: pd.DataFrame) -> pd.DataFrame:
    """Compara errores globales entre el pronostico historico y el modelo."""
    mae_modelo = resultados["Error Absoluto Modelo"].mean()
    rmse_modelo = np.sqrt(resultados["Error Cuadratico Modelo"].mean())
    mae_historico = resultados["Error Absoluto Historico"].mean()
    rmse_historico = np.sqrt(resultados["Error Cuadratico Historico"].mean())
    tasa_modelo = resultados["agotamiento_modelo"].mean()
    tasa_historico = resultados["agotamiento_historico"].mean()

    comparacion = pd.DataFrame(
        {
            "Modelo": ["Pronostico Historico", "Modelo XGBoost"],
            "MAE": [mae_historico, mae_modelo],
            "RMSE": [rmse_historico, rmse_modelo],
            "Tasa Agotamiento": [tasa_historico, tasa_modelo],
        }
    )
    comparacion[["MAE", "RMSE", "Tasa Agotamiento"]] = comparacion[
        ["MAE", "RMSE", "Tasa Agotamiento"]
    ].round(2)

    return comparacion

def mostrar_tablas(
    resultados: pd.DataFrame,
    resumen_temporada: pd.DataFrame,
    comparacion_global: pd.DataFrame | None = None,
    ruta_detalle: Path | None = Path("outputs/data/pronostico_detalle.csv"),
    ruta_temporada: Path | None = Path("outputs/data/pronostico_temporada.csv"),
    ruta_pdf: Path | None = None,
) -> None:
    """Presenta tablas en consola y guarda los CSV de resultados."""
    tabulate_fn = None
    try:
        modulo_tabulate = importlib.import_module("tabulate")
    except ModuleNotFoundError:
        usar_tabulate = False
    else:
        tabulate_fn = getattr(modulo_tabulate, "tabulate", None)
        usar_tabulate = callable(tabulate_fn)

    print("\n[Tablas] Vista preliminar del detalle (primeras 10 filas):")
    vista_detalle = resultados.head(10)
    if usar_tabulate and tabulate_fn:
        print(tabulate_fn(vista_detalle, headers="keys", tablefmt="psql", showindex=False))
    else:
        print(vista_detalle.to_string(index=False))

    print("\n[Tablas] Resumen por temporada:")
    if usar_tabulate and tabulate_fn:
        print(tabulate_fn(resumen_temporada, headers="keys", tablefmt="psql", showindex=False))
    else:
        print(resumen_temporada.to_string(index=False))

    if comparacion_global is not None:
        print("\n[Tablas] Comparacion global de errores:")
        if usar_tabulate and tabulate_fn:
            print(tabulate_fn(comparacion_global, headers="keys", tablefmt="psql", showindex=False))
        else:
            print(comparacion_global.to_string(index=False))

    if ruta_detalle:
        ruta_detalle = Path(ruta_detalle)
        _asegurar_directorio(ruta_detalle)
        resultados.to_csv(ruta_detalle, index=False)
        print(f"[Tablas] Archivo de detalle guardado en {ruta_detalle.as_posix()}")

    if ruta_temporada:
        ruta_temporada = Path(ruta_temporada)
        _asegurar_directorio(ruta_temporada)
        resumen_temporada.to_csv(ruta_temporada, index=False)
        print(f"[Tablas] Archivo por temporada guardado en {ruta_temporada.as_posix()}")

    if ruta_pdf:
        print(
            "[Tablas] La generacion de PDF fue eliminada; el parametro ruta_pdf se ignora."
        )


def graficar_curva_de_aprendizaje(
    modelo_xgb,
    ruta_salida: Path | None = Path("outputs/plots/1_Precision/modelo_curva_aprendizaje.png"),
    mostrar: bool = True,
) -> Path | None:
    """Grafica el RMSE por iteracion durante el entrenamiento de XGBoost."""

    resultados = modelo_xgb.evals_result()
    if not resultados:
        print("[Visualizacion] No se encontraron metricas de entrenamiento para graficar.")
        return

    rmse_entrenamiento = resultados.get("validation_0", {}).get("rmse")
    rmse_validacion = resultados.get("validation_1", {}).get("rmse")

    if rmse_entrenamiento is None and rmse_validacion is None:
        print("[Visualizacion] El modelo no reporta valores de RMSE en evals_result().")
        return

    figura, eje = plt.subplots(figsize=(12, 5))

    if rmse_entrenamiento is not None:
        eje.plot(rmse_entrenamiento, label="RMSE entrenamiento", color="#1f77b4")
    if rmse_validacion is not None:
        eje.plot(rmse_validacion, label="RMSE validacion", color="#ff7f0e")

    eje.set_title("Curva de aprendizaje del modelo XGBoost")
    eje.set_xlabel("Iteraciones (arboles)")
    eje.set_ylabel("RMSE")
    eje.grid(True, linestyle="--", linewidth=0.5, alpha=0.7)
    eje.legend()

    mejor_iteracion = getattr(modelo_xgb, "best_iteration", None)
    mejor_rmse = getattr(modelo_xgb, "best_score", None)
    if mejor_iteracion is not None and mejor_rmse is not None:
        eje.axvline(x=mejor_iteracion, color="red", linestyle="--")
        eje.text(
            mejor_iteracion,
            eje.get_ylim()[0],
            f"  Mejor iteracion: {mejor_iteracion}\n  RMSE: {mejor_rmse:.2f}",
            color="red",
            va="bottom",
        )

    ruta_guardada: Path | None = None
    if ruta_salida:
        ruta_guardada = Path(ruta_salida)
        _asegurar_directorio(ruta_guardada)
        figura.savefig(ruta_guardada, dpi=200)
        print(
            f"[Visualizacion] Curva de aprendizaje guardada en {ruta_guardada.as_posix()}"
        )

    if mostrar:
        plt.show()

    plt.close(figura)

    return ruta_guardada


def graficar_dii_comparativo(
    resultados: pd.DataFrame,
    mostrar: bool = True,
) -> Path | None:
    """Grafica la comparativa de DII (Base vs Modelo) con media movil de 30 dias y guarda la figura."""
    if resultados.empty:
        print("[Visualizacion] DataFrame de resultados vacio; no se puede graficar DII comparativo.")
        return

    datos = resultados.copy()

    # Calculo de DII base y modelo
    datos["DII_Base"] = (datos["Nivel Inventario"] / datos["Demanda Historica"]).replace([np.inf, -np.inf], np.nan)
    datos["DII_Modelo"] = (datos["Nivel Inventario"] / datos["Demanda Modelo"]).replace([np.inf, -np.inf], np.nan)

    # Agregacion diaria (media de DII por fecha)
    dii_diario = datos.groupby("Date").agg(
        DII_Base=("DII_Base", "mean"),
        DII_Modelo=("DII_Modelo", "mean"),
    ).sort_index()

    # Medias moviles 30 dias
    dii_base_30d = dii_diario["DII_Base"].rolling(30, min_periods=1).mean()
    dii_modelo_30d = dii_diario["DII_Modelo"].rolling(30, min_periods=1).mean()

    # Figura comparativa
    fig_dii_comp, ax = plt.subplots(figsize=(12, 5))
    ax.plot(dii_base_30d.index, dii_base_30d.values, label="DII Línea Base", color="#2ca02c", linewidth=1.5)
    ax.plot(dii_modelo_30d.index, dii_modelo_30d.values, label="DII Modelo XGBoost", color="#ff7f0e", linewidth=1.5, linestyle="--")
    ax.set_title("Comparativa de Tendencia de Días de Inventario (DII Móvil 30 Días)")
    ax.set_xlabel("Fecha")
    ax.set_ylabel("DII (días)")
    ax.grid(True, linestyle="--", linewidth=0.5, alpha=0.7)
    ax.legend(loc="upper left")

    # Guardado
    ruta_png = Path("outputs/plots/2_DII/comparativa_tendencia_dii.png")
    _asegurar_directorio(ruta_png)
    fig_dii_comp.savefig(ruta_png, dpi=200)
    print(f"[Visualizacion] DII comparativo guardado en {ruta_png.as_posix()}")

    if mostrar:
        plt.show()

    plt.close(fig_dii_comp)

    return ruta_png


def graficar_scatter_modelo(
    y_test: pd.Series,
    y_pred: np.ndarray,
    mostrar: bool = True,
) -> Path | None:
    """Grafica un scatter Real vs Prediccion del modelo y guarda la figura."""
    if y_test is None or y_pred is None or len(y_test) == 0 or len(y_pred) == 0:
        print("[Visualizacion] Entradas vacias; no se puede graficar scatter.")
        return

    x = pd.Series(y_test).to_numpy(dtype=float)
    y = np.asarray(y_pred, dtype=float)

    # Filtrar valores no finitos
    mask = np.isfinite(x) & np.isfinite(y)
    x = x[mask]
    y = y[mask]
    if x.size == 0 or y.size == 0:
        print("[Visualizacion] No hay datos finitos para graficar scatter.")
        return

    figura, eje = plt.subplots(figsize=(6, 6))
    eje.scatter(x, y, alpha=0.5, edgecolors="none")

    lim_min = float(min(np.min(x), np.min(y), 0))
    lim_max = float(max(np.max(x), np.max(y)))
    eje.plot([lim_min, lim_max], [lim_min, lim_max], color="red", linestyle="--", linewidth=1)

    eje.set_xlim(lim_min, lim_max)
    eje.set_ylim(lim_min, lim_max)
    eje.set_xlabel("Ventas Reales")
    eje.set_ylabel("Predicción Modelo XGBoost")
    eje.set_title("Comparativa de Dispersión: Ventas Reales vs. Predicción XGBoost")
    eje.grid(True, linestyle=":", linewidth=0.5, alpha=0.6)

    ruta_png = Path("outputs/plots/1_Precision/modelo_scatter_real_vs_pred.png")
    _asegurar_directorio(ruta_png)
    figura.savefig(ruta_png, dpi=200)
    print(f"[Visualizacion] Scatter Real vs Pred guardado en {ruta_png.as_posix()}")

    if mostrar:
        plt.show()

    plt.close(figura)

    return ruta_png
