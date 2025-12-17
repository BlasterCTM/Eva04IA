"""Punto de entrada del flujo de pronostico de demanda."""

from __future__ import annotations

from pathlib import Path


from .dashboard_logica import (
    generar_kpis_resumen,
    generar_lista_riesgo_quiebre,
    generar_lista_sobrestock,
    generar_grafico_ventas_categoria,
    generar_predicciones_por_producto,
)
from .ingenieria_caracteristicas import agregar_caracteristicas
from .kpi_baseline import generar_kpis_linea_base
from .kpi_sobrestock import analizar_sobrestock_y_precision
from .modelo_demanda import evaluar_modelo, entrenar_modelo, preparar_conjuntos
from .procesamiento_datos import cargar_y_explorar, limpiar_datos
from .visualizacion import (
    compilar_resultados,
    graficar_curva_de_aprendizaje,
    graficar_dii_comparativo,
    graficar_indicadores,
    graficar_scatter_modelo,
    mostrar_tablas,
    resumen_global_modelos,
    resumen_por_temporada,
)


def ejecutar_pipeline(
    data_path: Path | None = None,
    mostrar_graficos: bool = False,
    abrir_interfaz: bool = True,
) -> tuple[object, dict]:
    """Ejecuta el flujo completo de generacion de KPIs y pronostico."""
    data_path = data_path or Path("data/retail_store_inventory.csv")

    # Importar solo si se requiere interfaz gráfica
    if mostrar_graficos or abrir_interfaz:
        from .interfaz_graficos import mostrar_interfaz_graficos

    raw_df = cargar_y_explorar(data_path)
    clean_df = limpiar_datos(raw_df)

    print("\n[KPI linea base] Generando analisis de linea base...")
    baseline_out = generar_kpis_linea_base(clean_df, mostrar=mostrar_graficos)

    feature_df = agregar_caracteristicas(clean_df)

    (
        X_train,
        X_test,
        y_train,
        y_test,
        preprocessor,
        _,
        _,
        columnas_modelo,
    ) = preparar_conjuntos(feature_df)

    workflow, X_test_processed, xgb_model = entrenar_modelo(
        X_train, X_test, y_train, y_test, preprocessor
    )

    mae, rmse, y_pred = evaluar_modelo(xgb_model, X_test_processed, y_test)
    curva_aprendizaje = graficar_curva_de_aprendizaje(xgb_model, mostrar=mostrar_graficos)
    predicciones_completas = workflow.predict(feature_df[columnas_modelo])
    results_df = compilar_resultados(feature_df, predicciones_completas)
    print("\n[Dashboard] Generando metricas y reportes del dashboard...")

    kpis = generar_kpis_resumen(results_df)
    print(f"[Dashboard] KPIs Resumen (para el {kpis['fecha_hoy']}):")
    print(f"  - Productos en Riesgo: {kpis['productos_en_riesgo']}")
    print(f"  - Capital Inmovilizado: ${kpis['capital_inmovilizado']:,.2f}")
    print(f"  - Ventas Pronosticadas: {kpis['ventas_pronosticadas']:.0f} unidades")

    tabla_riesgo = generar_lista_riesgo_quiebre(results_df)
    print(f"[Dashboard] Tabla 'Riesgo de Quiebre' generada con {len(tabla_riesgo)} productos.")

    tabla_sobrestock = generar_lista_sobrestock(results_df)
    print(f"[Dashboard] Tabla 'Sobrestock' generada con {len(tabla_sobrestock)} productos.")
    tabla_predicciones = generar_predicciones_por_producto(results_df)
    print(
        f"[Dashboard] Tabla de predicciones diarias generada con {len(tabla_predicciones)} registros."
    )

    ruta_grafico_dashboard = Path("outputs/plots/dashboard/ventas_por_categoria.png")
    generar_grafico_ventas_categoria(results_df, ruta_salida=ruta_grafico_dashboard.as_posix())
    print("\n[KPI DII] Generando analisis comparativo de DII...")
    grafico_dii = graficar_dii_comparativo(results_df, mostrar=mostrar_graficos)
    print("\n[KPI Sobrestock y precision] Generando analisis comparativo de Sobrestock y precision...")
    sobrestock_out = analizar_sobrestock_y_precision(results_df, mostrar=mostrar_graficos)
    figuras_indicadores = graficar_indicadores(results_df, mostrar=mostrar_graficos)
    season_summary = resumen_por_temporada(results_df)
    global_comparison = resumen_global_modelos(results_df)
    mostrar_tablas(results_df, season_summary, global_comparison)
    print("\n[Visualizacion] Generando diagrama de dispersion del modelo...")
    grafico_scatter = graficar_scatter_modelo(y_test, y_pred, mostrar=mostrar_graficos)

    figuras_sobrestock = (
        sobrestock_out.get("figures") if isinstance(sobrestock_out, dict) else {}
    )

    figuras_baseline = (
        baseline_out.get("figures") if isinstance(baseline_out, dict) else {}
    )

    secciones_dashboard: dict[str, list[Path]] = {}

    def _agregar_figura(seccion: str, ruta: Path | None) -> None:
        if ruta is None:
            return
        ruta_path = Path(ruta)
        if not ruta_path.exists():
            return
        if ruta_path.suffix.lower() not in {".png", ".jpg", ".jpeg"}:
            return
        secciones_dashboard.setdefault(seccion, []).append(ruta_path)

    if figuras_baseline:
        _agregar_figura("Linea base", figuras_baseline.get("baseline_mape"))
        _agregar_figura("Linea base", figuras_baseline.get("baseline_scatter"))
        _agregar_figura("Linea base", figuras_baseline.get("baseline_dii"))

    _agregar_figura(
        "Demanda y modelo", figuras_indicadores.get("comparativa_baseline") if figuras_indicadores else None
    )
    _agregar_figura(
        "Demanda y modelo", figuras_indicadores.get("comparativa_modelo") if figuras_indicadores else None
    )
    _agregar_figura("Demanda y modelo", grafico_scatter)
    _agregar_figura("Curva de aprendizaje", curva_aprendizaje)
    _agregar_figura("Inventario", grafico_dii)
    _agregar_figura("Dashboard", ruta_grafico_dashboard)

    if figuras_sobrestock:
        _agregar_figura("Sobrestock", figuras_sobrestock.get("sobrestock_baseline"))
        _agregar_figura("Sobrestock", figuras_sobrestock.get("sobrestock_modelo"))
        _agregar_figura("Sobrestock", figuras_sobrestock.get("sobrestock_comparativo"))
        _agregar_figura("Sobrestock", figuras_sobrestock.get("oos_comparativo"))
        _agregar_figura("Sobrestock", figuras_sobrestock.get("mae_comparativo"))
        _agregar_figura("Sobrestock", figuras_sobrestock.get("rmse_comparativo"))

    print("\n[Resumen]")
    print(
        f"Error absoluto medio: {mae:.2f} unidades. "
        f"Raiz del error cuadratico medio: {rmse:.2f} unidades."
    )

    print("\n[Recomendacion operativa]")
    print(
        "Ejecute el flujo entrenado a diario para anticipar la demanda, compararla con "
        "el inventario disponible y activar acciones proactivas de reposicion."
    )

    print("\n[Modelo vs pronostico historico]")
    print(global_comparison.to_string(index=False))

    print("\n[Pronostico por temporada]")
    print(season_summary.to_string(index=False))

    tablas_dashboard = {
        "Riesgo de quiebre": tabla_riesgo,
        "Sobrestock": tabla_sobrestock,
        "Predicciones por producto": tabla_predicciones,
    }

    if abrir_interfaz and (
        secciones_dashboard
        or any(not getattr(tabla, "empty", True) for tabla in tablas_dashboard.values())
        or bool(kpis)
    ):
        mostrar_interfaz_graficos(
            secciones_dashboard,
            kpis=kpis,
            tablas=tablas_dashboard,
        )

    metrics = {"mae": float(mae), "rmse": float(rmse)}

    return workflow, metrics
