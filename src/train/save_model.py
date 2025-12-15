"""Script para entrenar (usar `flujo_principal`) y guardar el pipeline entrenado.

Genera `models/modelo_demanda_v1.joblib`.
"""

from __future__ import annotations

from pathlib import Path
import json
import joblib
from datetime import datetime

from src.flujo_principal import ejecutar_pipeline


def main() -> None:
    data_path = Path("data/retail_store_inventory.csv")
    print("[Train] Ejecutando pipeline de entrenamiento (esto puede tardar)...")
    result = ejecutar_pipeline(data_path=data_path, mostrar_graficos=False, abrir_interfaz=False)

    if result is None:
        print("[Train] No se obtuvo pipeline entrenado. Verifique errores en el pipeline.")
        return

    # ejecutar_pipeline ahora devuelve (pipeline, metrics)
    try:
        pipeline, metrics = result
    except Exception:
        pipeline = result
        metrics = {}

    modelos_dir = Path("models")
    modelos_dir.mkdir(parents=True, exist_ok=True)
    ruta_modelo = modelos_dir / "modelo_demanda_v1.joblib"
    joblib.dump(pipeline, ruta_modelo)
    print(f"[Train] Modelo guardado en {ruta_modelo.as_posix()}")

    # Guardar metadata del modelo
    metadata = {
        "version": "v1.0",
        "saved_at": datetime.utcnow().isoformat() + "Z",
        "metrics": metrics,
        "model_path": ruta_modelo.as_posix(),
    }
    ruta_meta = modelos_dir / "metadata.json"
    with open(ruta_meta, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)
    print(f"[Train] Metadata guardada en {ruta_meta.as_posix()}")


if __name__ == "__main__":
    main()
