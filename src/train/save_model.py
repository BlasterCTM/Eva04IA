"""Script para entrenar (usar `flujo_principal`) y guardar el pipeline entrenado.

Genera un archivo versionado en `models/` y mantiene `models/modelo_demanda_v1.joblib`
como la versión activa (compatibilidad hacia atrás). También actualiza
`models/metadata.json` con un historial de versiones para permitir rollback.
"""

from __future__ import annotations

from pathlib import Path
import json
import joblib
from datetime import datetime

from src.flujo_principal import ejecutar_pipeline


def _load_registry(ruta_meta: Path) -> dict:
    if not ruta_meta.exists():
        return {"versions": [], "current": None}
    try:
        with open(ruta_meta, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return {"versions": [], "current": None}


def _save_registry(ruta_meta: Path, registry: dict) -> None:
    with open(ruta_meta, "w", encoding="utf-8") as f:
        json.dump(registry, f, indent=2, ensure_ascii=False)


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

    # Version basado en timestamp para evitar colisiones y facilitar tracking
    version = datetime.utcnow().strftime("v%Y%m%d%H%M%S")
    filename_versioned = f"modelo_demanda_{version}.joblib"
    ruta_versioned = modelos_dir / filename_versioned
    joblib.dump(pipeline, ruta_versioned)
    print(f"[Train] Modelo versionado guardado en {ruta_versioned.as_posix()}")

    # Mantener un archivo 'activo' compatible con la version anterior
    ruta_compatible = modelos_dir / "modelo_demanda_v1.joblib"
    joblib.dump(pipeline, ruta_compatible)
    print(f"[Train] Modelo actualizado (compatible) en {ruta_compatible.as_posix()}")

    # Registrar metadata y versiones
    ruta_meta = modelos_dir / "metadata.json"
    registry = _load_registry(ruta_meta)
    entry = {
        "version": version,
        "saved_at": datetime.utcnow().isoformat() + "Z",
        "metrics": metrics,
        "model_path": ruta_versioned.as_posix(),
    }
    registry.setdefault("versions", []).append(entry)
    registry["current"] = version
    _save_registry(ruta_meta, registry)
    print(f"[Train] Registry actualizado en {ruta_meta.as_posix()} (current={version})")


if __name__ == "__main__":
    main()
