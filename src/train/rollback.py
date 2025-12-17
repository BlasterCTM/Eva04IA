"""Herramientas simples para listar/rollback del modelo guardado.

Uso:
    python -m src.train.rollback --list
    python -m src.train.rollback --previous
    python -m src.train.rollback --to v20250101120000

El script actualiza `models/metadata.json` y sobreescribe
`models/modelo_demanda_v1.joblib` con la versión seleccionada (compatibilidad).
"""

from __future__ import annotations

from pathlib import Path
import argparse
import json
import shutil


def _load_registry(ruta_meta: Path) -> dict:
    if not ruta_meta.exists():
        return {"versions": [], "current": None}
    with open(ruta_meta, "r", encoding="utf-8") as f:
        return json.load(f)


def _save_registry(ruta_meta: Path, registry: dict) -> None:
    with open(ruta_meta, "w", encoding="utf-8") as f:
        json.dump(registry, f, indent=2, ensure_ascii=False)


def list_versions(ruta_meta: Path) -> None:
    registry = _load_registry(ruta_meta)
    versions = registry.get("versions", [])
    current = registry.get("current")
    if not versions:
        print("No hay versiones registradas.")
        return
    for v in versions:
        mark = "*" if v.get("version") == current else " "
        print(f"{mark} {v.get('version')} - {v.get('saved_at')} - {v.get('model_path')}")


def set_current(ruta_meta: Path, modelos_dir: Path, target_version: str) -> bool:
    registry = _load_registry(ruta_meta)
    versions = registry.get("versions", [])
    match = next((v for v in versions if v.get("version") == target_version), None)
    if not match:
        print(f"Versión no encontrada: {target_version}")
        return False
    src_path = Path(match.get("model_path"))
    if not src_path.exists():
        print(f"Archivo de modelo no encontrado: {src_path}")
        return False
    dest = modelos_dir / "modelo_demanda_v1.joblib"
    shutil.copy2(src_path, dest)
    registry["current"] = target_version
    _save_registry(ruta_meta, registry)
    print(f"Rollback completado. Current set to {target_version}")
    return True


def rollback_previous(ruta_meta: Path, modelos_dir: Path) -> None:
    registry = _load_registry(ruta_meta)
    versions = registry.get("versions", [])
    if not versions or registry.get("current") is None:
        print("No hay versiones para hacer rollback.")
        return
    current = registry["current"]
    idx = next((i for i, v in enumerate(versions) if v.get("version") == current), None)
    if idx is None:
        print("La versión actual no está en el registro.")
        return
    if idx == 0:
        print("No hay versión anterior para hacer rollback.")
        return
    target = versions[idx - 1]["version"]
    set_current(ruta_meta, modelos_dir, target)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--list", action="store_true", help="Listar versiones registradas")
    parser.add_argument("--previous", action="store_true", help="Hacer rollback a la versión anterior")
    parser.add_argument("--to", type=str, help="Hacer rollback a la versión indicada (ej: v20250101120000)")
    args = parser.parse_args()

    modelos_dir = Path("models")
    ruta_meta = modelos_dir / "metadata.json"

    if args.list:
        list_versions(ruta_meta)
        return
    if args.previous:
        rollback_previous(ruta_meta, modelos_dir)
        return
    if args.to:
        set_current(ruta_meta, modelos_dir, args.to)
        return

    parser.print_help()
