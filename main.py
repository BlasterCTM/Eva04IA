"""Script de arranque del pipeline de pronostico de demanda."""

from __future__ import annotations

from src.flujo_principal import ejecutar_pipeline


def main() -> None:
    """Invoca el pipeline principal con rutas por defecto."""
    ejecutar_pipeline()


if __name__ == "__main__":
    main()
