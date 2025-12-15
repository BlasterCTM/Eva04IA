#!/usr/bin/env python3
"""Run uvicorn programmatically to avoid shell wrapper issues (e.g., missing sed)."""
from importlib import import_module
import uvicorn


def main(host: str = "127.0.0.1", port: int = 8001) -> None:
    mod = import_module("src.serving.api_endpoints_only")
    app = getattr(mod, "app")
    uvicorn.run(app, host=host, port=port, log_level="info")


if __name__ == "__main__":
    main()
