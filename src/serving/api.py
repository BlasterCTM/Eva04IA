"""Wrapper que exporta la versión minimal de los endpoints.

Este módulo expone `app` importado desde `api_endpoints_only` para que
`uvicorn src.serving.api:app` arranque la versión minimal (solo endpoints JSON).
"""

from __future__ import annotations

from .api_endpoints_only import app  # type: ignore


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("src.serving.api:app", host="0.0.0.0", port=8000, reload=True)
