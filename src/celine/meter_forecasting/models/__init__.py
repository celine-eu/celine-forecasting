"""Model backends (strategies)."""

from . import (
    lightgbm,  # noqa: F401  (import registers the backend)
    ttm,  # noqa: F401  (registers the TTM backend; torch-free import)
)
