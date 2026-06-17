"""Model backends (strategies)."""

from . import (  # noqa: F401  (each import registers a backend; all torch-free)
    chronos2,
    chronos_bolt,
    lightgbm,
    moirai,
    timesfm25,
    ttm,
)
