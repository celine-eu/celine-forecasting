"""Write ``golden.json`` for a Chronos-2 model dir, in the TRAINING (reference) env.

The server replays ``golden.json`` at startup and refuses to serve when its
forecasts drift beyond tolerance (see ``celine.forecasting.serve.golden``). The
golden forecasts must therefore come from the environment the model was trained
and validated in (same torch / transformers / chronos-forecasting versions as
training), never from the serving image.

The request goes through the server's own code (``ChronosServingModel.forecast``:
``build_inputs`` + ``predict_quantiles`` + clip/sort). When ``celine.forecasting``
is not installed (e.g. a training venv without LightGBM), only the torch-free
``celine.forecasting.serve`` subpackage is imported from ``--src`` (default: the
``src`` folder of the checkout holding this script), without running the
package ``__init__``.

Usage, inside the training environment, after notebook 07 produced
``runs/nb07/<RUN>/model``::

    python deploy/gx10/make_golden.py \\
        --model-dir runs/nb07/<RUN>/model --request request.json

``request.json`` is a real ``POST /forecast`` body (2-4 series, one PV export),
either the body itself or ``{"payload": body}``. The resulting ``golden.json`` is
written into the model dir and must be shipped with the model to the server.
"""

from __future__ import annotations

import argparse
import importlib
import json
import logging
import sys
import types
from pathlib import Path
from typing import Any

logger = logging.getLogger("make_golden")

DEFAULT_SRC = Path(__file__).resolve().parents[2] / "src"


def import_serve(src: Path | None) -> tuple[types.ModuleType, types.ModuleType]:
    """Import ``celine.forecasting.serve.{model,golden}``.

    Args:
        src: A ``src`` folder to import the serve subpackage from, bypassing the
            package ``__init__`` (and its LightGBM imports); None uses the
            installed package.

    Returns:
        ``(model_module, golden_module)``.
    """
    if src is not None:
        for name in ("celine", "celine.forecasting"):
            stub = types.ModuleType(name)
            stub.__path__ = [str(src.joinpath(*name.split(".")))]
            sys.modules[name] = stub
    return (
        importlib.import_module("celine.forecasting.serve.model"),
        importlib.import_module("celine.forecasting.serve.golden"),
    )


def read_request(path: Path, key: str | None) -> dict[str, Any]:
    """Read a ``POST /forecast`` body, optionally nested under ``key``."""
    data = json.loads(path.read_text())
    if key and key in data:
        return data[key]
    return data


def main(argv: list[str] | None = None) -> int:
    """Generate ``golden.json`` (see the module docstring). Returns the exit code."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--request", type=Path, required=True, help="POST /forecast body JSON")
    parser.add_argument("--request-key", default="payload", help="key holding the body, if any")
    parser.add_argument("--out", type=Path, help="default: <model-dir>/golden.json")
    parser.add_argument("--src", type=Path, default=DEFAULT_SRC if DEFAULT_SRC.is_dir() else None)
    parser.add_argument("--installed", action="store_true", help="use installed celine package")
    parser.add_argument("--device", default="cuda", choices=["auto", "cuda", "cpu"])
    parser.add_argument("--dtype", default="float32", choices=["float32", "bfloat16", "float16"])
    parser.add_argument("--atol", type=float, default=0.05)
    parser.add_argument("--rtol", type=float, default=0.02)
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    serve_model, serve_golden = import_serve(None if args.installed else args.src)
    pipeline, device = serve_model._load_pipeline(args.model_dir, args.dtype, args.device)
    # Pipeline injected: no golden self-check of a golden.json being replaced.
    model = serve_model.ChronosServingModel(args.model_dir, dtype=args.dtype, pipeline=pipeline)
    golden = serve_golden.make_golden(
        model,
        read_request(args.request, args.request_key),
        atol=args.atol,
        rtol=args.rtol,
        device=device,
    )
    out = args.out or args.model_dir / serve_golden.GOLDEN_FILE
    out.write_text(json.dumps(golden))
    logger.info(
        "wrote %s: model_version=%s series=%s env=%s",
        out,
        golden["model_version"],
        sorted(golden["response_quantiles"]),
        golden["reference_env"],
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
