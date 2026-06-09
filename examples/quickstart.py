"""End-to-end quickstart on the synthetic fixture.

Generates the sample data if needed, then runs the full pipeline and prints a
short summary. This is the fastest way to confirm a working install.

Run:
    python examples/quickstart.py
"""

from __future__ import annotations

import logging
from pathlib import Path

from celine.meter_forecasting import load_config, load_meters, load_weather, train_pipeline

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")

HERE = Path(__file__).resolve().parent
SAMPLE = HERE / "sample_data"


def main() -> None:
    """Run the quickstart pipeline."""
    if not (SAMPLE / "meters_sample.csv").exists():
        from generate_sample_data import generate

        generate()

    config = load_config()
    meters = load_meters(SAMPLE / "meters_sample.csv")
    weather = load_weather(SAMPLE / "weather_sample.csv")

    result = train_pipeline(meters, config, df_weather=weather, do_cv=True, output_dir=HERE / "out")

    print("\n=== Eligibility ===")
    print(result.eligibility.to_string(index=False))
    print("\n=== Cross-validation skill (vs seasonal naive) ===")
    print(result.cv_results.round(3).to_string(index=False))
    print(f"\nTrained {len(result.trained_models)} device(s). Artifacts in {HERE / 'out'}.")


if __name__ == "__main__":
    main()
