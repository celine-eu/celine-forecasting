# TODO — neural backend verification

Status legend: ✅ done · 🔜 next · ⏳ blocked/needs GPU · ❓ open question

The torch seams for all neural backends are implemented but **can only run on the
GPU box** (`sa@192.168.1.144`, RTX 3080) — the dev laptop is torch-free (Python
3.13, no CUDA). The dev-env checks below (lint/types/imports/tests) are green; the
GPU smoke + serving checks are the outstanding work.

## 0. Environment recap (GPU VM)

- Use the **RTX 3080**, never the 1080 Ti (Pascal sm_61 is unsupported by modern
  torch). In `~/.bashrc`: `CUDA_DEVICE_ORDER=PCI_BUS_ID` + `CUDA_VISIBLE_DEVICES=1`.
- One venv per backend (deps conflict). Pattern:
  ```bash
  cd ~/celine-forecasting
  uv venv --python 3.12 .venv-<backend> && source .venv-<backend>/bin/activate
  uv pip install torch --index-url https://download.pytorch.org/whl/cu124
  uv pip install -e . -r src/celine/meter_forecasting/models/<backend>/requirements.txt
  ```
- Sync code from the laptop after each edit: plain `rsync` (NO `--delete` — it would
  wipe the `.venv-*` venvs):
  ```bash
  rsync -avz --filter=':- .gitignore' --exclude='.git' ./ sa@192.168.1.144:~/celine-forecasting/
  ```

## 1. Dev-env checks (laptop) — ✅ done

- ✅ torch-free import of all backend modules
- ✅ `ruff check src/celine/meter_forecasting/models/`
- ✅ `mypy` clean (20 source files)
- ✅ full pytest suite (torch backends skip; extras absent)
- ❓ `test_serving_tracking.py::test_pipeline_logs_per_device_child_runs` fails only
  in the FULL suite, passes in isolation (with and without these changes). Backend
  code is inert here (extras not installed), so suspected pre-existing flake —
  confirming via a clean-tree full run. If confirmed: file separately, not a blocker.

## 2. GPU smoke tests — 🔜 (per backend)

For each backend: set up the venv (§0), then:
```bash
CUDA_VISIBLE_DEVICES=1 python -m celine.meter_forecasting.models.<backend>.smoke_<backend>
```
Expect a printed horizon forecast + `<BACKEND> smoke OK`. Watch VRAM with
`watch -n1 nvidia-smi` (process should sit on GPU 1).

- ✅ **ttm** — green on the 3080 (fit + fine-tune + predict).
- ✅ **chronos_bolt** — `BaseChronosPipeline`, univariate, zero-shot. Green (first try).
- ✅ **chronos2** — `Chronos2Pipeline`, covariates via dict inputs. Green (first try).
  Shares the chronos_bolt venv (same `chronos-forecasting==2.2.2`).
- ✅ **timesfm25** — lazy decode-compile; univariate; median = slot 5. Green (first try).
- ✅ **moirai** — GluonTS predictor + `feat_dynamic_real`. Green (first try).

All five verified zero-shot on the RTX 3080 with no seam fixes needed (2026-06-30).
Next gate is §3 (serving roundtrip).

**Likely first-run breakages to watch for** (the seams are faithful but the libs'
exact APIs couldn't be exercised locally — fixes are one-liners at the failure point):
- chronos*: `pipeline.model.save_pretrained` attribute name; `predict_quantiles`
  return shape (`qt_list[0].squeeze(0)`).
- timesfm: `ForecastConfig` field names; `TimesFM_2p5_200M_torch.from_pretrained`;
  output tuple `(_, quantiles)` shape `(1, H, 10)`.
- moirai: `MoiraiForecast` constructor kwargs; `forecast.quantile(0.5)`; predictor
  `.prediction_length` (used by persistence to rebuild on load).

## 3. Persistence / MLflow serving roundtrip — ⏳ (after §2 each green)

The `[[mlflow-all-backends]]` requirement: tracking + serving must work for every
backend, not just LightGBM. Per backend, on the GPU box:
- Log a fitted model via the pipeline, then load + predict from the MLflow artifact
  (mirror `tests/test_serving_all_backends.py`, which skips here because extras are
  absent). Verify `NeuralFitted.__getstate__/__setstate__` round-trips:
  - chronos*: HF model saved under `model/`, reloaded via `from_pretrained`.
  - timesfm/moirai: weights NOT saved (fixed checkpoint) — reloaded from the HF id
    using geometry in `meta.json`. ❓ Confirm offline reload works (HF cache present).
- ❓ Decide whether to run these serving tests in CI on the GPU box (currently they
  skip in the torch-free env).

## 4. Real-data backtest sanity — ⏳ (after §3)

- Run a short `meter-forecast` backtest per backend on a real device frame and eye
  the MAE vs the LightGBM baseline + naive-yesterday skill (sanity, not a benchmark).
- ❓ Confirm the `log1p`+standardize target transform composes sensibly with each
  model's own internal scaling (esp. chronos/moirai, which also normalize inputs).

## 5. Optional / later

- ⏳ Wire real fine-tune for **chronos2** (clean `Chronos2Pipeline.fit` API) and
  **timesfm25** (bespoke loop) if fine-tuned variants are wanted — currently zero-shot.
- ⏳ Decide GPU-CI strategy for the neural smoke + serving tests.
- ⏳ `pyproject.toml`/`uv.lock` torch pin note: the reference forced torch past
  `uni2ts`'s `torch<2.5` pin for CUDA wheels — keep an eye on moirai install.
