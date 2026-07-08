# Wiki Index

Catalog of all wiki pages. Update on every ingest/change.

## Overview
- [[overview]] — what the PdM classifier is, architecture, current status snapshot

## Topics
- [[pipeline]] — the 7 stages + shared engine, constants, robustness fixes
- [[datasets]] — run inventory, exclusions, column-drift + leakage-pressure saga
- [[results]] — leakage-free metrics, ablation, feature importance, honesty notes
- [[training-and-models]] — two-stage training documentation: why 68%→89%, every model explained in plain language
- [[open-issues]] — ranked remaining tasks (Simulink verify, data gaps, SHAP)

## Records
- `log.md` — chronological append-only record of every change
- `user-instructions-log.md` — standing user preferences/decisions to follow on future work
- `channel_validation.csv` (in `pdm/artifacts/`) — per-channel unit/range/decision report

## Running the pipeline
- `notebooks/00_config_inventory.ipynb` … `07_two_stage_pipeline.ipynb` —
  one self-contained notebook per stage; 07 trains the deployed two-stage
  system. See [[pipeline]] and [[training-and-models]].
- `README.md` (repo root) — install/setup + quick-start for the GUI and retraining.

## Sources
Raw Simulink logs in `RAW_DIR/raw/` (13 used runs; excluded/duplicate files
live in `_archive/raw_excluded/`). Mapped via `FILE_MAP` in `pdm_common.py`;
see [[datasets]] for the table.
