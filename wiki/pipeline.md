# Pipeline

Shared engine `pdm_common.py` + numbered stages. Every stage reads/writes
`artifacts/`. Training and the Streamlit GUI use the SAME `pdm_common` functions.

Each stage is a **standalone notebook** under `notebooks/` with printed tables
and plots after every step. Launch Jupyter from the project root,
`jupyter notebook notebooks/`, and run 00 → 07 in order.
(2026-07-08: the parallel `NN_name.py` scripts and the legacy all-in-one
`PdM_pipeline.ipynb` were moved to `_archive/` — notebooks are the single
source of truth; only `pdm_common.py` and `07_streamlit_app.py` stay as `.py`.)

## Stages
| stage | notebook | does | key output |
|---|---|---|---|
| 00 | `00_config_inventory.ipynb` | print constants + inventory | — |
| 01 | `01_load_harmonize_clean.ipynb` | read → harmonize → **de-duplicate → sanitize invalid values** → resample(10 kHz) → Hampel clean → drop dataset-wide constant columns; auto-derive COMMON | `clean_runs.parquet`, `common_signals.json`, `run_availability.csv` |
| 02 | `02_eda_signals.ipynb` | inspect availability + operating-point spread + **correlation heatmap, histograms, box plots** | (plots) |
| 02b | `02b_validate_channels.ipynb` | **channel/unit validation report** | `channel_validation.csv` |
| 03 | `03_label_window.ipynb` | window fault region (0.02 s, 50% overlap); label | `windows.npz`, `window_counts.csv` |
| 04 | `04_feature_extract.ipynb` | full feature set (no selection here) + **low-variance filter** + feature correlation heatmap (EDA only) | `features.parquet` |
| 05 | `05_train_compare.ipynb` | GroupKFold compare RF/ET/SVM/XGB (**each model's full metrics shown individually**, then combined) + **feature-group ablation** | `model_comparison.csv`, `best_model.json`, `feature_group_ablation.csv` |
| 06 | `06_evaluate_report.ipynb` | OOF report (flat model), confusion/ROC/calibration, **feature importance**, persist | `model.joblib`, `meta.json`, `feature_importance.csv`, `per_class_feature_stats.csv` |
| 07 | `07_two_stage_pipeline.ipynb` | **deployed system**: Stage-1 one-class health monitor + Stage-2 vote ensemble + 0.90 confidence gate; model league; see [[training-and-models]] | `two_stage_model.joblib`, `two_stage_metrics.json`, `two_stage_model_comparison.csv` |
| GUI | `07_streamlit_app.py` (root; `GUI_run.ipynb` launches it) | upload → predict verdict + confidence | — |

## Data-hygiene helpers added to `pdm_common.py` (stage 01/04, methodology unchanged)
- `dedup_run()` — drops exact duplicate rows + duplicate timestamps, returns counts to report.
- `sanitize_invalid()` — flags physically-impossible values (negative pressure/flow/fuel —
  `0` is a physics floor, not a fitted number) as missing; they're then filled by the
  same interpolation `resample_uniform` already uses for missing data (no new imputation path).
- `drop_constant_columns()` — drops a signal only if constant across **every** run
  combined; a column constant within a single run is left alone.
- `low_variance_filter()` — drops features with near-zero variance across all windows
  (`LOW_VAR_THRESH`, a documented constant, same order of magnitude as sklearn's
  `VarianceThreshold` default).
All four are wired into `process_file`/`process_uploaded`, so training and the
Streamlit app apply identical cleaning (no train/serve skew).

## Key constants (`pdm_common.py`)
- `FAULT_T=0.10s` (injection, confirmed from flag columns), stop time varies 0.2–0.5 s.
- `BASELINE_WIN=(0.05, 0.10)s` — per-run pre-fault reference; also resample grid
  start (discards solver init transient). Fixed window is correct while every run
  injects at 0.10 s; adaptive onset-detection marked as future upgrade (`ponytail:`).
- `FS=10 kHz`, `WINDOW_SEC=0.02` (200 samples), `STRIDE_FRAC=0.5`, `MIN_WINDOWS=3`.

## Robustness added this session
- **`_resolve`** — suffix/whitespace-tolerant column matching (Simulink exports
  drift `:1` suffixes and stray spaces; exact matching silently dropped channels).
- **Stage-01 hard assert** — auto-derived COMMON must equal documented `P.COMMON`,
  else fail loudly at ingest (was a silent "Matches: True/False" print).
- **`meta.json` common_signals** read from the training artifact, not the
  `P.COMMON` constant → kills a latent train/serve skew.
- **`infer_spec` pressure fingerprint** — disp and leakage share identical headers;
  tie-broken by the atmospheric pressure fingerprint so an upload never gets
  garbage pressure. See [[datasets]].

Related: [[datasets]] · [[results]] · [[open-issues]] · [[overview]]
