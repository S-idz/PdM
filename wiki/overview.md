# Overview — PdM Fault Classifier

Predictive-maintenance classifier for an **Engine + Alternator + Hydraulic-Pump**
rig, trained on Simulink/Simscape fault simulations. Given a sensor log it reports
**Healthy vs fault type** (5 classes) from operating-point-invariant features.

## Architecture (why it looks the way it does)
- Runs sit at **incompatible operating points** (DC bus 28 V vs 182 V, pump
  pressure 1.5 vs ~1900 bar). Absolute levels would let a classifier cheat on
  operating point instead of detecting faults.
- So every model feature is either **scale-invariant** (coefficient of variation,
  crest, ripple, spectral shape, correlations) or a **per-run baseline deviation**
  (post-fault change vs *this run's own* pre-fault window `[0.05, 0.10)s`).
- **One code path** (`pdm_common.py`) for training and serving → no train/serve skew.
- **Leakage-free evaluation**: run-grouped `GroupKFold` — windows of one run never
  split across train/test.

## Pipeline (7 stages)
`pdm_common.py` = shared engine. Numbered stages are notebook cells / scripts:
00 config · 01 load+harmonize+clean · 02 inspect · 02b channel validation ·
03 window+label · 04 features · 05 model compare + ablation · 06 final eval +
persistence · 07 Streamlit app. Each stage also exists as its own notebook
under `notebooks/` (recommended entry point — run 00→06 in order, each shows
its own tables/plots). See [[pipeline]].

## Current status (2026-07-08)
- **Architecture upgraded to a two-stage system** (`notebooks/07_two_stage_pipeline.ipynb`):
  Stage 1 one-class health monitor (Mahalanobis vs healthy baseline, 100% fault
  detection) → Stage 2 soft-vote ensemble (ExtraTrees+LightGBM+CatBoost) over
  3 fault classes → 0.90 confidence gate routing unseen faults to "Unknown".
- Honest leave-one-run-out: **acc 0.891, macro-F1 0.906** (was 0.69/0.49 flat
  5-class). Leakage 1.00 · PumpDisplacement 0.89 · GeneratorFault 0.83 (F1).
  FlexibleShaft: not a trained class — detected by Stage 1, rejected to
  "Unknown" by the gate (100% validated). See [[results]] and
  [[training-and-models]] for the full explanation.
- Repo reorganized: excluded/duplicate raw files + superseded `.py` stage
  scripts moved to `_archive/`; notebooks are the single source of truth.
- Deployed bundle: `artifacts/two_stage_model.joblib`.

## Open items
Wire the GUI to the two-stage bundle; Simulink-verify the leakage pressure
scope; 2nd Healthy run; comparable FlexibleShaft runs. See [[open-issues]].

Chronological record: `log.md`. Machine-readable index: `index.md`.
