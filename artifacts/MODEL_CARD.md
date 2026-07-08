# Model Card — Deployed PdM System

**Source of truth: `two_stage_model.joblib`** (this file supersedes the legacy
`model.joblib` / `meta.json` pair, which describe the earlier flat classifier
and are kept only for provenance).

| Field | Value |
|---|---|
| Bundle | `two_stage_model.joblib` |
| Stage 1 | Mahalanobis distance to healthy baseline (RobustScaler + Ledoit-Wolf), threshold = 1.5 × max healthy distance |
| Stage 2 | Soft-voting ensemble: ExtraTrees(500) + LightGBM(400, 15 leaves) + CatBoost(400, depth 4), balanced class weights |
| Stage 2 classes | Leakage, PumpDisplacement, GeneratorFault |
| Confidence gate | 0.90 — windows below it report "Unknown fault" |
| Features | 51 scale-invariant + baseline-deviation features (`feat_cols` in bundle) |
| Input channels | pressure, current, vdc, vac |
| Validation | leave-one-run-out; acc 0.891, macro F1 0.906 (`two_stage_metrics.json`) |
| Robustness checks | notebook 08; `robustness_*` artifacts (per-fold, calibration, gate, Stage-1 shootout) |
| Trained | 2026-07-08, notebook `07_two_stage_pipeline.ipynb` |
| Served by | `07_streamlit_app.py` (identical `pdm_common.py` preprocessing) |

Known caveats: single healthy run (Stage 1 false-alarm rate in-run only);
ensemble overconfident on the divergent `simplified_generator_fault` run
(see calibration check in notebook 08); FlexibleShaft reports as Unknown fault.
