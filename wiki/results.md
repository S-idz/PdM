# Results & Validation

All metrics are **leakage-free out-of-fold** (run-grouped splits).

## CURRENT: two-stage system (2026-07-08) — deployed

`notebooks/07_two_stage_pipeline.ipynb` · bundle `artifacts/two_stage_model.joblib`
· full explanation in [[training-and-models]].

- **Stage 1** (one-class Mahalanobis vs healthy baseline): **100%** fault
  detection (all 239 fault windows, incl. FlexibleShaft), 0% in-run false alarms.
- **Stage 2** (soft-vote ExtraTrees+LightGBM+CatBoost, 3 classes,
  leave-one-run-out): **acc 0.891, macro-F1 0.906**.

| class | precision | recall | note |
|---|---|---|---|
| Leakage | 1.00 | 1.00 | |
| PumpDisplacement | 0.80 | 1.00 | |
| GeneratorFault | 1.00 | 0.71 | `simplified_generator_fault` run exported differently |

- **Confidence gate 0.90**: 100% of unseen FlexibleShaft windows → "Unknown
  fault"; 78.7% of known windows pass the gate at 86.8% accuracy.
- Model league (macro-F1): Vote 0.906 > HistGB 0.902 > RandomForest 0.897 >
  LightGBM 0.887 > LogReg 0.872 > ExtraTrees 0.820 > SVM 0.798
  (`two_stage_model_comparison.csv`).
- `Medium_FlexibleShaft_Fault` run removed by project decision (2026-07-08);
  FlexibleShaft is detected+flagged, not named.

---

## HISTORICAL: flat 5-class model (superseded)

Deployment model was **RandomForest**, 51 features, COMMON =
`{pressure, current, vdc, vac}`. Kept for reference — the 0-recall rows below
are what motivated the two-stage redesign.

## Per-class OOF (flat model, pressure restored)
| class | precision | recall | F1 | support | note |
|---|---|---|---|---|---|
| Leakage | 1.00 | 1.00 | **1.00** | 43 | cleanly separable |
| PumpDisplacement | 0.68 | 0.86 | 0.76 | 96 | |
| GeneratorFault | 0.70 | 0.71 | 0.70 | 82 | now CV-valid (was 1 run) |
| FlexibleShaft | 0.00 | 0.00 | **0.00** | 37 | 2 dissimilar runs — data problem |
| Healthy | 0.00 | 0.00 | **0.00** | 9 | single run — honest worst case |

Overall: **accuracy 0.69, macro-F1 0.49, balanced-acc ~0.51.**

Model comparison (macro-F1): RandomForest 0.503 > ExtraTrees 0.494 > XGBoost 0.467
> SVM_RBF 0.457. LSTM league skipped (no TensorFlow). `model_comparison.csv`.

## Feature-group ablation (validates a design claim)
Claim: always-on faults (no baseline → deviation features ≈0) are separated by
**scale-invariant** features. Tested in stage 05 (`feature_group_ablation.csv`):

| group | # feats | macro-F1 | GeneratorFault recall |
|---|---|---|---|
| all | 51 | 0.49–0.50 | 0.71 |
| invariant only | 39 | ~0.50 | 0.65 |
| deviation only | 12 | ~0.50 | 0.71 |

Invariant-only holds GeneratorFault recall near full → **claim holds** (measured,
not asserted).

## Feature importance (evidence model learns physics, not artifacts)
`feature_importance.csv` — built-in impurity + **permutation** (SHAP fell back on
an env `ValueError`). Top drivers physically sensible: `vac_zcr` (AC-voltage
zero-cross rate), spectral centroids, `vdc/current` ripple & crest, deviation-stds.

## Honesty notes
- FlexibleShaft & Healthy F1 = 0 are **data limits**, not pipeline bugs: 2 runs
  (Mild vs Medium, different operating points) and 1 run respectively. Grouped CV
  correctly refuses to reward memorizing a single run.
- Everything is Simscape-trained → treat metrics as an **upper bound** until
  validated on hardware/experimental data (dominant real-world risk).

## Serving smoke test (app's exact path, headless)
Two-stage GUI (2026-07-08): **5/5 correct**, each at 100% of windows —
Healthy3→Healthy, Leakage_factor→Leakage, disp2→PumpDisplacement,
gen→GeneratorFault, MildFlexible→**Unknown fault** (the designed verdict for
FlexibleShaft). Historical flat-model smoke: 4/4.

Related: [[pipeline]] · [[datasets]] · [[open-issues]] · [[overview]]
