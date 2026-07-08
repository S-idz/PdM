# Open Issues

Ranked. Data limits dominate; pipeline logic is sound.
(2026-07-08: two-stage redesign shipped — issues 2/3 below are now *mitigated
by design* rather than blocking metrics; see [[training-and-models]].)

## Next code task
0. ~~**Wire the GUI to the two-stage bundle.**~~ Done 2026-07-08:
   `07_streamlit_app.py` now loads `two_stage_model.joblib` and runs the full
   decision flow (Stage 1 healthy check → Stage 2 ensemble → 0.90 gate →
   Healthy / named fault / "Unknown fault"). `process_uploaded` gained a
   filename-aware FILE_MAP fallback (fixes uploads whose channels live in
   generic `PS-Simulink Converter` columns, e.g. MildFlexible_shaft).
   Headless smoke test 5/5: Healthy3→Healthy, Leakage_factor→Leakage,
   disp2→PumpDisplacement, gen→GeneratorFault, MildFlexible→Unknown fault
   (each 100% of windows).

## Blocker (user action)
1. **Verify leakage pressure scope in Simulink.** `LEAKAGE_PRESSURE_COL=
   "Mass_Flow_Rate"` for `Leakage_factor.xlsx` is **inferred** (3 fingerprints),
   not confirmed against the model's scope wiring. One glance at the leakage
   subsystem settles it. If wrong: set `None` + shrink `COMMON` (one line each,
   stage 01 asserts). See [[datasets]].
1b. **`Leakage_factor.xlsx` column labels are scrambled at the source**
   (2026-07-08 session). Evidence: its `Load_Current` range [-1.257, 459.708]
   exactly matches the **torque** range in the leakage CSVs; its `Load_Pressure`
   range [0, 28.754] exactly matches the CSVs' **current** range; corr(vdc,
   current) = 0.28 in this run vs 1.0000 in all 12 other runs. So the run's
   `current` features are computed from the wrong physical signal. Fix =
   re-export the file from Simulink with verified scope wiring (same visit as
   issue 1). Leakage recall = 1.00 likely still genuine (CSV runs' pressure
   spikes to 1936–4126 bar), but treat this run's windows as suspect until then.

## Data gaps (limit metrics, not bugs)
2. **FlexibleShaft not nameable** *(mitigated 2026-07-08: Stage 1 detects it
   100%, confidence gate routes it to "Unknown fault" instead of mislabeling;
   `Medium` run removed by project decision, `Mild` kept as the unseen-class
   validator)* — originally: only 2 runs (Mild vs Medium) at different operating
   points; neither generalizes to the other. Fix = more comparable flex runs.
   Confirmed NOT a pressure-gate artifact (experiment disproved).
   Quantified (2026-07-08): standardized centroid distance Mild↔Medium = 5.3,
   but Medium↔pump/gen runs = 2.3 and Mild↔gen = 3.9 — each flex run sits
   inside another class's territory. Cause: different electrical operating
   points (Mild vdc max 142.8 V = leakage sims' setpoint; Medium 182.2 V =
   pump sims'). Biggest gaps: current/vdc crest, cov, ripple, speccen, zcr
   (~1.4 global std). Fix = re-simulate flex severities at ONE common
   operating point.
3. **Healthy F1 = 0** *(mitigated 2026-07-08: Healthy is now the Stage 1
   one-class baseline, not a supervised class — but its false-alarm rate is
   still only measurable in-run)* — single run → not CV-testable. Fix = ≥2
   dedicated healthy runs at varied operating points. Note: `Data_Healthy.xlsx`
   (added 2026-07-08) was a byte-duplicate of `Healthy Data 3.xlsx` → archived.
3b. **GeneratorFault recall 0.6–0.7: same disease as flex, milder**
   (2026-07-08). `simplified_generator_fault` sits 6.4–6.7 from the other two
   gen runs (which are 0.8 apart) — a third distinct operating point, so folds
   holding it out lose recall. Fix = a gen run re-simulated at the shared
   operating point.

## Nice-to-have
4. **SHAP** — preferred importance method, but hits an env `ValueError`; pipeline
   falls back to permutation + built-in importance (same evidence). Fix the
   shap/numpy version pin if the SHAP plots are wanted. See [[results]].
5. ~~**Notebook resync**~~ — resolved 2026-07-08: `PdM_pipeline.ipynb` and the
   duplicate `.py` stage scripts moved to `_archive/`; `notebooks/` is the
   single source of truth (kept: `pdm_common.py`, `07_streamlit_app.py`).
6. **Adaptive baseline** — `BASELINE_WIN` is fixed; correct while injection is
   always 0.10 s. Marked `ponytail:` upgrade path if a future dataset varies it.

Related: [[overview]] · [[pipeline]] · [[datasets]] · [[results]]
