# Datasets & Signal Mapping

Raw Simulink logs live in the repo root (`.xlsx`/`.csv`) + `pdm/`. `FILE_MAP` in
`pdm_common.py` maps each file → `{label, always_on, cols: canonical→raw}`.
Canonical modelling signals (COMMON) = `{pressure, current, vdc, vac}`.

## Run inventory (13 included)
| class | runs | always_on | notes |
|---|---|---|---|
| Healthy | 1 | no | `Healthy Data 3.xlsx`. Single run → not CV-testable. |
| PumpDisplacement | 4 | no | disp1/2/3 + `pump_disp(st-0.5)`. |
| Leakage | 3 | no | `Leakage_factor.xlsx` + 2 CSVs. Perfectly separable (F1 1.00). |
| FlexibleShaft | 2 | no | Mild + Medium. Dissimilar → OOF F1 0 (data problem). |
| GeneratorFault | 3 | mixed | 1 old always-on + 2 new **timed** (fault@0.1s → real baseline). Was 1 run; now CV-valid. |

## Excluded (evidence-based, in `EXCLUDED`)
- `healthy data 1/2.xlsx` — startup-only garbage (0–0.002 s).
- `X_Sevear/X_Critical_FlexibleShaft` — too short / no fault-active region.
- `Pump_Displacement.xlsx` — byte-identical dup of disp1.
- `simplifiied-generator-fault2(st-0.5).xlsx` — near-dup of gen(st-0.5) (max diff
  0.002) → dropped to avoid train/test leakage.

## Column-name drift (fixed)
Updated exports dropped `:1` suffixes and added stray whitespace (`Torque ` vs
`Torque`). Exact name matching mapped named channels to **nothing** → silent empty
runs. Fixed by `_resolve` (normalize: strip trailing `:N`, whitespace, case).

## The leakage pressure investigation
Evidence-driven, and a good record of *how* the mapping was settled:
1. **Symptom**: leakage `Pump_pressure` column reads ~0.09 (not pressure);
   `Mass_Flow_Rate` reads 1.7e8–4.1e8 Pa (≈1700–4100 bar). Scope names swapped.
2. **Gated** first (`LEAKAGE_PRESSURE_COL=None`) → COMMON shrank to 3 signals,
   pending Simulink check.
3. **Experiment** (`exp_leakage_pressure.py`): restoring pressure gives only
   +0.009 macro-F1 and does **not** fix FlexibleShaft → disproved the hypothesis
   that the gate caused the FlexibleShaft collapse.
4. **Adopted as inferred** on 3 fingerprints (atmospheric floor 1.013e5 Pa at t=0,
   ~1900 bar magnitude, coherent evolution). `LEAKAGE_PRESSURE_COL="Mass_Flow_Rate"`.
5. **Correction** — the channel validation report (02b) caught that the swap is in
   `Leakage_factor.xlsx` **only**; the two leakage CSVs have real pressure in the
   correctly named `Pump_pressure`. Fixed per-file: xlsx→`_LEAKAGE_XLSX`,
   CSVs→`_PUMP_NAMED`. Validation paid for itself.

Status: **inferred, not Simulink-verified.** Revert = `LEAKAGE_PRESSURE_COL=None`
+ shrink `COMMON` (stage 01 asserts the match). `channel_validation.csv` flags all
inferred mappings (leakage pressure; flex current/vac from generic converters).

Related: [[pipeline]] · [[results]] · [[open-issues]] · [[overview]]
