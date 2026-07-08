# Simulink Verification Checklist — Leakage_factor.xlsx (open issue 1b)

Evidence says this file's column labels are scrambled at the export source.
One visit to the Simulink model settles it. Check, in order:

1. **Open the leakage-fault variant** of the model (the one that produced
   `Leakage_factor.xlsx`, "Run 2_ Imported_Data" sheet).
2. **Find the To-Workspace / logging blocks** (or the Data Inspector export
   list) and note which physical signal feeds each logged name:
   - [ ] `Load_Current` — is it wired to the load current sensor, or to
     **shaft torque**? (Its exported range [-1.257, 459.708] exactly matches
     the torque range in the leakage CSV runs.)
   - [ ] `Load_Pressure` — is it wired to pump pressure, or to **current**?
     (Its range [0, 28.754] matches the CSVs' current range.)
   - [ ] `Mass_Flow_Rate` — is it actually **pump pressure**? (Fingerprints:
     atmospheric floor 1.013e5 Pa, ~1900 bar magnitude.)
3. **Cross-check** `corr(vdc, current)`: in every correctly-labelled run it is
   ≈ 1.0000; in this file it is 0.28 — after fixing the wiring it should
   return to ≈ 1.
4. **Re-export the run** with verified wiring, drop it in `RAW_DIR/raw/`, and
   re-run notebooks 01 → 04 → 07 → 08 (each is self-contained).
5. If the re-export changes Leakage windows materially, compare the new
   `two_stage_metrics.json` against the committed one — Leakage recall is
   expected to stay 1.00 (the two CSV leakage runs are correctly labelled and
   carry most of the class).

While in the model, also re-export `simplified_generator_fault` **at the same
operating point as the other two generator runs** — that single run causes
every remaining Stage-2 error (see `robustness_per_fold.csv`).
