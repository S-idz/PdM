# User Instructions Log (2026-07-07 session)

Standing preferences the user gave this session. Follow these on all future
work in this repo unless the user overrides them explicitly.

## Hard rules
1. **No hardcoding.** Every threshold/constant must be a named, commented
   constant (same style as existing `FAULT_T`, `WINDOW_SEC` in
   `pdm_common.py`) — physics facts (e.g. pressure ≥ 0) and standard-library
   defaults (e.g. sklearn's variance-filter floor) count as "documented", not
   hardcoded. Arbitrary tuned numbers do not.
3. **Separate Jupyter notebook per pipeline stage**, well-segregated cells,
   each notebook self-contained (imports `pdm_common`, reads/writes
   `artifacts/`). Not one big monolithic notebook. See `notebooks/`.
4. **Every notebook cell that does real work must show output** — tables,
   summary stats, and plots (heatmaps, histograms, box plots, confusion
   matrix, etc.) — not just silent variable assignment. Should read clearly
   for someone new to the project.
5. Keep code simple/readable over clever. Prefer plain pandas/sklearn calls
   the user can follow, not abstractions.
6. User has a **very low attention span** — answers must be short, direct,
   one-line-per-point where possible. No long paragraphs unless explicitly
   asked ("give full answer").

## Specific decisions made this session
- **Value cleaning, not row/column deletion**: invalid *values* (e.g.
  negative pressure) get replaced (interpolated), not the whole row/column —
  unless a column is constant across the **entire dataset** (then it's
  genuinely useless and can be dropped). A column constant in just one run
  (e.g. an always-on flag) must NOT be dropped for that reason.
- **Correlation heatmap: yes, always** (EDA/evidence). **Correlation-based
  feature removal: no**, unless the heatmap actually shows many feature pairs
  with `|r| > 0.95` — only remove with evidence, don't remove pre-emptively.
- **Low-variance feature filter: yes**, applied after feature extraction
  (`P.LOW_VAR_THRESH`, documented constant).
- Duplicate rows/timestamps: report the count first, then remove; if zero,
  say so explicitly rather than silently doing nothing.
- CSV exports wanted alongside the existing `.parquet` artifacts (parquet
  alone isn't inspectable enough for the user).
- Wants to see **every model's individual output** (not just one), plus a
  final side-by-side comparison table — this was already true in
  `05_train_compare.py`/`model_comparison.csv`, just wasn't obvious from a
  single glance, so notebook 05 now prints each model's full metrics block
  separately before the combined table.
- Plans to install TensorFlow locally to enable the LSTM sequence-league
  comparison (currently skipped if TF isn't present) — check compatibility
  when that happens, don't assume it works.

## Decisions from the 2026-07-08 session
- **Notebooks are the single source of truth.** No parallel `.py` stage scripts
  — the duplicates were archived. Keep only `pdm_common.py` (shared engine) and
  `07_streamlit_app.py` (GUI) as loose Python files. New pipeline work goes in
  a numbered notebook under `notebooks/`.
- **Two-stage architecture adopted** (user-approved plan): Stage 1 one-class
  healthy baseline, Stage 2 3-class fault ensemble + 0.90 confidence gate.
  This supersedes the earlier "don't restructure stages" rule for the modeling
  part; the data-prep stages 00–06 are unchanged.
- **`Medium_FlexibleShaft_Fault` is removed** from training data (user
  decision). FlexibleShaft is not a supervised class; it validates the
  "Unknown fault" gate.
- Keep `_archive/` for anything removed (excluded raw runs, superseded
  scripts) — archive, don't delete.
- Answers must stay short; when asked "in one line", answer in one line first.

## Things the user explicitly said NOT to do
- Don't drop entire rows/columns over a single bad **value**.
- Don't remove features based on correlation without evidence from the data.
- Don't restructure/rename the pipeline stages or change train/test strategy
  (GroupKFold by `run_id` is correct and intentional, not up for revision).

Related: [[overview]] · [[pipeline]] · [[open-issues]]
