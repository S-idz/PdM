# Predictive Maintenance — Engine + Alternator + Hydraulic Pump

Detects and diagnoses faults in a MATLAB Simulink/Simscape sensor log using a
**two-stage system**:

1. **Stage 1 — Health Monitor:** is this log Healthy or anomalous?
   (one-class detector built from the healthy baseline — 100% fault detection)
2. **Stage 2 — Fault Diagnoser:** which fault is it —
   **Leakage / PumpDisplacement / GeneratorFault** — or **"Unknown fault"**
   if the model isn't confident (validated on an unseen fault type).

Headline numbers (all leakage-free, leave-one-run-out):
**89.1% accuracy, macro F1 0.906** on the known fault classes; 100% of windows
from an *unseen* fault type correctly routed to "Unknown" instead of mislabeled.
Full story: [`wiki/training-and-models.md`](wiki/training-and-models.md).

---

## Quick start

```bash
# 1. Python 3.11+ recommended
python --version

# 2. Install dependencies (from this folder)
pip install -r requirements.txt

# 3. Run the prediction GUI
streamlit run 07_streamlit_app.py
```

A browser tab opens. Upload an `.xlsx`/`.csv` Simulink log → verdict, confidence,
per-window votes, and plots of the harmonized signals.
(Or open `GUI_run.ipynb` and run its single cell to launch the same GUI.)

---

## Repository layout

| Path | Purpose |
|------|---------|
| `pdm_common.py` | Shared engine: loading, cleaning, feature extraction, model factory. Training and the GUI use the exact same code path — no train/serve skew. |
| `notebooks/00…06` | The pipeline, one self-contained notebook per stage (config → ingest → EDA → validation → windows → features → model compare → final eval). Run in order to retrain from raw data. |
| `notebooks/07_two_stage_pipeline.ipynb` | **The final system**: Stage 1 health monitor + Stage 2 ensemble + confidence gate, model comparison, all validation plots. |
| `07_streamlit_app.py` | Upload-and-predict GUI (`GUI_run.ipynb` launches it from Jupyter). |
| `artifacts/` | Trained models (`two_stage_model.joblib`, legacy `model.joblib`), cached data (`features.parquet`), and every report table (`two_stage_metrics.json`, `two_stage_model_comparison.csv`, `feature_importance.csv`, …). |
| `RAW_DIR/raw/` | The 13 raw Simulink runs the pipeline uses (12 fault + 1 healthy). |
| `wiki/` | Project documentation — start at [`wiki/index.md`](wiki/index.md). Training deep-dive: [`wiki/training-and-models.md`](wiki/training-and-models.md). |
| `docs/` | Per-stage explanation documents (`.docx`), GUI output sample, Jupyter guide. |
| `_archive/` | Excluded raw files (duplicates / unusable clips) and superseded scripts — kept for provenance, not used by anything. |
| `requirements.txt` | Python dependencies. |

---

## How it works (short version)

**Features, not raw levels.** The simulation runs sit at very different operating
points (pump pressure 0.03–4126 bar, DC bus 26–182 V), so the models never see
absolute signal levels — only **scale-invariant** features (coefficient of
variation, crest factor, ripple, spectral shape) and **per-run baseline
deviation** (how signals change after the fault injection at t = 0.1 s relative
to that same run's pre-fault window). This detects the *fault*, not the
operating point. Common channels: **pressure, load current, V_dc, V_ac**.

**Honest validation.** Every score uses **leave-one-run-out** splits: the model
is always tested on a run it has never seen a single window of. Random window
splits would leak run identity and fake ~99% accuracy — see
[`wiki/training-and-models.md §3`](wiki/training-and-models.md).

**Cleaning before features** (all thresholds are physics facts or documented
constants in `pdm_common.py`): column harmonization across drifting Simulink
export names → duplicate removal (counted, not silent) → physically-impossible
values treated as missing → uniform 10 kHz resampling → Hampel outlier clipping
→ dataset-wide constant-column drop → low-variance feature filter.

---

## Retraining from raw data

```bash
jupyter notebook notebooks/
```

Run the stage notebooks in order **00 → 07**. Each one re-runs a stage and
prints its own tables/plots. Stage 01 ingests `RAW_DIR/raw/`, 04 rebuilds
features, 05 compares baseline models, 06 saves the legacy flat model,
**07 trains and saves the deployed two-stage system**
(`artifacts/two_stage_model.joblib`).

Raw data location defaults to `RAW_DIR/raw/` inside this repo; override with
the `PDM_RAW_DIR` environment variable if your logs live elsewhere.

---

## Known limitations (honest by design)

- **Healthy has 1 independent run** → Stage 1's false-alarm rate is measured
  in-run only. A second healthy run (new seed/operating point, not a re-export)
  is the top data request.
- **FlexibleShaft cannot be *named*, only *flagged*** — its runs were simulated
  at incompatible operating points, so it is not a trainable class. Stage 1
  detects it (100%) and the confidence gate routes it to "Unknown fault."
- **GeneratorFault recall 70.7%** — one of its three runs was exported with
  different Simulink settings; re-exporting it would close the gap.
- A **FlexibleShaft** fault is reported as **"Unknown fault (possible
  FlexibleShaft)"** — detected with 100% reliability, but not nameable until
  more FlexibleShaft runs exist.

---

## Troubleshooting

- **`streamlit: command not found`** → `python -m streamlit run 07_streamlit_app.py`
- **Model fails to load / version warning** → reinstall via `requirements.txt`
  (don't pin older scikit-learn/lightgbm/catboost).
- **GUI says "could not extract enough windows"** → the uploaded log is too
  short or missing expected channels; it needs data past t = 0.1 s.
