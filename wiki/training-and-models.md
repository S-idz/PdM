# Training & Models — Full Documentation

How the final system is trained, why accuracy jumped from **68.5% → 89.1%** (macro F1 **0.49 → 0.906**),
and what every model does, in plain language. Built and validated in
`notebooks/07_two_stage_pipeline.ipynb`; results in `artifacts/two_stage_metrics.json` and
`artifacts/two_stage_model_comparison.csv`.

---

## 1. The starting point: why accuracy was stuck at ~68%

The old design was one flat 5-class classifier:
`Healthy / Leakage / PumpDisplacement / GeneratorFault / FlexibleShaft`.

Its scores were not low because the models were weak — four very different models
(ExtraTrees, RandomForest, XGBoost, SVM) all landed within ~1% of each other. When every model
agrees, the ceiling is in the **data**, and here the ceiling was mathematical:

- **Healthy has exactly 1 usable run.** Our validation splits by *run* (see §3). So whenever the
  Healthy run is in the test fold, the training fold contains **zero** healthy examples. A model
  cannot predict a class it has never seen → Healthy recall = 0%, guaranteed, forever.
- **FlexibleShaft had 2 runs (Medium, Mild) that don't resemble each other.** Train on one, test
  on the other → 0% both directions, in every model. Same trap, slightly disguised.

Macro F1 averages all classes equally, so two guaranteed-zero classes out of five capped macro F1
at roughly 3/5 of whatever the good classes achieved. That is the whole story of the "low accuracy."

## 2. The three changes that fixed it (and how much each was worth)

The improvement came from **restructuring the problem**, not from a magic model. Measured
contribution of each step (all leave-one-run-out, so all honest):

| Step | What changed | Macro F1 | Accuracy |
|---|---|---|---|
| 0 | Old flat 5-class ExtraTrees | 0.494 | 68.5% |
| 1 | Drop the two unlearnable classes from supervised training (Healthy → Stage 1 baseline, FlexibleShaft → Unknown gate); same ExtraTrees, now 3 classes | 0.820 | 80.1% |
| 2 | Swap ExtraTrees for a soft-voting ensemble (ExtraTrees + LightGBM + CatBoost) | **0.906** | **89.1%** |
| 3 | Add the 0.90 confidence gate (doesn't change these numbers; adds "Unknown fault" safety) | — | — |

So ~75% of the gain came from **step 1** — asking the classifier only questions it can actually
answer — and the rest from a stronger model. Nothing was gained by leakage: no run ever appears in
both train and test, and the removed classes are still *handled*, just by the right mechanism
(anomaly detection and rejection instead of supervised classification).

### The final decision flow, per 0.02 s window

```
window features
   │
   ▼
Stage 1: distance to healthy baseline ≤ threshold? ──yes──► HEALTHY
   │ no (anomalous)
   ▼
Stage 2: vote ensemble → top probability ≥ 0.90? ──yes──► named fault
   │ no                                                   (Leakage / Pump / Generator)
   ▼
UNKNOWN FAULT — inspect
(validated: 100% of FlexibleShaft windows land here)
```

## 3. How training is validated: leave-one-run-out

Every score in this project comes from **Leave-One-Group-Out cross-validation with group = run**:

- Take one Simulink run out entirely (all its windows).
- Train on all remaining runs.
- Predict the held-out run. Repeat for every run; pool the predictions.

Why this matters: windows cut from the same run are near-duplicates of each other. A random
window split would put siblings on both sides, and the model would score ~99% by *recognizing the
run*, not the fault — that is **data leakage**, and it produces impressive fake numbers. Run-level
splitting is why our numbers are believable: every prediction is made on a run the model has
never seen a single window of.

## 4. Stage 1 — the Health Monitor, in plain language

Trained on only the 9 windows of the single healthy run (`Healthy Data 3`):

1. **RobustScaler** — rescales every feature using the healthy median and IQR (instead of
   mean/std) so a couple of odd values can't distort the scale. After this, "0" means "typical
   healthy value" for every feature.
2. **Ledoit-Wolf covariance** — learns how healthy features vary *together* (e.g. pressure ripple
   and current ripple move in sync). Plain covariance needs more samples than features
   (we have 9 samples, 51 features), so Ledoit-Wolf blends in a "shrinkage" prior that keeps the
   estimate mathematically valid at tiny sample sizes.
3. **Mahalanobis distance** — for any new window, one number: "how many standard deviations of
   normal variation is this window from the healthy cloud, accounting for correlations?"
4. **Threshold = 1.5 × the worst healthy window.** Below → Healthy. Above → anomalous → Stage 2.

Result: **100% of all 239 fault windows** (all four fault types, including FlexibleShaft) exceed
the threshold; 0% of healthy windows do. Honest caveat: the false-alarm rate is measured on the
same run that set the baseline — validating it properly needs a second independent healthy run.

## 5. Stage 2 — every model tested, what it is, and why it scored what it scored

All models saw identical inputs: 221 windows × 51 features, 3 classes, leave-one-run-out.
(XGBoost was also tested in exploration and scored ≈ HistGB; it errors on folds where a class is
missing from training, so it was dropped from the final notebook.)

| Model | Macro F1 | Accuracy |
|---|---|---|
| **Vote (ET+LGBM+CatBoost)** | **0.906** | **89.1%** |
| HistGradientBoosting | 0.902 | 88.7% |
| RandomForest | 0.897 | 88.2% |
| LightGBM | 0.887 | 87.3% |
| Logistic Regression | 0.872 | 85.5% |
| ExtraTrees | 0.820 | 80.1% |
| SVM (RBF) | 0.798 | 80.1% |

### Decision trees — the shared building block
A decision tree is a flowchart of yes/no questions on features: *"is pressure_ripple > 0.3?"* →
*"is current_kurt > 2.1?"* → …until it reaches a leaf that names a class. One tree memorizes noise
easily, so every strong model below is a way of combining **many** imperfect trees.

### RandomForest — many trees voting (0.897)
Builds ~500 trees, each on a random resample of the windows, and each split only considers a
random subset of features. Individual trees are wrong in different ways; their majority vote
cancels the errors out. Robust, near-zero tuning. Scored high here because tree ensembles suit
tabular sensor features.

### ExtraTrees — RandomForest with random split points (0.820)
Same idea, but instead of searching for the *best* threshold at every split, it picks thresholds
at **random**. That extra randomness usually reduces overfitting — but with only 10 training runs,
it also means splits often land in uninformative places, and GeneratorFault (the hardest class,
three runs exported with different settings) suffered: ExtraTrees got only 50% generator recall
vs ~70% for the boosted models. That single class explains its last-but-one rank.

### Gradient boosting (HistGB 0.902, LightGBM 0.887, CatBoost, XGBoost) — trees that fix each other's mistakes
Forests build trees independently, in parallel. Boosting builds them **in sequence**: tree 1 makes
predictions, tree 2 is trained specifically on tree 1's *errors*, tree 3 on the remaining errors,
and so on, each small tree nudging the answer closer. This focuses capacity exactly on the hard
cases — for us, separating GeneratorFault from PumpDisplacement — which is why all four boosting
implementations beat ExtraTrees. The four differ mainly in engineering: **HistGB** (scikit-learn)
and **LightGBM** (Microsoft) bucket feature values into histograms for speed; **CatBoost**
(Yandex) uses "ordered boosting," which is specifically designed to resist overfitting on small
datasets like ours; **XGBoost** is the classic regularized implementation. We used small trees
(depth 4 / 15 leaves) and a slow learning rate (0.05) — standard protection against overfitting
221 windows.

### Logistic Regression — the sanity check (0.872)
The simplest possible classifier: a weighted sum of the (standardized) features per class,
squashed into probabilities. No trees, no interactions, essentially draws straight lines between
classes. The fact that it scores 0.872 tells us something important: **the three fault classes are
mostly linearly separable in our feature space** — the feature engineering (ripple, kurtosis,
spectral centroid, etc.) already did the hard work. It also confirms the tree models aren't
hallucinating structure that isn't there.

### SVM with RBF kernel — struggled here (0.798)
Finds the boundary that leaves the widest possible margin between classes, using a kernel to bend
that boundary into curves. SVMs shine with moderate feature counts and lots of samples; with 51
features, 221 windows, and run-level distribution shifts between train and test folds, its
carefully-fit boundary generalizes worse than tree votes. Its leakage recall (63%) dragged it down.

### The winner: soft-voting ensemble (0.906 / 89.1%)
Combines **ExtraTrees + LightGBM + CatBoost** by averaging their predicted *probabilities*
("soft" voting) and picking the highest average. Why it wins: the three members make **different
kinds of mistakes** — ExtraTrees is high-randomness bagging, LightGBM is leaf-wise boosting,
CatBoost is ordered boosting. Averaging keeps shared signal and washes out individual quirks. It
beats every single member (including ExtraTrees's 0.820 → 0.906 inside the ensemble) and edges out
HistGB. A second benefit: averaged probabilities are better *calibrated*, which the confidence
gate (§6) depends on.

Final per-class performance (vote ensemble, leave-one-run-out):

| Class | Precision | Recall |
|---|---|---|
| Leakage | 1.000 | 1.000 |
| PumpDisplacement | 0.800 | 1.000 |
| GeneratorFault | 1.000 | 0.707 |

The only remaining confusion: ~29% of GeneratorFault windows are called PumpDisplacement — all
from the `simplified_generator_fault` run, which was exported with different Simulink settings
than the other two generator runs. More consistent generator runs would close this gap.

## 6. The confidence gate — refusing to guess

A classifier trained on 3 fault types will *force* every window into one of those 3, even a fault
it has never seen. That's dangerous in maintenance. So the deployed model only names a fault when
the ensemble's top probability is **≥ 0.90**; below that it reports **"Unknown fault — inspect."**

This was validated on a genuinely unseen fault type: the FlexibleShaft (Mild) run, which the model
never trained on. Its windows' confidences cluster at 0.83–0.86 — below the gate — so **100% of
them are routed to Unknown** instead of being mislabeled. Cost: 21.3% of known-fault windows also
fall below the gate and ask for inspection (a deliberate safety trade-off; the windows that pass
are 86.8% accurate). Because ~19 consecutive windows per run get pooled in practice, run-level
diagnosis is even more reliable than these per-window numbers.

## 7. What was NOT done (and why that matters)

- **No random window splits** — that would leak run identity and fake ~99% accuracy.
- **No SMOTE / synthetic oversampling** — interpolating between windows of the same run
  manufactures fake "independent" samples.
- **No training on Healthy or FlexibleShaft as supervised classes** — 1 run cannot represent a
  class; pretending otherwise produces guaranteed-zero recall (the old system) or leakage.
- **No hyperparameter search against the test folds** — settings are standard defaults for small
  tabular data.

## 8. Honest limits & the path to better numbers

1. **Stage 1 false-alarm rate is unvalidated across machines** — needs a 2nd independent healthy
   run (different seed/operating point, not a re-export).
2. **FlexibleShaft can't be *named*, only flagged** — needs ≥2 more genuinely different
   FlexibleShaft runs to become a trainable 4th class.
3. **GeneratorFault recall (70.7%)** — re-export `simplified_generator_fault` with the same
   settings as the other generator runs, or add one more generator run.

Every one of these is a data collection task, not a modeling task. The models are already at the
ceiling this dataset allows.

---
*Related: [[results]], [[datasets]], [[pipeline]], [[open-issues]]. Model bundle:
`artifacts/two_stage_model.joblib` (Stage 1 scaler+covariance+threshold, Stage 2 ensemble, gate).*
