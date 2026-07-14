"""
Generates the remaining Chapter 7 validation figures that don't already
exist in docs/report_figures/, straight from the artifacts/*.csv|json
that back every number in the report. Run from repo root:

    python scripts/make_chapter7_figures.py
"""
import json
import shutil
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
ART = ROOT / "artifacts"
SRC_FIGS = ROOT / "docs" / "report_figures"
OUT = ROOT / "docs" / "chapter7_validation_figures"
OUT.mkdir(parents=True, exist_ok=True)

plt.rcParams.update({"figure.dpi": 150, "font.size": 11})

# ---------------------------------------------------------------------
# 0. Copy over the figures that already exist and are reused in Ch.7
# ---------------------------------------------------------------------
REUSED = {
    "fig3_stage1_separation.png": "fig7_1_stage1_separation.png",
    "fig4_confusion_matrix.png": "fig7_2_confusion_matrix.png",
    "fig5_confidence_gate.png": "fig7_3_confidence_gate.png",
    "fig6_model_comparison.png": "fig7_4_model_comparison.png",
}
for src_name, dst_name in REUSED.items():
    shutil.copy2(SRC_FIGS / src_name, OUT / dst_name)

shutil.copy2(ART / "robustness_calibration.png", OUT / "fig7_9_calibration_curve.png")

# ---------------------------------------------------------------------
# 1. Per-class Precision / Recall / F1  (7.5 Fault Classification Performance)
# ---------------------------------------------------------------------
per_class = pd.DataFrame(
    {
        "class": ["Leakage", "PumpDisplacement", "GeneratorFault"],
        "precision": [1.000, 0.800, 1.000],
        "recall": [1.000, 1.000, 0.707],
        "f1": [1.000, 0.889, 0.829],
        "support": [43, 96, 82],
    }
)

fig, ax = plt.subplots(figsize=(8, 5))
x = range(len(per_class))
w = 0.25
ax.bar([i - w for i in x], per_class["precision"], width=w, label="Precision", color="#4C72B0")
ax.bar(list(x), per_class["recall"], width=w, label="Recall", color="#55A868")
ax.bar([i + w for i in x], per_class["f1"], width=w, label="F1", color="#C44E52")
ax.set_xticks(list(x))
ax.set_xticklabels([f"{c}\n(n={n})" for c, n in zip(per_class["class"], per_class["support"])])
ax.set_ylim(0, 1.15)
ax.set_ylabel("score")
ax.set_title("Stage 2 Per-Class Precision / Recall / F1 (Leave-One-Run-Out)")
ax.axhline(1.0, color="gray", linewidth=0.5, linestyle=":")
ax.legend(loc="lower right", ncol=3)
for i, (p, r, f) in enumerate(zip(per_class["precision"], per_class["recall"], per_class["f1"])):
    ax.text(i - w, p + 0.02, f"{p:.2f}", ha="center", fontsize=8)
    ax.text(i, r + 0.02, f"{r:.2f}", ha="center", fontsize=8)
    ax.text(i + w, f + 0.02, f"{f:.2f}", ha="center", fontsize=8)
fig.tight_layout()
fig.savefig(OUT / "fig7_5_per_class_precision_recall_f1.png")
plt.close(fig)

# ---------------------------------------------------------------------
# 2. Confidence gate ablation (7.6 Confidence Gate Validation)
# ---------------------------------------------------------------------
gate = pd.read_csv(ART / "robustness_gate_analysis.csv")
labels = ["0.90\n(deployed)", "0.701\n(5th-percentile)"]
metrics = ["known_pass_rate", "accuracy_on_passed", "unknown_fault_rejected"]
metric_labels = ["Known-fault pass rate", "Accuracy on passed windows", "Unseen fault rejected"]
colors = ["#4C72B0", "#55A868", "#C44E52"]

fig, ax = plt.subplots(figsize=(8, 5))
x = range(len(labels))
w = 0.25
for i, (m, lab, c) in enumerate(zip(metrics, metric_labels, colors)):
    vals = gate[m].values
    ax.bar([xi + (i - 1) * w for xi in x], vals, width=w, label=lab, color=c)
    for xi, v in zip(x, vals):
        ax.text(xi + (i - 1) * w, v + 0.02, f"{v*100:.1f}%", ha="center", fontsize=8)
ax.set_xticks(list(x))
ax.set_xticklabels(labels)
ax.set_ylim(0, 1.15)
ax.set_ylabel("rate")
ax.set_title("Confidence Gate Ablation: Fixed 0.90 vs. Percentile Threshold")
ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.12), ncol=1, fontsize=9)
fig.tight_layout()
fig.savefig(OUT / "fig7_6_gate_ablation.png")
plt.close(fig)

# ---------------------------------------------------------------------
# 3. Stage 1 detector shootout (7.3 Validation Strategy)
# ---------------------------------------------------------------------
s1 = pd.read_csv(ART / "robustness_stage1_comparison.csv")
fig, axes = plt.subplots(1, 2, figsize=(10, 5))

axes[0].bar(s1["detector"], s1["fault_detection"] * 100, color="#4C72B0")
axes[0].set_ylabel("fault detection rate (%)")
axes[0].set_title("Fault Detection Rate")
axes[0].set_ylim(0, 110)
for i, v in enumerate(s1["fault_detection"] * 100):
    axes[0].text(i, v + 2, f"{v:.0f}%", ha="center", fontsize=9)
axes[0].tick_params(axis="x", rotation=20)

axes[1].bar(s1["detector"], s1["separation_margin"], color="#C44E52")
axes[1].set_yscale("log")
axes[1].set_ylabel("separation margin (log scale)")
axes[1].set_title("Healthy/Fault Separation Margin")
for i, v in enumerate(s1["separation_margin"]):
    axes[1].text(i, v * 1.3, f"{v:,.1f}", ha="center", fontsize=8)
axes[1].tick_params(axis="x", rotation=20)

fig.suptitle("Stage 1 Detector Shootout (§7.3 Validation Strategy)")
fig.tight_layout()
fig.savefig(OUT / "fig7_7_stage1_detector_shootout.png")
plt.close(fig)

# ---------------------------------------------------------------------
# 4. Per-run diagnostic breakdown (7.7 End-to-End System Performance)
# ---------------------------------------------------------------------
pf = pd.read_csv(ART / "robustness_per_fold.csv")
pf = pf.sort_values("true_class")
colors_map = {"GeneratorFault": "#C44E52", "Leakage": "#55A868", "PumpDisplacement": "#4C72B0"}
bar_colors = [colors_map[c] for c in pf["true_class"]]

fig, ax = plt.subplots(figsize=(11, 5.5))
bars = ax.bar(pf["held_out_run"], pf["run_accuracy"] * 100, color=bar_colors,
              edgecolor="black", linewidth=0.6)
ax.set_ylabel("run accuracy (%)")
ax.set_ylim(0, 118)
ax.set_title("Per-Run Diagnostic Breakdown (Leave-One-Run-Out)")
ax.tick_params(axis="x", labelsize=8)
plt.setp(ax.get_xticklabels(), rotation=45, ha="right")
for rect, n in zip(bars, pf["n_windows"]):
    h = rect.get_height()
    label = f"{h:.0f}%\n(n={n})"
    y = h + 3 if h > 0 else 3
    va = "bottom"
    ax.text(rect.get_x() + rect.get_width() / 2, y, label, ha="center", va=va, fontsize=7)
from matplotlib.patches import Patch
ax.legend(
    handles=[Patch(color=c, label=k) for k, c in colors_map.items()],
    loc="upper left", bbox_to_anchor=(1.0, 1.0),
)
fig.tight_layout()
fig.savefig(OUT / "fig7_8_per_run_diagnostic_breakdown.png")
plt.close(fig)

# ---------------------------------------------------------------------
# 5. End-to-end summary scorecard (7.7 End-to-End System Performance)
# ---------------------------------------------------------------------
metrics_json = json.loads((ART / "two_stage_metrics.json").read_text())
scorecard = [
    ("Stage 1 fault detection", metrics_json["stage1_fault_detection_rate"] * 100),
    ("Stage 2 macro F1", metrics_json["stage2_macro_f1"] * 100),
    ("Stage 2 accuracy", metrics_json["stage2_accuracy"] * 100),
    ("Gate known-fault pass rate", metrics_json["gate_known_pass_rate"] * 100),
    ("Gate accuracy on passed", metrics_json["gate_accuracy_on_passed"] * 100),
    ("Unseen fault rejected", metrics_json["gate_flex_rejected_as_unknown"] * 100),
]
fig, ax = plt.subplots(figsize=(9, 5))
labels = [m for m, _ in scorecard]
vals = [v for _, v in scorecard]
bars = ax.barh(labels, vals, color="#4C72B0")
ax.set_xlim(0, 110)
ax.set_xlabel("%")
ax.set_title("End-to-End System Performance Summary")
for rect, v in zip(bars, vals):
    ax.text(v + 1.5, rect.get_y() + rect.get_height() / 2, f"{v:.1f}%", va="center", fontsize=9)
ax.invert_yaxis()
fig.tight_layout()
fig.savefig(OUT / "fig7_10_end_to_end_scorecard.png")
plt.close(fig)

print("Wrote figures to", OUT)
for p in sorted(OUT.glob("*.png")):
    print(" -", p.name)
