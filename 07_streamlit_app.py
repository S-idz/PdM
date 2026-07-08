# ============================================================================
# CELL 07 -- Streamlit GUI.   Run from a terminal:
#     streamlit run "F:/jyupter notebook/pdm/07_streamlit_app.py"
# Upload a Simulink xlsx/csv log -> the SAME pdm_common transforms used in
# training (no train/serve skew) -> per-window prediction -> aggregated verdict
# (Healthy / which fault) with confidence + signal plots.
#
# Layout follows standard condition-monitoring dashboards:
#   sidebar  = brand, model card, color-coded fault-class legend
#   main tab = Upload -> Status banner -> KPI row -> Diagnosis (probability
#              bars + detail table + per-window timeline) -> Sensor signals
#              (one panel per channel, labelled with its physical unit)
#   2nd tab  = plain-English glossary
# One shared color per class everywhere (legend, banner, charts, table).
# Palette validated for light AND dark surfaces incl. color-vision deficiency.
# ============================================================================
import json, numpy as np, pandas as pd, joblib
import matplotlib.pyplot as plt
import streamlit as st
import pdm_common as P

st.set_page_config(page_title="PdM — Engine/Alternator/Pump", layout="wide",
                   initial_sidebar_state="expanded")

# --------------------------------------------------------------------------- #
# Class identity: one color + one icon per class, used everywhere so the eye
# learns "this color = this fault" once and it holds for the whole page.
# Colors pass lightness/chroma/CVD/contrast checks on light & dark surfaces.
# --------------------------------------------------------------------------- #
CLASS_COLORS = {
    "Healthy": "#16a34a", "GeneratorFault": "#dc2626", "PumpDisplacement": "#ea580c",
    "Leakage": "#7c3aed", "Unknown fault": "#b45309",
}
CLASS_ICON = {
    "Healthy": "✅", "GeneratorFault": "⚡", "PumpDisplacement": "🛢️",
    "Leakage": "💧", "Unknown fault": "❓",
}
CLASS_DESCRIPTIONS = {  # display copy only
    "Healthy": "Signals sit inside the healthy baseline — no anomaly detected.",
    "GeneratorFault": "Alternator/generator signals show an abnormal pattern.",
    "PumpDisplacement": "Hydraulic pump displacement (flow output) looks abnormal.",
    "Leakage": "Pressure pattern is consistent with a hydraulic leak.",
    "Unknown fault": "Anomalous, but doesn't confidently match any trained fault — "
                     "possible FlexibleShaft fault or a new fault type. Inspect the machine.",
}
SIGNAL_LABEL = {  # channel -> (display name, physical unit after harmonization)
    "pressure": ("Pump pressure", "bar"), "current": ("Load current", "A"),
    "vdc": ("DC bus voltage", "V"), "vac": ("Alternator AC voltage", "V"),
    "speed": ("Shaft speed", "rpm"), "torque": ("Shaft torque", "N·m"),
    "flow": ("Flow rate", "L/min"), "fuel": ("Fuel rate", "kg/s"),
}
INK = "#8b9099"          # mid-gray chart ink: legible on light and dark themes
SIGNAL_COLOR = "#3b82f6"  # neutral accent for raw signals (not a class color)


def swatch(color: str) -> str:
    return (f'<span style="display:inline-block; width:10px; height:10px; '
            f'border-radius:3px; background:{color}; margin-right:0.5rem;"></span>')


# --------------------------------------------------------------------------- #
# Global style: one font, one heading scale, bordered "card" sections,
# uppercase micro-labels for section headers (standard dashboard chrome).
# --------------------------------------------------------------------------- #
st.markdown("""
<style>
html, body, [class*="css"] {
    font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
}
h1 { font-size: 1.7rem !important; font-weight: 700 !important; }
[data-testid="stMetricLabel"] { font-size: 0.8rem !important; opacity: 0.7; }
[data-testid="stMetricValue"] { font-size: 1.55rem !important; font-weight: 600; }
section[data-testid="stSidebar"] { border-right: 1px solid rgba(128,128,128,0.25); }
.pdm-brand { font-size: 1.15rem; font-weight: 700; margin-bottom: 0.1rem; }
.pdm-brand-sub { font-size: 0.82rem; opacity: 0.65; margin-bottom: 1rem; }
.pdm-section { font-size: 0.78rem; font-weight: 600; letter-spacing: 0.08em;
               text-transform: uppercase; opacity: 0.6; margin: 1.3rem 0 0.5rem 0; }
.pdm-card-title { font-size: 0.95rem; font-weight: 600; margin-bottom: 0.1rem; }
.pdm-legend-row { font-size: 0.9rem; margin: 0.25rem 0; }
.pdm-banner { border-radius: 12px; padding: 1.1rem 1.4rem; margin: 0.4rem 0 1rem 0; }
.pdm-banner-title { font-size: 1.7rem; font-weight: 700; margin: 0 0 0.25rem 0; }
.pdm-banner-desc { font-size: 1rem; margin: 0; opacity: 0.85; }
</style>
""", unsafe_allow_html=True)

plt.rcParams.update({
    "font.family": "sans-serif", "font.size": 10,
    "text.color": INK, "axes.labelcolor": INK,
    "xtick.color": INK, "ytick.color": INK, "axes.edgecolor": INK,
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.alpha": 0.2, "grid.linewidth": 0.8,
    "figure.facecolor": "none", "axes.facecolor": "none",
})


@st.cache_resource
def load_model():
    bundle = joblib.load(P.ART_DIR / "two_stage_model.joblib")
    meta = json.loads((P.ART_DIR / "meta.json").read_text())
    return bundle, meta


bundle, meta = load_model()
feat_cols = bundle["feat_cols"]
common = meta["common_signals"]
S1_SCALER, S1_COV = bundle["stage1_scaler"], bundle["stage1_cov"]
S1_THR = bundle["stage1_threshold"]
S2_MODEL, S2_LABELS = bundle["stage2_model"], bundle["stage2_labels"]
S2_GATE = bundle["stage2_gate"]
DISPLAY_CLASSES = ["Healthy"] + list(S2_LABELS) + ["Unknown fault"]

# ============================================================================
# Sidebar -- brand, model card, color-coded class legend
# ============================================================================
with st.sidebar:
    st.markdown('<div class="pdm-brand">🛠️ PdM Classifier</div>'
                '<div class="pdm-brand-sub">Engine · Alternator · Hydraulic Pump</div>',
                unsafe_allow_html=True)

    with st.container(border=True):
        st.caption("MODEL")
        st.markdown("**Two-stage: Health Monitor → Fault Diagnoser**")
        st.caption("Stage 1: one-class healthy baseline (Mahalanobis). "
                   "Stage 2: vote ensemble (ExtraTrees + LightGBM + CatBoost) "
                   f"with a {S2_GATE:.0%} confidence gate.")
        st.caption("Input signals: " + ", ".join(common))

    with st.container(border=True):
        st.caption("VERDICT CLASSES")
        for c in DISPLAY_CLASSES:
            st.markdown(
                f'<div class="pdm-legend-row">{swatch(CLASS_COLORS.get(c, "#666"))}'
                f'{CLASS_ICON.get(c, "•")} {c}</div>',
                unsafe_allow_html=True)
        st.caption("❓ Unknown fault = anomalous but below the confidence gate — "
                   "includes FlexibleShaft, which has too little training data "
                   "to be named directly.")

tab_dashboard, tab_glossary = st.tabs(["📊  Dashboard", "📖  Glossary"])

# ============================================================================
# Dashboard tab
# ============================================================================
with tab_dashboard:
    st.markdown("# Predictive Maintenance Dashboard")
    st.caption("Upload a Simulink sensor log to get a Healthy / fault-type verdict. "
               "Terms are explained in the Glossary tab.")

    # ---- 1. Upload ---------------------------------------------------------
    st.markdown('<div class="pdm-section">1 · Sensor log</div>', unsafe_allow_html=True)
    with st.container(border=True):
        up = st.file_uploader("Simulink log (.xlsx or .csv)", type=["xlsx", "csv"],
                              label_visibility="collapsed")
        if up is not None:
            df_raw = pd.read_csv(up, low_memory=False) if up.name.endswith(".csv") else pd.read_excel(up)
            st.caption(f"**{up.name}** — {df_raw.shape[0]:,} rows × {df_raw.shape[1]} columns")

    if up is None:
        st.info("Awaiting upload. The log must contain a time column plus the "
                f"trained signal channels ({', '.join(common)}) and cover the "
                f"fault-active region (t ≥ {P.FAULT_T} s).")
        st.stop()

    try:
        feats = P.process_uploaded(df_raw, common, fname=up.name)
    except KeyError as e:
        st.error(f"Signal channel {e} was not found in this log. The model needs "
                 f"all of: {', '.join(common)}. Check the file's column names "
                 "against a known-good Simulink export.")
        st.stop()
    if feats.empty:
        st.error(f"Could not extract ≥{P.MIN_WINDOWS} complete windows from the "
                 f"fault-active region (t ≥ {P.FAULT_T}s). Is the log long enough / "
                 "are the expected signal columns present?")
        st.stop()

    X = np.nan_to_num(
        feats.reindex(columns=feat_cols, fill_value=0.0).to_numpy(),
        nan=0.0, posinf=0.0, neginf=0.0)

    # ---- Two-stage decision flow, per window -------------------------------
    # Stage 1: distance to healthy baseline. Below threshold -> Healthy.
    dist = S1_COV.mahalanobis(S1_SCALER.transform(X))
    is_anomalous = dist > S1_THR
    # Stage 2: name the fault, but only above the confidence gate.
    proba2 = S2_MODEL.predict_proba(X)                  # cols in S2_LABELS order
    top_p = proba2.max(1)
    top_lbl = np.array(S2_LABELS)[proba2.argmax(1)]
    win_pred = [
        "Healthy" if not anom else
        (lbl if p >= S2_GATE else "Unknown fault")
        for anom, lbl, p in zip(is_anomalous, top_lbl, top_p)
    ]
    votes = pd.Series(win_pred).value_counts()
    verdict = votes.idxmax()
    agreement = votes.max() / len(feats)
    # confidence: healthy = share below baseline; named fault = mean stage-2
    # probability on its windows; unknown = share of anomalous-but-gated windows
    if verdict == "Healthy":
        conf = float((~is_anomalous).mean())
    elif verdict == "Unknown fault":
        conf = float(agreement)
    else:
        mask = np.array(win_pred) == verdict
        conf = float(proba2[mask, S2_LABELS.index(verdict)].mean())
    # vote share per display class drives the "Class probability" chart
    mean_p = np.array([votes.get(c, 0) / len(feats) for c in DISPLAY_CLASSES])
    cls_order = DISPLAY_CLASSES

    # ---- 2. Machine status -------------------------------------------------
    st.markdown('<div class="pdm-section">2 · Machine status</div>', unsafe_allow_html=True)
    accent = CLASS_COLORS.get(verdict, "#374151")
    icon = CLASS_ICON.get(verdict, "❓")
    st.markdown(
        f"""<div class="pdm-banner" style="background:{accent}14; border:1px solid {accent}44;">
        <p class="pdm-banner-title" style="color:{accent};">{icon} {verdict}</p>
        <p class="pdm-banner-desc">{CLASS_DESCRIPTIONS.get(verdict, "")}</p>
        </div>""", unsafe_allow_html=True)

    if verdict == "Unknown fault":
        st.warning("The machine is **anomalous** (Stage 1 is confident about that), "
                   "but the fault type doesn't confidently match Leakage, "
                   "PumpDisplacement, or GeneratorFault. A **FlexibleShaft fault "
                   "produces exactly this verdict** — it has too little training "
                   "data to be named directly. Schedule an inspection.")
    if verdict == "Healthy":
        st.info("Note: the healthy baseline comes from one reference run — "
                "a log from a very different operating point may be flagged "
                "anomalous even if the machine is fine.")
    if conf < 0.5:
        st.warning(f"Confidence is {conf:.0%} — the windows disagree. "
                   "Treat this verdict as indicative, not conclusive.")

    m1, m2, m3 = st.columns(3)
    m1.metric("Confidence", f"{conf:.0%}",
              help="Healthy: share of windows inside the healthy baseline. Named fault: "
                   "mean ensemble probability on that fault's windows. Unknown: share of "
                   "windows routed to Unknown.")
    m2.metric("Anomalous windows", f"{is_anomalous.mean():.0%}",
              help="Stage 1: share of windows beyond the healthy baseline threshold.")
    m3.metric("Windows analysed", f"{len(feats):,}",
              help="Number of overlapping time-slices extracted from the fault-active region.")

    # ---- 3. Diagnosis ------------------------------------------------------
    st.markdown('<div class="pdm-section">3 · Diagnosis</div>', unsafe_allow_html=True)
    order = np.argsort(mean_p)                     # ascending -> top class on top of hbar
    cls_sorted = [cls_order[i] for i in order]
    p_sorted = mean_p[order]

    c1, c2 = st.columns([3, 2], gap="large")
    with c1:
        with st.container(border=True):
            st.markdown('<div class="pdm-card-title">Verdict share</div>',
                        unsafe_allow_html=True)
            st.caption("Share of windows assigned to each verdict by the two-stage "
                       "flow — the final verdict is the highest bar.")
            fig, ax = plt.subplots(figsize=(5.6, 2.9))
            bars = ax.barh(cls_sorted, p_sorted, height=0.55,
                           color=[CLASS_COLORS.get(c, "#7f7f7f") for c in cls_sorted])
            for c, p, b in zip(cls_sorted, p_sorted, bars):
                ax.text(b.get_width() + 0.015, b.get_y() + b.get_height() / 2,
                        f"{p:.1%}", va="center", fontsize=9.5,
                        fontweight="bold" if c == verdict else "normal")
            ax.set_xlim(0, 1.14)
            ax.set_xticks([]); ax.grid(False)
            ax.spines["bottom"].set_visible(False)
            ax.tick_params(axis="y", length=0)
            fig.tight_layout()
            st.pyplot(fig, use_container_width=True)
            plt.close(fig)
    with c2:
        with st.container(border=True):
            st.markdown('<div class="pdm-card-title">Detail table</div>',
                        unsafe_allow_html=True)
            st.caption("Exact numbers behind the chart.")
            detail = pd.DataFrame({
                "Class": cls_sorted[::-1],
                "Probability": (p_sorted[::-1] * 100).round(1),
                "Window votes": [int(votes.get(c, 0)) for c in cls_sorted[::-1]],
            })
            st.dataframe(
                detail, hide_index=True, use_container_width=True,
                column_config={
                    "Probability": st.column_config.ProgressColumn(
                        "Probability", format="%.1f%%", min_value=0, max_value=100),
                    "Window votes": st.column_config.NumberColumn("Window votes"),
                })

    with st.container(border=True):
        st.markdown('<div class="pdm-card-title">Per-window classification timeline</div>',
                    unsafe_allow_html=True)
        st.caption("Each cell is one time-slice of the log, colored by the class it "
                   "voted for (legend in the sidebar). A solid strip = unanimous; "
                   "a color change = the signal changed behaviour partway through.")
        fig, ax = plt.subplots(figsize=(10.5, 0.55))
        ax.bar(range(len(win_pred)), [1] * len(win_pred), width=0.92,
               color=[CLASS_COLORS.get(c, "#7f7f7f") for c in win_pred])
        ax.set_xlim(-0.6, len(win_pred) - 0.4); ax.set_ylim(0, 1)
        ax.set_yticks([]); ax.grid(False)
        ax.spines["left"].set_visible(False); ax.spines["bottom"].set_visible(False)
        ax.set_xticks([0, len(win_pred) - 1])
        ax.set_xticklabels(["first window", "last window"], fontsize=8.5)
        ax.tick_params(axis="x", length=0)
        fig.tight_layout(pad=0.3)
        st.pyplot(fig, use_container_width=True)
        plt.close(fig)

    # ---- 4. Sensor signals -------------------------------------------------
    spec = P.infer_spec(df_raw)
    h = P.harmonize(df_raw.rename(columns={df_raw.columns[0]: "time"}), spec)
    clean = P.clean_run(P.resample_uniform(h))
    if not clean.empty:
        st.markdown('<div class="pdm-section">4 · Sensor signals</div>',
                    unsafe_allow_html=True)
        with st.container(border=True):
            st.caption("Uploaded data after cleaning and unit conversion — the input "
                       "the model actually reads. One panel per channel; units differ, "
                       "so each panel has its own scale.")
            sig_cols = [c for c in common if c in clean.columns]
            ncols = 2
            nrows = -(-len(sig_cols) // ncols)
            fig, axes = plt.subplots(nrows, ncols, figsize=(10.5, 2.1 * nrows),
                                     sharex=True, squeeze=False)
            for ax, c in zip(axes.flat, sig_cols):
                name, unit = SIGNAL_LABEL.get(c, (c, ""))
                ax.plot(clean["time"], clean[c], lw=1, color=SIGNAL_COLOR)
                ax.set_title(name, fontsize=9.5, loc="left", fontweight="bold")
                ax.set_ylabel(unit)
            for ax in axes.flat[len(sig_cols):]:
                ax.set_visible(False)
            for ax in axes[-1]:
                if ax.get_visible():
                    ax.set_xlabel("time [s]")
            fig.tight_layout()
            st.pyplot(fig, use_container_width=True)
            plt.close(fig)

# ============================================================================
# Glossary tab
# ============================================================================
with tab_glossary:
    st.markdown("# 📖 Glossary")
    st.caption("Plain-English explanation of every term on the Dashboard.")

    terms = [
        ("Machine status / verdict", "The model's final answer: is the machine "
         "Healthy, or which fault it looks like. It is simply the class with the "
         "highest mean probability."),
        ("Confidence", "How sure the system is about the verdict. For Healthy: the "
         "share of time-slices inside the healthy baseline. For a named fault: the "
         "diagnoser's average probability on that fault's slices. For Unknown "
         "fault: the share of slices routed to Unknown."),
        ("Anomalous windows", "The share of time-slices Stage 1 flags as beyond "
         "the healthy baseline. 0% means fully healthy-looking; 100% means every "
         "slice is abnormal."),
        ("Windows analysed", "The uploaded log is cut into many small, overlapping "
         'time-slices ("windows"). The model scores each slice separately, then '
         "combines the results. The count depends only on how long the uploaded "
         "file is — it is not a fixed or chosen number."),
        ("Verdict share", "For every possible verdict, the share of time-slices "
         "the two-stage flow assigned to it. The numbers add up to 100% across "
         "all verdicts."),
        ("Stage 1 — Health Monitor", "Before any fault naming, every window is "
         "compared against a healthy reference baseline (a statistical distance). "
         "Windows inside the baseline are Healthy; windows beyond it move on to "
         "fault diagnosis."),
        ("Confidence gate / Unknown fault", "The fault diagnoser only names a "
         "fault when its probability is at least 90%. Below that, the window is "
         "reported as 'Unknown fault' instead of a forced guess. A FlexibleShaft "
         "fault lands here by design — it has too little training data to be "
         "named reliably."),
        ("Per-window classification timeline", "The windows laid out left-to-right "
         "in time order, each colored by the class it voted for. A solid strip is "
         "a unanimous verdict; a color change means the signal changed behaviour "
         "partway through the log."),
        ("Sensor signals", "The uploaded file's raw sensor readings (pressure, "
         "current, DC voltage, AC voltage) after cleaning and conversion to "
         "standard units — the actual data the model reads. Shown one panel per "
         "channel because the units differ."),
        ("Healthy baseline caution", "The healthy reference comes from a single "
         "simulation run. A log recorded at a very different operating point may "
         "be flagged anomalous even if the machine is fine — more healthy runs "
         "would widen the baseline."),
    ]
    cols = st.columns(2, gap="medium")
    for i, (title, body) in enumerate(terms):
        with cols[i % 2]:
            with st.container(border=True):
                st.markdown(f"**{title}**")
                st.caption(body)
