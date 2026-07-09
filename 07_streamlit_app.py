# ============================================================================
# CELL 07 -- Streamlit dashboard (v2).   Run:  streamlit run 07_streamlit_app.py
#
# Two-stage PdM dashboard, styled after industrial condition-monitoring UIs:
#   header bar -> upload card -> color-coded KPI row -> alarm trend + gauge ->
#   diagnosis (verdict share + "why this diagnosis") -> per-window timeline ->
#   event log -> sensor signals with healthy baseline band -> channel tiles ->
#   export + model trust panel.  Tabs: Dashboard / Window Data / Glossary.
#
# Performance: ALL inference runs once per upload inside st.cache_data; the
# model bundle loads once via st.cache_resource; widget interaction only
# re-renders cached results.  Dual theme via .streamlit/config.toml — cards
# use st.container(border=True) + theme-neutral CSS so both themes work.
# Decision logic is IDENTICAL to the validated two-stage flow (notebook 07).
# ============================================================================
import io, json, time
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

import pdm_common as P

st.set_page_config(page_title="PdM — Engine/Alternator/Pump", layout="wide")

# --------------------------------------------------------------------------- #
# Class identity: ONE fixed color + icon per verdict, used everywhere.
# --------------------------------------------------------------------------- #
CLASS_COLORS = {
    "Healthy": "#16a34a", "Leakage": "#7c3aed", "PumpDisplacement": "#ea580c",
    "GeneratorFault": "#dc2626", "Unknown fault": "#b45309",
}
CLASS_ICON = {
    "Healthy": "✅", "Leakage": "💧", "PumpDisplacement": "🛢️",
    "GeneratorFault": "⚡", "Unknown fault": "❓",
}
CLASS_DESCRIPTIONS = {
    "Healthy": "Signals sit inside the healthy baseline — no anomaly detected.",
    "GeneratorFault": "Alternator/generator signals show an abnormal pattern.",
    "PumpDisplacement": "Hydraulic pump displacement (flow output) looks abnormal.",
    "Leakage": "Pressure pattern is consistent with a hydraulic leak.",
    "Unknown fault": "Anomalous, but doesn't confidently match any trained fault — "
                     "possible FlexibleShaft fault or a new fault type. Inspect the machine.",
}
SIGNAL_LABEL = {
    "pressure": ("Pump pressure", "bar"), "current": ("Load current", "A"),
    "vdc": ("DC bus voltage", "V"), "vac": ("Alternator AC voltage", "V"),
}
INK = "#8b9099"  # mid-gray: legible on light and dark

st.markdown("""
<style>
/* One type scale for the whole app: 0.72 / 0.8 / 0.95 / 1.45 rem. Every card
   title, label and caption below maps onto one of these four sizes so nothing
   looks like it wandered in from a different UI. */
h1 { font-size: 1.45rem !important; font-weight: 700 !important; }
h2, h3 { font-size: 0.95rem !important; font-weight: 700 !important; }
.pdm-micro { font-size: 0.72rem; font-weight: 600; letter-spacing: 0.09em;
             text-transform: uppercase; opacity: 0.6; margin-bottom: 0.15rem; }
.pdm-kpi   { font-size: 1.45rem; font-weight: 700; line-height: 1.2; }
.pdm-sub   { font-size: 0.8rem; opacity: 0.65; }
.pdm-dot   { display:inline-block; width:10px; height:10px; border-radius:50%;
             margin-right:0.45rem; }
.pdm-chip  { display:inline-block; padding:2px 10px; border-radius:999px;
             font-size:0.72rem; font-weight:600; border:1px solid rgba(128,128,128,.35);
             margin-left:0.4rem; opacity:.85; }
.pdm-tile  { display:inline-block; text-align:center; border:1px solid;
             border-radius:10px; padding:0.55rem 1.1rem; margin:0.15rem 0.3rem;
             font-size:0.8rem; font-weight:600; }
.pdm-section { font-size: 0.72rem; font-weight: 700; letter-spacing: 0.08em;
               text-transform: uppercase; opacity: 0.6; margin: 1.0rem 0 0.4rem 0; }
.pdm-card-title { font-size: 0.95rem; font-weight: 700; margin-bottom: 0.1rem; }
div[data-testid="stMarkdownContainer"] p { font-size: 0.8rem; }
div[data-testid="stMarkdownContainer"] strong { font-size: inherit; }
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


def dot(color): return f'<span class="pdm-dot" style="background:{color};"></span>'


def kpi_card(col, label, value, color=None, sub=""):
    with col:
        with st.container(border=True):
            st.markdown(f'<div class="pdm-micro">{label}</div>', unsafe_allow_html=True)
            c = f'color:{color};' if color else ""
            d = dot(color) if color else ""
            st.markdown(f'<div class="pdm-kpi" style="{c}">{d}{value}</div>',
                        unsafe_allow_html=True)
            if sub:
                st.markdown(f'<div class="pdm-sub">{sub}</div>', unsafe_allow_html=True)


# --------------------------------------------------------------------------- #
# Cached loading + inference (compute once, render many)
# --------------------------------------------------------------------------- #
@st.cache_resource
def load_model():
    bundle = joblib.load(P.ART_DIR / "two_stage_model.joblib")
    meta = json.loads((P.ART_DIR / "meta.json").read_text())
    return bundle, meta


@st.cache_data
def load_trust_panel():
    out = {}
    for f, key in [("two_stage_metrics.json", "metrics"), ("robustness_summary.txt", "robust")]:
        p = P.ART_DIR / f
        if p.exists():
            out[key] = json.loads(p.read_text()) if f.endswith(".json") else p.read_text()
    return out


@st.cache_data
def healthy_band():
    """Per-channel healthy min/max band from the cleaned healthy run (once)."""
    p = P.ART_DIR / "clean_runs.parquet"
    if not p.exists():
        return {}
    df = pd.read_parquet(p)
    h = df[df["run_id"].astype(str).str.contains("Healthy", case=False)]
    return {c: (float(h[c].min()), float(h[c].max()))
            for c in SIGNAL_LABEL if c in h.columns and h[c].notna().any()}


bundle, meta = load_model()
feat_cols = bundle["feat_cols"]
common = meta["common_signals"]
S1_SCALER, S1_COV = bundle["stage1_scaler"], bundle["stage1_cov"]
S1_THR = bundle["stage1_threshold"]
S2_MODEL, S2_LABELS = bundle["stage2_model"], bundle["stage2_labels"]
S2_GATE = bundle["stage2_gate"]
DISPLAY_CLASSES = ["Healthy"] + list(S2_LABELS) + ["Unknown fault"]


@st.cache_data(show_spinner="Analyzing log — running the two-stage model …")
def analyze(file_bytes: bytes, fname: str):
    """Full pipeline for one uploaded log. Runs ONCE per file (cached)."""
    buf = io.BytesIO(file_bytes)
    df_raw = pd.read_csv(buf, low_memory=False) if fname.lower().endswith(".csv") \
        else P.read_excel_fast(buf)
    feats = P.process_uploaded(df_raw, common, fname=fname)
    if feats.empty:
        return {"error": "windows", "shape": df_raw.shape}

    X = np.nan_to_num(feats.reindex(columns=feat_cols, fill_value=0.0).to_numpy(),
                      nan=0.0, posinf=0.0, neginf=0.0)
    # ---- identical two-stage decision flow (validated in notebook 07) ------
    dist = S1_COV.mahalanobis(S1_SCALER.transform(X))
    anom = dist > S1_THR
    proba2 = S2_MODEL.predict_proba(X)
    top_p = proba2.max(1)
    top_lbl = np.array(S2_LABELS)[proba2.argmax(1)]
    win_pred = ["Healthy" if not a else (l if p >= S2_GATE else "Unknown fault")
                for a, l, p in zip(anom, top_lbl, top_p)]
    votes = pd.Series(win_pred).value_counts()
    verdict = votes.idxmax()
    if verdict == "Healthy":
        conf = float((~anom).mean())
    elif verdict == "Unknown fault":
        conf = float(votes.max() / len(win_pred))
    else:
        m = np.array(win_pred) == verdict
        conf = float(proba2[m, S2_LABELS.index(verdict)].mean())

    # ---- evidence: mean robust z-score per feature vs healthy baseline ----
    Z = S1_SCALER.transform(X)
    z_mean = pd.Series(np.abs(Z).mean(0), index=feat_cols).sort_values(ascending=False)
    z_dir = pd.Series(Z.mean(0), index=feat_cols)

    # ---- cleaned signals for plotting (downsampled once) -------------------
    if fname in P.FILE_MAP:
        m_ = P.FILE_MAP[fname]
        spec = dict(label="unknown", always_on=m_["always_on"], cols=m_["cols"])
    else:
        spec = P.infer_spec(df_raw)
    h = P.harmonize(df_raw.rename(columns={df_raw.columns[0]: "time"}), spec)
    clean = P.clean_run(P.resample_uniform(h))
    if len(clean) > 4000:
        clean = clean.iloc[::len(clean) // 4000].reset_index(drop=True)

    # ---- event log ----------------------------------------------------------
    events, state = [], None
    for i, (t0, wp, d) in enumerate(zip(feats["t_start"], win_pred, dist)):
        s = "HEALTHY" if wp == "Healthy" else "FAULT"
        if s != state:
            events.append({"window": i + 1, "t_start [s]": round(float(t0), 4),
                           "event": ("Returned inside healthy baseline" if s == "HEALTHY"
                                     else f"Crossed alarm limit — verdict {wp}"),
                           "distance / limit": round(float(d / S1_THR), 1)})
            state = s
    return {
        "fname": fname, "shape": df_raw.shape, "t_start": feats["t_start"].to_numpy(),
        "dist": dist, "anom": anom, "proba2": proba2, "win_pred": win_pred,
        "votes": votes.to_dict(), "verdict": verdict, "conf": conf,
        "z_mean": z_mean.head(8), "z_dir": z_dir,
        "clean": clean, "events": pd.DataFrame(events),
        "channels": [c for c in common if c in clean.columns],
    }


# ============================================================================
# Header bar (no sidebar)
# ============================================================================
with st.container(border=True):
    h1, h2 = st.columns([3, 2])
    with h1:
        st.markdown(
            "# 🛠️ PdM Dashboard — Engine · Alternator · Hydraulic Pump")
    with h2:
        st.markdown(
            '<div style="text-align:right; padding-top:0.5rem;">'
            '<span class="pdm-chip">Two-Stage model</span>'
            '<span class="pdm-chip">ET + LGBM + CatBoost</span>'
            f'<span class="pdm-chip">gate {S2_GATE:.2f}</span>'
            f'<span class="pdm-chip">inputs: {", ".join(common)}</span></div>',
            unsafe_allow_html=True)

tab_dash, tab_data, tab_gloss = st.tabs(["📊  Dashboard", "🗂  Window Data", "📖  Glossary"])

# ---- Upload (shared state) --------------------------------------------------
with tab_dash:
    st.markdown('<div class="pdm-section">1 · Sensor log</div>', unsafe_allow_html=True)
    with st.container(border=True):
        u1, u2 = st.columns([3, 2])
        with u1:
            up = st.file_uploader("Upload a Simulink log (.xlsx / .csv)",
                                  type=["xlsx", "csv"], label_visibility="collapsed")
        with u2:
            sample = None
            raw_dir = next((d for d in (Path("RAW_DIR/raw"),
                                        Path(getattr(P, "RAW_DIR", "")) / "raw",
                                        Path(getattr(P, "RAW_DIR", "")))
                            if d.exists() and any(d.glob("*.xlsx"))), Path("RAW_DIR/raw"))
            if raw_dir.exists():
                opts = ["—"] + sorted(f.name for f in raw_dir.iterdir()
                                      if f.suffix in (".xlsx", ".csv"))
                pick = st.selectbox("Sample dataset", opts, index=0)
                if pick != "—":
                    sample = raw_dir / pick

# deep-link support: ?sample=<raw filename> preloads a sample (demo/screenshots)
if up is None and sample is None:
    qp = st.query_params.get("sample")
    if qp and (raw_dir / qp).exists():
        sample = raw_dir / qp

res = None
if up is not None:
    res = analyze(up.getvalue(), up.name)
elif sample is not None:
    res = analyze(sample.read_bytes(), sample.name)

with tab_dash:
    if res is None:
        st.info("Awaiting a sensor log. The file needs a time column, the trained "
                f"channels ({', '.join(common)}), and data past t = {P.FAULT_T} s.")
    elif res.get("error"):
        st.error(f"Could not extract ≥{P.MIN_WINDOWS} complete windows from the "
                 f"fault-active region (t ≥ {P.FAULT_T}s) of this "
                 f"{res['shape'][0]:,}-row file. Check length and column names.")
    else:
        verdict, conf = res["verdict"], res["conf"]
        accent = CLASS_COLORS[verdict]
        n_win = len(res["win_pred"])
        anom_share = float(res["anom"].mean())

        # ---- 2 · KPI row ----------------------------------------------------
        st.markdown('<div class="pdm-section">2 · Machine status</div>',
                    unsafe_allow_html=True)
        k1, k2, k3, k4 = st.columns(4)
        status = "Healthy" if verdict == "Healthy" else "Fault detected"
        status_icon = "✅" if verdict == "Healthy" else "⚠️"
        kpi_card(k1, "System status", f"{status_icon} {status}",
                 CLASS_COLORS["Healthy"] if verdict == "Healthy" else CLASS_COLORS["GeneratorFault"],
                 f"{res['fname']} · {n_win} windows")
        kpi_card(k2, "Diagnosis", verdict, accent, CLASS_DESCRIPTIONS[verdict][:58] + "…")
        kpi_card(k3, "Anomalous windows", f"{anom_share:.0%}",
                 CLASS_COLORS["Healthy"] if anom_share == 0 else CLASS_COLORS["Unknown fault"] if anom_share < .5 else CLASS_COLORS["GeneratorFault"],
                 "share beyond the healthy baseline (Stage 1)")
        kpi_card(k4, "Confidence", f"{conf:.0%}", accent,
                 "see Glossary for the per-verdict definition")

        if verdict == "Unknown fault":
            st.warning("Anomalous, but the pattern doesn't confidently match a "
                       "trained fault. Schedule an inspection.")

        # ---- 3 · Alarm trend + gauge ----------------------------------------
        st.markdown('<div class="pdm-section">3 · Alarm trend (Stage 1)</div>',
                    unsafe_allow_html=True)
        a1, a2 = st.columns([3, 1.4])
        with a1:
            with st.container(border=True):
                fig = go.Figure()
                fig.add_hline(y=float(S1_THR), line_dash="dash", line_color="#9ca3af",
                              annotation_text="alarm limit", annotation_font_color=INK)
                fig.add_trace(go.Scatter(
                    x=res["t_start"], y=res["dist"], mode="lines+markers",
                    line=dict(color="#9ca3af", width=1),
                    marker=dict(size=9, color=[CLASS_COLORS[w] for w in res["win_pred"]]),
                    hovertemplate="t=%{x:.3f}s<br>distance=%{y:.3g}<extra></extra>",
                    name="window distance"))
                fig.update_yaxes(type="log", title="distance to healthy baseline (log)")
                fig.update_xaxes(title="window start time [s]")
                fig.update_layout(height=300, margin=dict(l=10, r=10, t=28, b=10),
                                  paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                                  showlegend=False,
                                  title=dict(text="Distance vs alarm limit — dots colored by verdict",
                                             font=dict(size=13)))
                st.plotly_chart(fig, use_container_width=True, theme="streamlit")
        with a2:
            with st.container(border=True):
                g = go.Figure(go.Indicator(
                    mode="gauge+number", value=anom_share * 100,
                    number={"suffix": "%", "font": {"size": 34}},
                    gauge={"axis": {"range": [0, 100]},
                           "bar": {"color": accent},
                           "steps": [
                               {"range": [0, 5], "color": "rgba(22,163,74,.25)"},
                               {"range": [5, 50], "color": "rgba(180,83,9,.25)"},
                               {"range": [50, 100], "color": "rgba(220,38,38,.25)"}]},
                    title={"text": "Anomalous windows", "font": {"size": 13}}))
                g.update_layout(height=272, margin=dict(l=25, r=25, t=40, b=5),
                                paper_bgcolor="rgba(0,0,0,0)")
                st.plotly_chart(g, use_container_width=True, theme="streamlit")

        # ---- 4 · Diagnosis --------------------------------------------------
        st.markdown('<div class="pdm-section">4 · Diagnosis</div>', unsafe_allow_html=True)
        d1, d2 = st.columns([3, 2])
        with d1:
            with st.container(border=True):
                st.markdown('<div class="pdm-card-title">Verdict share</div>'
                            '<div class="pdm-sub">windows assigned to each verdict</div>',
                            unsafe_allow_html=True)
                share = [res["votes"].get(c, 0) / n_win for c in DISPLAY_CLASSES]
                vs = go.Figure(go.Bar(
                    x=share, y=DISPLAY_CLASSES, orientation="h",
                    marker_color=[CLASS_COLORS[c] for c in DISPLAY_CLASSES],
                    text=[f"{s:.0%}" for s in share], textposition="outside"))
                vs.update_xaxes(range=[0, 1.12], showticklabels=False)
                vs.update_layout(height=250, margin=dict(l=10, r=10, t=6, b=6),
                                 paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)")
                st.plotly_chart(vs, use_container_width=True, theme="streamlit")
        with d2:
            with st.container(border=True):
                st.markdown('<div class="pdm-card-title">Why this diagnosis</div>'
                            '<div class="pdm-sub">top deviating features</div>',
                            unsafe_allow_html=True)
                st.caption("Mean deviation from the healthy baseline, in robust "
                           "standard units (σ). What the model actually reacted to.")
                ev = res["z_mean"]
                for f_, v in ev.head(5).items():
                    direction = "above" if res["z_dir"][f_] > 0 else "below"
                    st.markdown(
                        f'{dot(accent)}`{f_}` — **{v:,.0f}σ** {direction} healthy',
                        unsafe_allow_html=True)

        # ---- 5 · Timeline -----------------------------------------------------
        with st.container(border=True):
            st.markdown('<div class="pdm-card-title">Per-window timeline</div>'
                        '<div class="pdm-sub">each cell is one time-slice, colored by its verdict</div>',
                        unsafe_allow_html=True)
            figt, ax = plt.subplots(figsize=(10.5, 0.5))
            ax.bar(range(n_win), [1] * n_win, width=0.92,
                   color=[CLASS_COLORS[w] for w in res["win_pred"]])
            ax.set_xlim(-0.6, n_win - 0.4); ax.set_ylim(0, 1)
            ax.set_yticks([]); ax.grid(False)
            ax.spines["left"].set_visible(False); ax.spines["bottom"].set_visible(False)
            ax.set_xticks([0, n_win - 1])
            ax.set_xticklabels([f"{res['t_start'][0]:.2f}s", f"{res['t_start'][-1]:.2f}s"],
                               fontsize=8.5)
            ax.tick_params(axis="x", length=0)
            figt.tight_layout(pad=0.3)
            st.pyplot(figt, use_container_width=True)
            plt.close(figt)
            legend = "&nbsp;&nbsp;".join(
                f'{dot(CLASS_COLORS[c])}<span class="pdm-sub">{c}</span>'
                for c in DISPLAY_CLASSES)
            st.markdown(legend, unsafe_allow_html=True)

        # ---- 6 · Event log ----------------------------------------------------
        st.markdown('<div class="pdm-section">5 · Event log</div>', unsafe_allow_html=True)
        with st.container(border=True):
            if res["events"].empty:
                st.caption("No state transitions — the log is uniform.")
            else:
                st.dataframe(res["events"], hide_index=True, use_container_width=True)

        # ---- 7 · Signals with healthy band -----------------------------------
        st.markdown('<div class="pdm-section">6 · Sensor signals</div>',
                    unsafe_allow_html=True)
        with st.expander("Cleaned signals vs healthy baseline band", expanded=False):
            band = healthy_band()
            clean = res["clean"]
            sig_cols = res["channels"]
            ncols = 2
            nrows = -(-len(sig_cols) // ncols)
            figs, axes = plt.subplots(nrows, ncols, figsize=(10.5, 2.2 * nrows),
                                      sharex=True, squeeze=False)
            for ax, c in zip(axes.flat, sig_cols):
                name, unit = SIGNAL_LABEL.get(c, (c, ""))
                if c in band:
                    ax.axhspan(band[c][0], band[c][1], color=CLASS_COLORS["Healthy"],
                               alpha=0.12, label="healthy range")
                ax.plot(clean["time"], clean[c], lw=1, color="#3b82f6")
                ax.set_title(name, fontsize=9.5, loc="left", fontweight="bold")
                ax.set_ylabel(unit)
            for ax in axes.flat[len(sig_cols):]:
                ax.set_visible(False)
            for ax in axes[-1]:
                if ax.get_visible():
                    ax.set_xlabel("time [s]")
            axes.flat[0].legend(fontsize=8, frameon=False)
            figs.tight_layout()
            st.pyplot(figs, use_container_width=True)
            plt.close(figs)
            st.caption("Green band = min–max range of the healthy reference run. "
                       "Bands far from the signal indicate a different operating "
                       "point as well as (or instead of) a fault.")

        # ---- 8 · Channel tiles ------------------------------------------------
        tiles = ""
        for c in common:
            ok = c in res["channels"]
            col = CLASS_COLORS["Healthy"] if ok else CLASS_COLORS["GeneratorFault"]
            tiles += (f'<span class="pdm-tile" style="border-color:{col}; color:{col};">'
                      f'{SIGNAL_LABEL.get(c, (c, ""))[0]}<br>'
                      f'{"✓ OK" if ok else "✗ MISSING"}</span>')
        with st.container(border=True):
            st.markdown('<div class="pdm-card-title">Channel status</div>' + tiles,
                        unsafe_allow_html=True)

        # ---- 9 · Export + trust panel ----------------------------------------
        e1, e2 = st.columns(2)
        with e1:
            with st.container(border=True):
                st.markdown('<div class="pdm-card-title">Export report</div>',
                            unsafe_allow_html=True)
                wdf = pd.DataFrame({
                    "window": range(1, n_win + 1), "t_start_s": res["t_start"],
                    "stage1_distance": res["dist"], "anomalous": res["anom"],
                    "verdict": res["win_pred"],
                    "stage2_confidence": res["proba2"].max(1)})
                summary = (f"PdM verdict report\nfile: {res['fname']}\n"
                           f"verdict: {verdict}\nconfidence: {conf:.1%}\n"
                           f"anomalous windows: {anom_share:.1%} of {n_win}\n"
                           f"model: two-stage (gate {S2_GATE})\n")
                c1, c2 = st.columns(2)
                c1.download_button("⬇ Verdict summary (.txt)", summary,
                                   file_name="pdm_verdict.txt")
                c2.download_button("⬇ Window table (.csv)", wdf.to_csv(index=False),
                                   file_name="pdm_windows.csv")
        with e2:
            trust = load_trust_panel()
            with st.container(border=True):
                st.markdown('<div class="pdm-card-title">Model trust panel</div>',
                            unsafe_allow_html=True)
                with st.expander("Validation evidence (leave-one-run-out)"):
                    if "metrics" in trust:
                        m = trust["metrics"]
                        st.markdown(
                            f"- Stage 1 fault detection: **{m['stage1_fault_detection_rate']:.0%}**\n"
                            f"- Stage 2 accuracy: **{m['stage2_accuracy']:.1%}**, "
                            f"macro F1 **{m['stage2_macro_f1']:.3f}**\n"
                            f"- Gate {m['gate']}: rejects **{m['gate_flex_rejected_as_unknown']:.0%}** "
                            f"of an unseen fault type\n"
                            f"- Validation: {m['validation']}")
                    if "robust" in trust:
                        st.code(trust["robust"], language=None)

        # ---- 10 · Live replay (optional demo) ---------------------------------
        with st.expander("▶ Live replay (demo) — stream this log window by window"):
            if st.button("Start replay"):
                slot = st.empty()
                for i in range(1, n_win + 1):
                    with slot.container():
                        w = res["win_pred"][i - 1]
                        st.markdown(
                            f'{dot(CLASS_COLORS[w])}**t = {res["t_start"][i-1]:.3f}s** — '
                            f'window {i}/{n_win} → **{w}**', unsafe_allow_html=True)
                        figr, axr = plt.subplots(figsize=(10.5, 0.4))
                        axr.bar(range(i), [1] * i, width=0.92,
                                color=[CLASS_COLORS[x] for x in res["win_pred"][:i]])
                        axr.set_xlim(-0.6, n_win - 0.4); axr.set_ylim(0, 1)
                        axr.axis("off")
                        st.pyplot(figr, use_container_width=True)
                        plt.close(figr)
                    time.sleep(0.12)
                st.success(f"Replay complete — final verdict: {verdict}")

# ============================================================================
# Window Data tab
# ============================================================================
with tab_data:
    if res is None or res.get("error"):
        st.info("Upload a log on the Dashboard tab first.")
    else:
        wdf = pd.DataFrame({
            "Window": range(1, len(res["win_pred"]) + 1),
            "t start [s]": np.round(res["t_start"], 4),
            "Stage 1 distance": np.round(res["dist"], 1),
            "Anomalous": res["anom"],
            "Verdict": res["win_pred"],
            "Confidence": np.round(res["proba2"].max(1) * 100, 1),
        })
        f1, f2 = st.columns([1, 3])
        with f1:
            pick = st.multiselect("Filter verdicts", DISPLAY_CLASSES, default=[])
        if pick:
            wdf = wdf[wdf["Verdict"].isin(pick)]
        st.dataframe(
            wdf, hide_index=True, use_container_width=True, height=520,
            column_config={
                "Confidence": st.column_config.ProgressColumn(
                    "Stage 2 confidence", format="%.1f%%", min_value=0, max_value=100),
            })
        st.caption(f"{len(wdf)} windows shown. Verdict colors on the Dashboard tab.")

# ============================================================================
# Glossary tab
# ============================================================================
with tab_gloss:
    st.markdown("### 📖 Glossary")
    terms = [
        ("Machine status / Diagnosis", "Stage 1 answers *is it healthy?* by measuring "
         "each time-slice's statistical distance to a healthy baseline. If anomalous, "
         "Stage 2 (an ensemble of three tree models) names the fault — but only when "
         "its probability clears the 0.90 confidence gate; otherwise the verdict is "
         "'Unknown fault'."),
        ("Anomalous windows", "Share of time-slices beyond the healthy baseline's "
         "alarm limit. 0% = fully healthy-looking, 100% = every slice abnormal."),
        ("Confidence", "Healthy: share of slices inside the baseline. Named fault: "
         "the ensemble's mean probability on that fault's slices. Unknown: share of "
         "slices routed to Unknown. Treat it as a ranking signal, not a literal "
         "probability (see the Model trust panel)."),
        ("Alarm trend", "Each dot is one time-slice's distance to the healthy "
         "baseline (log scale) against the dashed alarm limit — the classic "
         "condition-monitoring trend view."),
        ("Why this diagnosis", "The features that deviate most from healthy, in "
         "robust standard units (σ). These are the model's actual inputs — evidence, "
         "not decoration."),
        ("Healthy range band", "Green band on the signal plots = min–max of the "
         "healthy reference run for that channel."),
        ("Event log", "State transitions: the first window that crossed the alarm "
         "limit (and any return below it), with time stamps."),
        ("Unknown fault", "Anomalous but below the confidence gate. A FlexibleShaft "
         "fault lands here by design — it has too little training data to be named "
         "reliably. Inspect the machine."),
        ("Healthy baseline caution", "The baseline comes from a single reference "
         "run; a healthy log at a very different operating point may be flagged "
         "anomalous. More healthy runs would widen the baseline."),
    ]
    cols = st.columns(2, gap="medium")
    for i, (title, body) in enumerate(terms):
        with cols[i % 2]:
            with st.container(border=True):
                st.markdown(f"**{title}**")
                st.caption(body)
