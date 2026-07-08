"""
pdm_common.py
=============
Shared engine for the Engine + Alternator + Hydraulic-Pump predictive-maintenance
pipeline. Every numbered stage script AND the Streamlit app import from here, so
training and serving run ONE identical code path (no train/serve skew).

Why the design looks the way it does (driven by what the raw data actually is):

* The Simulink runs are at INCOMPATIBLE operating points (DC bus 28 V vs 182 V,
  pump pressure 1.5 bar vs 1900 bar across runs). Absolute signal levels would
  let a classifier cheat on operating point instead of detecting faults.
  -> every model feature is either SCALE-INVARIANT (coefficient of variation,
     crest factor, ripple, correlation, spectral shape) or a per-run
     BASELINE-DEVIATION (post-injection change measured against THIS run's own
     pre-fault reference). Absolute raw stats are kept for reporting only.

* Faults are injected at t = 0.1 s (confirmed from the flag columns across every
  updated export); stop time VARIES 0.2-0.5 s between runs. The solver dumps a
  dense non-physical transient near t = 0. We never touch it: the uniform
  resample grid starts at BASELINE_WIN[0] = 0.05 s, which discards the init
  transient by construction and gives a clean pre-fault reference window
  [0.05, 0.1). Windowing keys off each run's actual t.max(), so longer runs just
  yield more fault windows.

* Healthy class is taken ONLY from dedicated healthy runs (conservative; we do
  NOT relabel the pre-fault part of fault runs as Healthy). One generator run
  (simplified_generator_fault) is always-on so its baseline-deviation is ~0 and
  it leans on the invariant features; the 2 newer generator runs inject at 0.1 s
  and DO have a real pre-fault baseline -- documented, not hidden.

* The Simulink re-exports drift column names (":1" suffix appears/disappears,
  stray whitespace), so every raw<->canonical lookup goes through _resolve. The
  leakage exports additionally swap the pressure/mass-flow scope names, so
  leakage pressure is gated behind LEAKAGE_PRESSURE_COL pending verification.

* The flexible-shaft files log a different, generically named scope set; only
  {pressure, current, V_dc, V_ac} could be identified by unit/value-range
  evidence, so that is the common modelling basis (see FILE_MAP + COMMON).

Known limitations (honest scope, not bugs):
* Everything here is trained on Simscape/Simulink runs. The dominant real-world
  risk is sim-to-hardware generalization (unmodelled noise, sensor dynamics,
  operating points), NOT the baseline scheme -- treat reported metrics as an
  upper bound until validated on measured data.
* The invariant-vs-deviation claim for always-on faults is validated by the
  feature-group ablation in stage 05 (feature_group_ablation.csv), not assumed.
"""
from __future__ import annotations

import re
from pathlib import Path
import numpy as np
import pandas as pd
from scipy import stats, signal as spsignal


# --------------------------------------------------------------------------- #
# Column-name resolver -- the Simulink re-exports drift the scope names between
# runs: a trailing ":1"/":2" suffix appears or disappears and stray whitespace
# creeps in ("Torque " vs "Torque"). Matching raw column names EXACTLY would
# silently drop channels, so every raw<->canonical lookup goes through _resolve,
# which matches case-insensitively after stripping a trailing ":<n>" and spaces.
# --------------------------------------------------------------------------- #
def _norm(name) -> str:
    return re.sub(r":\d+$", "", str(name).strip()).strip().lower()


def _resolve(raw: str, columns) -> str | None:
    """Actual column in `columns` matching raw name under _norm, else None."""
    target = _norm(raw)
    for c in columns:
        if _norm(c) == target:
            return c
    return None

# --------------------------------------------------------------------------- #
# Paths
# --------------------------------------------------------------------------- #
# Portable: derive from this file's location, not a machine-specific path.
# RAW_DIR (parent) is only needed for TRAINING (reading raw logs); the GUI uses
# only PDM_DIR/ART_DIR, so the folder runs on any laptop once deps are installed.
# Override RAW_DIR via the PDM_RAW_DIR env var if your raw logs live elsewhere.
import os
try:                                    # normal: run/imported as a .py file
    PDM_DIR = Path(__file__).resolve().parent
except NameError:                       # pasted into a notebook/REPL cell -> no __file__
    PDM_DIR = Path.cwd()
RAW_DIR = Path(os.environ.get("PDM_RAW_DIR", PDM_DIR.parent))
ART_DIR = PDM_DIR / "artifacts"
ART_DIR.mkdir(parents=True, exist_ok=True)

# --------------------------------------------------------------------------- #
# Physical / pipeline constants  (no magic numbers without justification)
# --------------------------------------------------------------------------- #
FAULT_T = 0.10           # s  -- fault injection time (confirmed from flag columns)
STOP_T = 0.30            # s  -- nominal stop time
BASELINE_WIN = (0.05, 0.10)   # s -- per-run pre-fault reference; also the grid start
# ponytail: fixed window is correct while every run injects at FAULT_T=0.10 s
# (confirmed from the flag columns). Make it adaptive -- detect fault onset per
# run from the flag/derivative and end the baseline just before it -- only if a
# future dataset varies the injection time.
FS = 10_000              # Hz -- uniform resample rate. Well above the electrical
                         #       fundamental seen in the logs; decimates the MHz
                         #       CSV logs and lightly up-samples the xlsx logs to
                         #       ONE common rate so windows are comparable.
WINDOW_SEC = 0.02        # s  -- 200 samples @ FS. Long enough for stable RMS/CoV,
                         #       short enough to yield several windows from the
                         #       ~0.2 s fault region. (FFT res ~50 Hz -> spectral
                         #       features used only as coarse shape descriptors.)
STRIDE_FRAC = 0.5        # 50 % overlap -- limits correlation between windows of
                         #       the same run (run-grouped CV handles the rest).
MIN_WINDOWS = 3          # a run/file must yield >= this many complete fault
                         #       windows to be included; else excluded + reported.
EPS = 1e-12

# --------------------------------------------------------------------------- #
# Data-quality thresholds -- documented physics facts / standard ML floors,
# NOT numbers tuned to this dataset.
# --------------------------------------------------------------------------- #
NONNEGATIVE_SIGNALS = ("pressure", "flow", "fuel")   # cannot be negative by physics
LOW_VAR_THRESH = 1e-8    # variance floor for feature pruning (same order of
                         #       magnitude as sklearn's VarianceThreshold default);
                         #       a feature this flat carries ~no class information.

# Rated operating points (datasheet / user spec). The 0.3 s sims rarely reach
# these, so they are PHYSICS REFERENCES for % deviation features, never the
# learned healthy distribution.
RATED = {
    "torque_Nm": 500.0,      # 450-550 band -> midpoint
    "speed_rpm": 2200.0,
    "vdc": 28.0,             # rectifier output @ 5 ohm
    "vac": 535.0,            # alternator line voltage (AC)
    "pressure_bar": 200.0,
}

# Canonical signals the pipeline understands. Unit target in the comment.
CANON = ["pressure", "current", "vdc", "vac", "speed", "torque", "flow", "fuel"]
#         bar        A         V      V      rpm      Nm       Lpm     kg/s

# Per-canonical unit conversion from the raw Simulink unit -> model unit.
def _to_bar(x):   return x / 1e5          # Pa  -> bar
def _to_rpm(x):   return x * 60.0 / (2 * np.pi)   # rad/s -> rpm
def _to_lpm(x):   return x * 6.0e4        # m^3/s -> L/min
def _ident(x):    return x

UNIT = {
    "pressure": _to_bar, "speed": _to_rpm, "flow": _to_lpm,
    "current": _ident, "vdc": _ident, "vac": _ident,
    "torque": _ident, "fuel": _ident,
}

# --------------------------------------------------------------------------- #
# Leakage pressure mapping -- INFERRED from numeric evidence, not yet confirmed
# against the Simulink scope wiring. The pressure/mass-flow scope-name swap
# exists ONLY in Leakage_factor.xlsx (caught by 02b channel validation when a
# class-wide override briefly fed ~0.002 bar "pressure" from the CSVs):
#   Leakage_factor.xlsx : pressure lives in "Mass_Flow_Rate"
#                         (atmospheric floor 1.013e5 Pa at t=0, 1.7e8-1.9e8 Pa
#                         ~= rated ~1900 bar; "Pump_pressure" holds ~0.09)
#   leakage_fault(*.csv): pressure lives in the correctly named "Pump_pressure"
#                         (same atmospheric floor, 1.9e8-4.1e8 Pa)
# Set to None to gate the xlsx pressure again (that run then contributes no
# pressure -> COMMON must shrink to {current, vdc, vac}; stage 01 asserts).
# --------------------------------------------------------------------------- #
LEAKAGE_PRESSURE_COL: str | None = "Mass_Flow_Rate"   # Leakage_factor.xlsx ONLY

# --------------------------------------------------------------------------- #
# FILE_MAP -- raw file -> {label, always_on, cols: canonical->raw_column}
# Raw column names are matched via _resolve (suffix/whitespace tolerant), so the
# entries below use the plain scope name regardless of any ":1" drift on disk.
# Identities were re-confirmed from the updated exports by value-range evidence
# (see the value-range probe in project notes):
#   pressure=Pump_pressure(Pa) | current=Load_Current(A) | vdc=Rectifier_Voltage
#   (==Load_Voltage) | vac=Voltage(AC swing). Flex scopes: current/vac live in the
#   generic "PS-Simulink Converter4/9" channels; pressure & vdc are named.
# --------------------------------------------------------------------------- #
_PUMP_NAMED = {   # disp*, pump_disp, and the NEW timed generator runs share this
    "pressure": "Pump_pressure", "current": "Load_Current",
    "vdc": "Rectifier_Voltage", "vac": "Voltage",
    "speed": "Angular_Velocity", "torque": "Torque",
    "flow": "Pump_Flow_volume", "fuel": "Fuel_Consumption",
}
# Leakage_factor.xlsx ONLY: pressure/mass-flow scope names swapped (see gate
# comment above). The leakage CSVs use the normal _PUMP_NAMED scheme.
_LEAKAGE_XLSX = {
    "current": "Load_Current", "vdc": "Rectifier_Voltage", "vac": "Voltage",
    "speed": "Angular_Velocity", "torque": "Torque",
    "flow": "Pump_Flow_volume", "fuel": "Fuel_Consumption",
}
if LEAKAGE_PRESSURE_COL:
    _LEAKAGE_XLSX = {"pressure": LEAKAGE_PRESSURE_COL, **_LEAKAGE_XLSX}

_FLEX_MILD = {   # MildFlexible: pressure & vdc named, current/vac in converters
    "pressure": "Pump_pressure", "current": "PS-Simulink Converter4:1",
    "vdc": "Rectifier_Voltage", "vac": "PS-Simulink Converter9:1",
}
_FLEX_MEDIUM = {   # Medium: lost Converter4/16, gained plain "current" + named p
    "pressure": "Pump_pressure", "current": "current",
    "vdc": "Rectifier_Voltage", "vac": "PS-Simulink Converter9:1",
}
_GEN_OLD = {   # simplified_generator_fault.xlsx -- distinct names, ALWAYS ON
    "pressure": "Pump_Pressure", "current": "Load_current", "vdc": "Rectifier_Output",
    "vac": "Voltage", "speed": "Angular_Velocity", "torque": "Engine_Torque",
    "fuel": "Fuel_Consumption",
}

FILE_MAP: dict[str, dict] = {
    # ---- Healthy (dedicated runs only) -------------------------------------
    "Healthy Data 3.xlsx": dict(label="Healthy", always_on=False, cols={
        "pressure": "pressure", "current": "bus_current", "vdc": "bus_voltage",
        "vac": "line_voltage(1,1)", "speed": "angular_velocity",
        "torque": "shaft_torque", "flow": "flow_rate"}),
    # ---- Pump displacement fault (4 runs) ----------------------------------
    "disp1_fault(0.5).xlsx": dict(label="PumpDisplacement", always_on=False, cols=_PUMP_NAMED),
    "disp2_fault(0.3).xlsx": dict(label="PumpDisplacement", always_on=False, cols=_PUMP_NAMED),
    "disp3_fault(0.2).xlsx": dict(label="PumpDisplacement", always_on=False, cols=_PUMP_NAMED),
    "pump_disp(st-0.5).xlsx": dict(label="PumpDisplacement", always_on=False, cols=_PUMP_NAMED),
    # ---- Leakage fault (3 runs; xlsx has the swapped pressure scope) --------
    "Leakage_factor.xlsx": dict(label="Leakage", always_on=False, cols=_LEAKAGE_XLSX),
    "leakage_fault(0.5).csv": dict(label="Leakage", always_on=False, cols=_PUMP_NAMED),
    "leakage_fault(1.0).csv": dict(label="Leakage", always_on=False, cols=_PUMP_NAMED),
    # ---- Flexible-shaft fault (2 runs) -------------------------------------
    "MildFlexible_shaft.xlsx": dict(label="FlexibleShaft", always_on=False, cols=_FLEX_MILD),
    "Medium_FlexibleShaft_Fault.xlsx": dict(label="FlexibleShaft", always_on=False, cols=_FLEX_MEDIUM),
    # ---- Generator fault (3 runs: 1 always-on + 2 new timed @0.1s) ----------
    "simplified_generator_fault.xlsx": dict(label="GeneratorFault", always_on=True, cols=_GEN_OLD),
    "simplifiied-generator-fault.xlsx": dict(label="GeneratorFault", always_on=False, cols=_PUMP_NAMED),
    "simplifiied-generator-fault(st-0.5).xlsx": dict(label="GeneratorFault", always_on=False, cols=_PUMP_NAMED),
}

# Files deliberately excluded, with evidence-based reasons (reported by stage 02).
EXCLUDED = {
    "healthy data 1.xlsx": "only 0-0.002 s captured (startup only); non-physical "
                           "overshoot (speed 3.5e4 rad/s, p 113 bar). No steady healthy region.",
    "Healthy data 2.xlsx": "identical to healthy data 1: 0-0.002 s startup garbage only.",
    "X_Sevear_FlexibleShaft_Fault.xlsx": "span 0-0.005 s < fault time 0.1 s: NO "
                                         "fault-active region exists.",
    "X_Critical_FlexibleShaft_Fault.xlsx": "span 0-0.125 s -> fault region only "
                                           "~0.025 s < window+stride needed for "
                                           f"{MIN_WINDOWS} windows.",
    "Pump_Displacement.xlsx": "byte-identical duplicate of disp1_fault(0.5) "
                              "(max signal diff 0.0) -> dropped to avoid hidden "
                              "train/test leakage between identical runs.",
    "simplifiied-generator-fault2(st-0.5).xlsx": "near-identical re-export of "
                              "simplifiied-generator-fault(st-0.5) (max signal diff "
                              "0.002 on signals up to 3.8e6) -> dropped to avoid "
                              "train/test leakage between the same simulation.",
}

# Modelling basis = canonical signals present in EVERY included run. Computed
# dynamically by stage 02 and saved (stage 01 asserts it matches this constant).
# Pressure is present because LEAKAGE_PRESSURE_COL maps the leakage pressure
# channel; set that to None and shrink this to {current, vdc, vac} to revert.
COMMON = ["pressure", "current", "vdc", "vac"]


# --------------------------------------------------------------------------- #
# Stage 1 -- load + harmonize a single raw file into a canonical frame
# --------------------------------------------------------------------------- #
def read_run(path: Path, spec: dict) -> pd.DataFrame:
    """Read csv/xlsx, keep only `time` + the mapped raw signal columns.

    Column names are matched via _resolve (suffix/whitespace tolerant), so the
    duplicate `time`/`time.1`/`Constant`/scope columns are ignored and a scope
    that lost/gained a ":1" suffix still resolves. First column is the time base.
    """
    path = Path(path)
    wanted = list(spec["cols"].values())
    if path.suffix.lower() == ".csv":
        header = pd.read_csv(path, nrows=0).columns
        keep = [header[0] if "time" not in header else "time"]
        for w in wanted:
            a = _resolve(w, header)
            if a is not None and a not in keep:
                keep.append(a)
        df = pd.read_csv(path, usecols=keep, low_memory=False)
    else:
        df = pd.read_excel(path)
    if "time" not in df.columns:                 # first column is the time base
        df = df.rename(columns={df.columns[0]: "time"})
    keep, seen = ["time"], {"time"}
    for w in wanted:
        a = _resolve(w, df.columns)
        if a is not None and a not in seen:
            keep.append(a); seen.add(a)
    return df.loc[:, keep]


def harmonize(df: pd.DataFrame, spec: dict) -> pd.DataFrame:
    """Rename raw -> canonical, unit-convert, return tidy [time, <canon...>]."""
    out = pd.DataFrame({"time": pd.to_numeric(df["time"], errors="coerce")})
    for canon, raw in spec["cols"].items():
        actual = _resolve(raw, df.columns)
        if actual is not None:
            out[canon] = UNIT[canon](pd.to_numeric(df[actual], errors="coerce"))
    out = out.dropna(subset=["time"]).sort_values("time").reset_index(drop=True)
    return out


def dedup_run(df: pd.DataFrame) -> tuple[pd.DataFrame, int, int]:
    """Drop exact duplicate rows, then duplicate timestamps (keep first).
    Returns (deduped_df, n_duplicate_rows, n_duplicate_timestamps) for reporting.
    """
    n0 = len(df)
    df = df.drop_duplicates()
    n_dup_rows = n0 - len(df)
    n1 = len(df)
    df = df.drop_duplicates(subset="time", keep="first")
    n_dup_times = n1 - len(df)
    return df.reset_index(drop=True), n_dup_rows, n_dup_times


def sanitize_invalid(df: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """Flag physically-impossible values (negative pressure/flow/fuel -- zero is a
    physics fact, not a tuned threshold) as missing. Isolated bad points are then
    filled by the SAME interpolation resample_uniform already uses for missing
    data, so no separate imputation logic is needed. Returns (df, report) with a
    per-signal count of flagged values for the caller to print."""
    out = df.copy()
    report = {}
    for s in NONNEGATIVE_SIGNALS:
        if s not in out.columns:
            continue
        bad = out[s] < 0
        report[s] = int(bad.sum())
        out.loc[bad, s] = np.nan
    return out, report


def drop_constant_columns(df: pd.DataFrame, signal_cols: list[str]) -> tuple[pd.DataFrame, list[str]]:
    """Drop signal columns that are constant across the ENTIRE dataset (all runs
    combined). A column that is constant within a single run only (e.g. an
    always-on flag) is normal and must NOT be dropped for that reason alone."""
    const_cols = [c for c in signal_cols
                  if c in df.columns and df[c].nunique(dropna=True) <= 1]
    return df.drop(columns=const_cols), const_cols


def low_variance_filter(feat: pd.DataFrame, feat_cols: list[str],
                        thresh: float = LOW_VAR_THRESH) -> tuple[list[str], list[str]]:
    """Split feat_cols into (kept, dropped) by variance across ALL windows.
    A near-constant feature (variance < thresh) cannot separate classes."""
    var = feat[feat_cols].var()
    dropped = var[var < thresh].index.tolist()
    kept = [c for c in feat_cols if c not in dropped]
    return kept, dropped


# --------------------------------------------------------------------------- #
# Stage 2 -- resample to a uniform grid + robust outlier clip
# --------------------------------------------------------------------------- #
def resample_uniform(df: pd.DataFrame, fs: int = FS) -> pd.DataFrame:
    """Linear-interpolate every canonical signal onto a uniform grid that starts
    at BASELINE_WIN[0] (0.05 s) -- this discards the solver init transient by
    construction. Returns a frame with a uniform `time` column.
    """
    t = df["time"].to_numpy()
    t0 = max(BASELINE_WIN[0], float(t.min()))
    t1 = float(t.max())
    if t1 - t0 < WINDOW_SEC:                      # nothing usable
        return pd.DataFrame(columns=df.columns)
    grid = np.arange(t0, t1, 1.0 / fs)
    out = {"time": grid}
    sig_cols = [c for c in df.columns if c != "time"]
    for c in sig_cols:
        v = df[c].to_numpy()
        m = np.isfinite(t) & np.isfinite(v)
        out[c] = np.interp(grid, t[m], v[m]) if m.sum() >= 2 else np.full_like(grid, np.nan)
    return pd.DataFrame(out)


def hampel_clip(x: np.ndarray, win: int = 11, n_sig: float = 3.0) -> np.ndarray:
    """Hampel filter: replace points >n_sig robust-sigma from the rolling median.
    Removes solver spikes without distorting genuine dynamics (median-based)."""
    s = pd.Series(x)
    med = s.rolling(win, center=True, min_periods=1).median()
    mad = (s - med).abs().rolling(win, center=True, min_periods=1).median()
    sigma = 1.4826 * mad                          # MAD -> std for normal data
    diff = (s - med).abs()
    out = s.where(diff <= n_sig * sigma.replace(0, np.nan).fillna(np.inf), med)
    return out.to_numpy()


def clean_run(df_uniform: pd.DataFrame) -> pd.DataFrame:
    """Outlier-clip every signal; drop constant/all-NaN signal columns."""
    if df_uniform.empty:
        return df_uniform
    out = df_uniform.copy()
    for c in [c for c in out.columns if c != "time"]:
        v = out[c].to_numpy()
        if not np.isfinite(v).any() or np.nanstd(v) == 0:
            continue                              # leave constant signals as-is
        out[c] = hampel_clip(v)
    return out


# --------------------------------------------------------------------------- #
# Stage 3/4 -- baseline reference, windowing, feature extraction
# --------------------------------------------------------------------------- #
def baseline_stats(df_clean: pd.DataFrame, signals: list[str]) -> dict:
    """Per-signal reference stats from the pre-fault window [0.05, 0.10).
    `scale` is a robust spread (IQR -> std -> |mean| -> 1) used to make
    deviation features dimensionless and division-safe.
    """
    t = df_clean["time"].to_numpy()
    m = (t >= BASELINE_WIN[0]) & (t < BASELINE_WIN[1])
    base = {}
    for s in signals:
        v = df_clean[s].to_numpy()[m]
        v = v[np.isfinite(v)]
        if v.size == 0:
            base[s] = dict(mean=0.0, std=0.0, rms=0.0, scale=1.0)
            continue
        iqr = np.subtract(*np.percentile(v, [75, 25]))
        scale = iqr if iqr > EPS else (v.std() if v.std() > EPS else max(abs(v.mean()), 1.0))
        base[s] = dict(mean=float(v.mean()), std=float(v.std()),
                       rms=float(np.sqrt(np.mean(v**2))), scale=float(scale))
    return base


def iter_windows(df_clean: pd.DataFrame, always_on: bool):
    """Yield (t_start, dict[signal->array]) for each complete window in the
    fault-active region. Timed faults -> t >= FAULT_T; always-on -> whole run
    (from 0.05 s). Healthy runs use the same t >= FAULT_T region so Healthy and
    fault windows occupy the SAME time region (no time confound)."""
    t = df_clean["time"].to_numpy()
    lo = BASELINE_WIN[0] if always_on else FAULT_T
    sig_cols = [c for c in df_clean.columns if c != "time"]
    region = np.where(t >= lo)[0]
    if region.size == 0:
        return
    i0, i1 = region[0], region[-1] + 1
    wlen = int(WINDOW_SEC * FS)
    step = max(1, int(wlen * STRIDE_FRAC))
    for start in range(i0, i1 - wlen + 1, step):
        sl = slice(start, start + wlen)
        yield float(t[start]), {c: df_clean[c].to_numpy()[sl] for c in sig_cols}


def _spectral(x: np.ndarray, fs: int = FS):
    """Coarse spectral shape (centroid Hz, low-band energy fraction). Window is
    short so this is a shape descriptor, not fine spectroscopy (issue #4)."""
    x = x - np.mean(x)
    f, p = spsignal.periodogram(x, fs=fs)
    tot = p.sum()
    if tot <= EPS:
        return 0.0, 0.0
    centroid = float((f * p).sum() / tot)
    low = float(p[f <= f.max() / 4].sum() / tot)
    return centroid, low


def window_features(seg: dict, base: dict, signals: list[str]) -> dict:
    """Scale-invariant + baseline-deviation features for one window.

    Per signal (invariant): cov, crest, ripple, skew, kurtosis, zero-cross rate,
    normalized slope, spectral centroid, low-band fraction.
    Per signal (deviation vs this run's pre-fault baseline): d_mean, d_std, d_rms.
    Cross-signal (invariant): pairwise correlation + CoV of the V_dc*I power proxy.
    All dimensionless -> operating-point invariant.
    """
    f = {}
    for s in signals:
        x = np.asarray(seg[s], float)
        x = x[np.isfinite(x)]
        if x.size < 4:
            continue
        mean, std = x.mean(), x.std()
        rms = np.sqrt(np.mean(x**2))
        amean = abs(mean) + EPS
        f[f"{s}_cov"] = std / amean
        f[f"{s}_crest"] = np.max(np.abs(x)) / (rms + EPS)
        f[f"{s}_ripple"] = (x.max() - x.min()) / amean
        f[f"{s}_skew"] = float(stats.skew(x)) if std > EPS else 0.0
        f[f"{s}_kurt"] = float(stats.kurtosis(x)) if std > EPS else 0.0
        f[f"{s}_zcr"] = np.mean(np.abs(np.diff(np.sign(x - mean)))) / 2.0
        slope = np.polyfit(np.arange(x.size), x, 1)[0]
        f[f"{s}_slope"] = slope * x.size / amean              # relative change/window
        cen, low = _spectral(x)
        f[f"{s}_speccen"] = cen
        f[f"{s}_speclow"] = low
        b = base[s]
        f[f"{s}_dmean"] = (mean - b["mean"]) / b["scale"]
        f[f"{s}_dstd"] = (std - b["std"]) / (b["scale"] + EPS)
        f[f"{s}_drms"] = (rms - b["rms"]) / b["scale"]
    # cross-signal invariants on the common electrical/pressure set
    def corr(a, c):
        xa, xc = np.asarray(seg[a], float), np.asarray(seg[c], float)
        n = min(xa.size, xc.size)
        if n < 4 or xa.std() < EPS or xc.std() < EPS:
            return 0.0
        return float(np.corrcoef(xa[:n], xc[:n])[0, 1])
    if {"vdc", "current"} <= set(signals):
        f["corr_vdc_i"] = corr("vdc", "current")
        p = np.asarray(seg["vdc"], float) * np.asarray(seg["current"], float)
        f["power_cov"] = p.std() / (abs(p.mean()) + EPS)
    if {"pressure", "vac"} <= set(signals):
        f["corr_p_vac"] = corr("pressure", "vac")
    return f


def build_run_features(df_clean: pd.DataFrame, label: str, run_id: str,
                       always_on: bool, signals: list[str]) -> pd.DataFrame:
    """All window feature rows for one run (empty if < MIN_WINDOWS)."""
    if df_clean.empty:
        return pd.DataFrame()
    base = baseline_stats(df_clean, signals)
    rows = []
    for t_start, seg in iter_windows(df_clean, always_on):
        feats = window_features(seg, base, signals)
        feats.update(run_id=run_id, label=label, t_start=t_start)
        rows.append(feats)
    if len(rows) < MIN_WINDOWS:
        return pd.DataFrame()
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------- #
# One-call pipeline used by BOTH training stages and the Streamlit app
# --------------------------------------------------------------------------- #
def process_file(path: Path, spec: dict, run_id: str, signals: list[str]) -> pd.DataFrame:
    """raw file -> clean uniform run -> windowed feature rows (one code path)."""
    h = harmonize(read_run(path, spec), spec)
    h, _, _ = dedup_run(h)
    h, _ = sanitize_invalid(h)
    clean = clean_run(resample_uniform(h))
    return build_run_features(clean, spec["label"], run_id, spec["always_on"], signals)


def process_uploaded(df_raw: pd.DataFrame, signals: list[str],
                     fname: str | None = None) -> pd.DataFrame:
    """Streamlit path: infer the column scheme of an uploaded file, harmonize,
    and extract features for the fault-active region (label unknown).

    If the uploaded filename matches a FILE_MAP entry, its exact column scheme
    is used (same as training — covers exports whose channels live in generic
    `PS-Simulink Converter` columns that inference cannot guess). Otherwise we
    pick the known scheme whose raw columns are present, falling back to a
    best-effort match on the common signals.
    """
    if fname is not None and fname in FILE_MAP:
        m = FILE_MAP[fname]
        spec = dict(label="unknown", always_on=m["always_on"], cols=m["cols"])
    else:
        spec = infer_spec(df_raw)
    h = harmonize(df_raw.rename(columns={df_raw.columns[0]: "time"}), spec)
    h, _, _ = dedup_run(h)
    h, _ = sanitize_invalid(h)
    clean = clean_run(resample_uniform(h))
    return build_run_features(clean, "unknown", "upload", spec["always_on"], signals)


def infer_spec(df_raw: pd.DataFrame) -> dict:
    """Choose the FILE_MAP column scheme best matching an uploaded frame.

    Name hits alone cannot separate the disp and leakage schemes (identical
    headers, but leakage stores pressure in the mislabelled Mass_Flow_Rate
    column). Tie-break with the pressure fingerprint: a real Simulink pressure
    channel sits at/above atmospheric (~1.013e5 Pa), the mislabelled column is
    ~0.09. An implausible pressure mapping loses its hit and is stripped, so an
    upload never gets garbage pressure features.
    """
    def _pressure_ok(spec) -> bool:
        raw = spec["cols"].get("pressure")
        actual = _resolve(raw, df_raw.columns) if raw else None
        if actual is None:
            return False
        v = pd.to_numeric(df_raw[actual], errors="coerce")
        return bool(np.isfinite(v).any() and np.nanmedian(v) > 1e4)   # Pa

    best, best_hits, best_p_ok = None, -1, False
    for spec in FILE_MAP.values():
        hits = sum(_resolve(c, df_raw.columns) is not None
                   for c in spec["cols"].values())
        p_ok = _pressure_ok(spec)
        if "pressure" in spec["cols"] and not p_ok:
            hits -= 1                       # implausible pressure = not a hit
        if hits > best_hits:
            best, best_hits, best_p_ok = spec, hits, p_ok
    cols = dict(best["cols"])
    if "pressure" in cols and not best_p_ok:
        del cols["pressure"]                # never serve garbage pressure
    return dict(label="unknown", always_on=best["always_on"], cols=cols)


# --------------------------------------------------------------------------- #
# Model factory -- shared by stage 05 (compare), 06 (final), and the app, so the
# selected pipeline is defined in exactly ONE place. Selection is a pipeline step
# (fitted per CV fold). Trees: no scaler. SVM: StandardScaler first.
# --------------------------------------------------------------------------- #
from sklearn.base import BaseEstimator, ClassifierMixin


class _XGBSafe(ClassifierMixin, BaseEstimator):
    """Thin XGBoost wrapper that label-encodes y internally, so it tolerates the
    non-contiguous / missing class subsets that GroupKFold produces when a class
    has only one run. Keeps the XGBoost model (your preference) usable here."""
    def __init__(self, **kw):
        self.kw = dict(kw)
    def get_params(self, deep=True):
        return dict(self.kw)
    def set_params(self, **p):
        self.kw.update(p); return self
    def fit(self, X, y):
        from xgboost import XGBClassifier
        from sklearn.preprocessing import LabelEncoder
        self.le_ = LabelEncoder()
        yt = self.le_.fit_transform(y)
        self.classes_ = self.le_.classes_
        self.model_ = XGBClassifier(**self.kw).fit(X, yt)
        return self
    def predict(self, X):
        return self.le_.inverse_transform(self.model_.predict(X))
    def predict_proba(self, X):
        return self.model_.predict_proba(X)


def build_feature_models(k: int):
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler
    from sklearn.feature_selection import SelectKBest, mutual_info_classif
    from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
    from sklearn.svm import SVC

    sel = lambda: SelectKBest(mutual_info_classif, k=k)
    models = {
        "RandomForest": Pipeline([("sel", sel()),
            ("clf", RandomForestClassifier(n_estimators=300, class_weight="balanced",
                                           random_state=0))]),
        "ExtraTrees": Pipeline([("sel", sel()),
            ("clf", ExtraTreesClassifier(n_estimators=300, class_weight="balanced",
                                         random_state=0))]),
        "SVM_RBF": Pipeline([("scaler", StandardScaler()), ("sel", sel()),
            ("clf", SVC(C=10, gamma="scale", class_weight="balanced",
                        probability=True, random_state=0))]),
    }
    try:
        import xgboost  # noqa: F401
        models["XGBoost"] = Pipeline([("sel", sel()),
            ("clf", _XGBSafe(n_estimators=300, max_depth=4, learning_rate=0.1,
                             subsample=0.9, colsample_bytree=0.8, random_state=0,
                             tree_method="hist", eval_metric="mlogloss"))])
    except Exception:
        from sklearn.ensemble import HistGradientBoostingClassifier
        models["HistGB"] = Pipeline([("sel", sel()),
            ("clf", HistGradientBoostingClassifier(max_depth=4, random_state=0))])
    return models


if __name__ == "__main__":   # tiny self-check (ponytail: one runnable check)
    # synthetic run: invariant + deviation features must be finite & operating
    # -point invariant (scaling the whole signal must not change cov/deviation).
    n = int(STOP_T * FS)
    t = np.linspace(0, STOP_T, n)
    base = 5 + 0.5 * np.sin(2 * np.pi * 50 * t)
    step = np.where(t >= FAULT_T, 3.0, 0.0)            # a "fault" jump
    df = pd.DataFrame({"time": t, "pressure": base + step, "current": base,
                       "vdc": base, "vac": base})
    f1 = build_run_features(clean_run(resample_uniform(df)), "X", "r1", False, COMMON)
    df2 = df.copy()
    for c in ["pressure", "current", "vdc", "vac"]:
        df2[c] *= 1000.0                              # 1000x operating scale
    f2 = build_run_features(clean_run(resample_uniform(df2)), "X", "r2", False, COMMON)
    assert not f1.empty and len(f1) >= MIN_WINDOWS, "no windows produced"
    inv = f"{COMMON[0]}_cov"
    rel = abs(f1[inv].mean() - f2[inv].mean()) / (abs(f1[inv].mean()) + EPS)
    assert rel < 1e-6, f"feature {inv} not scale-invariant ({rel})"
    assert np.isfinite(f1.select_dtypes(float).to_numpy()).all(), "non-finite features"
    print(f"self-check OK: {len(f1)} windows, {f1.shape[1]-3} features, "
          f"{inv} scale-invariant (rel diff {rel:.1e})")
