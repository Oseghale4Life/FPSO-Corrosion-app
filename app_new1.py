"""
========================================================================
FPSO CORROSION LIFE EXTENSION ASSESSMENT APP
========================================================================
Project : Machine Learning-Based Prediction of Corrosion Progression
          for Life Extension Assessment of Offshore Hull Structures
Author  : Oseghale Aimuelimiobor Paschal
Department : Mechatronics Engineering
Project Supervisor : Dr. Celestine Ebieto
Dataset : Paik et al. (2004), Corrosion Science,
          1,937 seawater ballast tank measurements
========================================================================
"""

import html
import streamlit as st
import numpy as np
import pandas as pd
import altair as alt
import joblib
import os

# ── Page configuration ────────────────────────────────────────────
st.set_page_config(
    page_title="FPSO Corrosion Assessment",
    page_icon="🛳️",
    layout="wide"
)

# ── Constants ─────────────────────────────────────────────────────
LIMIT_ABS    = 1.0    # ABS NDCV average — Paik (2004) Eq 5a
LIMIT_SEVERE = 3.5    # ABS severe bound — Paik (2004) Eq 5c
HORIZON      = 27.0   # max age in Paik training data

# Paik et al. (2004) Table 2 — one C1 per coating life Tc.
#   average : Eq. 3a, 3b, 3c   (all corrosion data)
#   severe  : Eq. 4a, 4b, 4c   (95 % and above band)
C1_AVG_TABLE = {5.0: 0.0466, 7.5: 0.0579, 10.0: 0.0823}
C1_SEVERE_TABLE = {5.0: 0.1469, 7.5: 0.1938, 10.0: 0.2894}
EQ_AVG    = {5.0: "3a", 7.5: "3b", 10.0: "3c"}
EQ_SEVERE = {5.0: "4a", 7.5: "4b", 10.0: "4c"}


def c1_avg(coating_life):
    return C1_AVG_TABLE[float(coating_life)]


def c1_severe(coating_life):
    return C1_SEVERE_TABLE[float(coating_life)]

FEATURE_COLUMNS = [
    "age_midpoint_yr",
    "log_age",
    "effective_corr_age_yr",
    "effect_corr_age_Tc5yrs",
    "effect_corr_age_Tc75yrs",
    "effect_corr_age_Tc10yrs",
]

# ── Project information (shown in the app) ────────────────────────
PROJECT_TITLE = (
    "Machine Learning-Based Prediction of Corrosion Progression "
    "for Life Extension Assessment of Offshore Hull Structures"
)
AUTHOR       = "Oseghale Aimuelimiobor Paschal"
DEPARTMENT   = "Mechatronics Engineering"
SUPERVISOR   = "Dr. Celestine Ebieto"
DATASET_INFO = (
    "Paik et al. (2004), Corrosion Science — "
    "1,937 seawater ballast tank measurements"
)

# ── Load model ────────────────────────────────────────────────────
@st.cache_resource
def load_model():
    return joblib.load("best_reg_model.pkl")

model = load_model()

# ══════════════════════════════════════════════════════════════════
# BACKEND FUNCTIONS  (unchanged)
# ══════════════════════════════════════════════════════════════════

def build_feature_row(age, coating_life):
    row = {
        "age_midpoint_yr":         float(age),
        "log_age":                 np.log(float(age)),
        "effective_corr_age_yr":   max(0.0, age - coating_life),
        "effect_corr_age_Tc5yrs":  max(0.0, age - 5.0),
        "effect_corr_age_Tc75yrs": max(0.0, age - 7.5),
        "effect_corr_age_Tc10yrs": max(0.0, age - 10.0),
    }
    return pd.DataFrame([row])[FEATURE_COLUMNS]


def years_to_limit(model, current_age, coating_life,
                   threshold, offset=0.0, step=0.25):
    age = float(current_age)
    while age <= HORIZON:
        X    = build_feature_row(age, coating_life)
        pred = model.predict(X)[0] + offset
        if pred >= threshold:
            return round(age - current_age, 2), round(age, 2)
        age = round(age + step, 4)
    return None, None


def ml_assessment(current_age, coating_life, measured_wastage=None):
    X_now         = build_feature_row(current_age, coating_life)
    predicted_now = float(model.predict(X_now)[0])

    if measured_wastage is not None:
        offset         = float(measured_wastage) - predicted_now
        wastage_to_use = float(measured_wastage)
    else:
        offset         = 0.0
        wastage_to_use = predicted_now

    yrs_abs,    age_abs    = years_to_limit(
        model, current_age, coating_life,
        LIMIT_ABS, offset=offset
    )
    yrs_severe, age_severe = years_to_limit(
        model, current_age, coating_life,
        LIMIT_SEVERE, offset=offset
    )

    return {
        "predicted_mm":    round(predicted_now, 4),
        "wastage_used_mm": round(wastage_to_use, 4),
        "offset_mm":       round(offset, 4),
        "yrs_to_abs":      yrs_abs,
        "age_at_abs":      age_abs,
        "yrs_to_severe":   yrs_severe,
        "age_at_severe":   age_severe,
    }


def paik_assessment(current_age, coating_life):
    Te           = max(0.0, current_age - coating_life)
    C1_AVG       = c1_avg(coating_life)
    C1_SEVERE    = c1_severe(coating_life)
    paik_avg_now = round(C1_AVG    * Te, 4)
    paik_sev_now = round(C1_SEVERE * Te, 4)

    def paik_years_to(C1, threshold):
        age = float(current_age)
        while age <= HORIZON:
            Te_f = max(0.0, age - coating_life)
            if C1 * Te_f >= threshold:
                return round(age - current_age, 2), round(age, 2)
            age = round(age + 0.25, 4)
        return None, None

    avg_yrs_abs,    avg_age_abs    = paik_years_to(C1_AVG, LIMIT_ABS)
    avg_yrs_severe, avg_age_severe = paik_years_to(C1_AVG, LIMIT_SEVERE)

    return {
        "paik_avg_mm":         paik_avg_now,
        "paik_severe_mm":      paik_sev_now,
        "paik_yrs_to_abs":     avg_yrs_abs,
        "paik_age_at_abs":     avg_age_abs,
        "paik_yrs_to_severe":  avg_yrs_severe,
        "paik_age_at_severe":  avg_age_severe,
    }


def get_status(wastage):
    if wastage >= LIMIT_SEVERE:
        return "🔴 SEVERE", "Exceeds 3.5mm limit. Steel renewal required immediately."
    elif wastage >= LIMIT_ABS:
        return "🟠 EXCEEDED", "Above 1.0mm ABS limit. Maintenance action required."
    elif wastage >= 0.75:
        return "🟡 APPROACHING", "Within 25% of 1.0mm limit. Increase survey frequency."
    else:
        return "🟢 SAFE", "Within acceptable corrosion limits."


def format_years(yrs, breach_age):
    if yrs is None:
        return f"Beyond {HORIZON}yr range"
    return f"{yrs} yrs  (vessel age {breach_age}yr)"


# ══════════════════════════════════════════════════════════════════
# STYLE  (works in both light and dark theme, nothing hard-coded
#         for background or text colour)
# ══════════════════════════════════════════════════════════════════

CSS = """
<style>
:root{
    --ok:#2f9e63; --approach:#d9a21d; --exceeded:#e0622f; --severe:#c2303f;
    --accent:#3b82f6;
    --line:rgba(128,128,128,.30); --soft:rgba(128,128,128,.07);
}
.block-container{padding-top:2.6rem; max-width:1240px;}

.eyebrow{font-size:.72rem; letter-spacing:.13em; text-transform:uppercase; opacity:.6;}
.hero-title{font-size:2.05rem; font-weight:700; line-height:1.15; margin:.2rem 0 0 0;}
.hero-sub{opacity:.72; margin:.4rem 0 1rem 0;}

.meta-card{border:1px solid var(--line); border-radius:10px; padding:.95rem 1.15rem; background:var(--soft);}
.meta-project{font-size:.93rem; line-height:1.45;}
.meta-grid{display:grid; grid-template-columns:repeat(auto-fit,minmax(190px,1fr));
           gap:.7rem 1.5rem; margin-top:.8rem; padding-top:.8rem; border-top:1px solid var(--line);}
.meta-label,.tile-label,.sec,.side-label{font-size:.68rem; letter-spacing:.11em;
           text-transform:uppercase; font-weight:600;}
.meta-label{opacity:.55;}
.meta-value{font-size:.9rem; font-weight:600; margin-top:.1rem;}

.sec{opacity:.6; margin:1.5rem 0 .55rem 0;}
.side-label{opacity:.6; margin:1.1rem 0 .3rem 0;}

.tile{border:1px solid var(--line); border-radius:10px; padding:.8rem 1rem; min-height:106px;}
.tile-label{opacity:.6;}
.tile-value{font-size:1.7rem; font-weight:700; line-height:1.2; margin:.25rem 0 .1rem 0;
            font-variant-numeric:tabular-nums;}
.tile-unit{font-size:.85rem; font-weight:500; opacity:.6; margin-left:.25rem;}
.tile-sub{font-size:.78rem; opacity:.62; line-height:1.35;}

.status{border:1px solid var(--line); border-left:6px solid var(--c); border-radius:10px;
        padding:1rem 1.25rem 1.1rem 1.25rem; background:var(--bg); margin:.4rem 0 1rem 0;}
.status.ok{--c:var(--ok); --bg:rgba(47,158,99,.10);}
.status.approach{--c:var(--approach); --bg:rgba(217,162,29,.11);}
.status.exceeded{--c:var(--exceeded); --bg:rgba(224,98,47,.11);}
.status.severe{--c:var(--severe); --bg:rgba(194,48,63,.12);}
.status-head{font-size:1.35rem; font-weight:700;}
.status-msg{opacity:.9; margin-top:.1rem;}
.status-meta{font-size:.78rem; opacity:.6; margin-top:.35rem;}

.gauge{position:relative; height:10px; border-radius:6px; margin:2.4rem .6rem 1.8rem .6rem;}
.g-marker{position:absolute; top:-7px; width:5px; height:24px; border-radius:3px;
          background:#fff; box-shadow:0 0 0 2px #1f2937; transform:translateX(-50%);}
.g-bubble{position:absolute; top:-34px; transform:translateX(-50%); font-size:.76rem; font-weight:700;
          padding:.05rem .5rem; border-radius:4px; background:#1f2937; color:#fff; white-space:nowrap;}
.g-tick{position:absolute; top:15px; font-size:.7rem; opacity:.65; white-space:nowrap;}

.tbl-wrap{overflow-x:auto;}
table.grid{width:100%; border-collapse:collapse; font-size:.9rem;}
table.grid th{text-align:left; font-size:.68rem; letter-spacing:.09em; text-transform:uppercase;
              font-weight:600; opacity:.65; padding:.55rem .75rem; border-bottom:2px solid var(--line);}
table.grid td{padding:.65rem .75rem; border-bottom:1px solid var(--line);
              font-variant-numeric:tabular-nums; vertical-align:top;}
table.grid tr.now td{background:var(--soft); font-weight:600;}
.dim{display:block; font-size:.75rem; opacity:.6; font-weight:400;}
.pos{color:var(--ok); font-weight:600;}
.neg{color:#d64545; font-weight:600;}
.flat{opacity:.6;}

.note{border-left:3px solid var(--accent); background:var(--soft); border-radius:0 8px 8px 0;
      padding:.6rem .95rem; font-size:.9rem; margin:.9rem 0 .2rem 0;}
.note.warn{border-left-color:var(--approach);}
.note.bad{border-left-color:var(--severe);}

.step{display:flex; gap:.85rem; align-items:flex-start; margin:.7rem 0;}
.step-n{flex:0 0 1.7rem; height:1.7rem; border-radius:50%; border:1.5px solid var(--accent);
        color:var(--accent); display:flex; align-items:center; justify-content:center;
        font-weight:700; font-size:.85rem;}
.step-t{padding-top:.1rem; line-height:1.5;}
.mono{font-family:ui-monospace,SFMono-Regular,Menlo,Consolas,monospace; font-size:.85rem; line-height:1.9;}
</style>
"""


# ══════════════════════════════════════════════════════════════════
# DISPLAY HELPERS  (visual only)
# ══════════════════════════════════════════════════════════════════

STATUS_CLASS = {"🟢": "ok", "🟡": "approach", "🟠": "exceeded", "🔴": "severe"}


def md(markup):
    """Render HTML on one line so markdown never mistakes it for a code block."""
    flat = " ".join(line.strip() for line in markup.splitlines() if line.strip())
    st.markdown(flat, unsafe_allow_html=True)


def sec(text):
    md(f'<div class="sec">{text}</div>')


def note(text, kind=""):
    md(f'<div class="note {kind}">{text}</div>')


def tile(label, value, unit="", sub=""):
    u = f'<span class="tile-unit">{unit}</span>' if unit else ""
    s = f'<div class="tile-sub">{sub}</div>' if sub else ""
    return (f'<div class="tile"><div class="tile-label">{label}</div>'
            f'<div class="tile-value">{value}{u}</div>{s}</div>')


def meta(label, value):
    return (f'<div><div class="meta-label">{label}</div>'
            f'<div class="meta-value">{html.escape(value)}</div></div>')


def html_table(headers, rows, highlight_first=False):
    head = "".join(f"<th>{h}</th>" for h in headers)
    body = ""
    for i, r in enumerate(rows):
        cls = ' class="now"' if (highlight_first and i == 0) else ""
        body += f"<tr{cls}>" + "".join(f"<td>{c}</td>" for c in r) + "</tr>"
    return (f'<div class="tbl-wrap"><table class="grid"><thead><tr>{head}</tr></thead>'
            f'<tbody>{body}</tbody></table></div>')


def fmt_diff(ml_yrs, paik_yrs):
    if ml_yrs is None or paik_yrs is None:
        return '<span class="flat">—</span>'
    d = round(ml_yrs - paik_yrs, 2)
    cls = "pos" if d > 0 else "neg" if d < 0 else "flat"
    return f'<span class="{cls}">{d:+.2f} yr</span>'


def gauge_html(w):
    """Bar from 0 to 3.5 mm with the 1.0 mm limit marked and the plate on it."""
    pos    = min(max(w / LIMIT_SEVERE, 0.0), 1.0) * 100
    bubble = min(max(pos, 6.0), 94.0)
    z1     = 0.75 / LIMIT_SEVERE * 100
    z2     = LIMIT_ABS / LIMIT_SEVERE * 100
    bg = (f"linear-gradient(to right,#2f9e63 0,#2f9e63 {z1:.2f}%,"
          f"#e0a21d {z1:.2f}%,#e0a21d {z2:.2f}%,#e0622f {z2:.2f}%,#b3263a 100%)")
    return (
        f'<div class="gauge" style="background:{bg};">'
        f'<div class="g-bubble" style="left:{bubble:.2f}%;">{w:.3f} mm</div>'
        f'<div class="g-marker" style="left:{pos:.2f}%;"></div>'
        f'<div class="g-tick" style="left:0;">0</div>'
        f'<div class="g-tick" style="left:{z2:.2f}%;transform:translateX(-50%);">1.0 mm  ABS average</div>'
        f'<div class="g-tick" style="left:100%;transform:translateX(-100%);">3.5 mm  severe</div>'
        f'</div>'
    )


def ml_curve(ages, coating_life, offset):
    X = pd.concat([build_feature_row(a, coating_life) for a in ages],
                  ignore_index=True)
    return model.predict(X) + offset


def projection_table(current_age, coating_life, offset, plate_mm, step):
    """Wastage from the current age to the end of the validated range."""
    ages = [round(float(a), 2)
            for a in np.arange(current_age, HORIZON + 0.001, step)]
    if ages[-1] < HORIZON - 1e-9:
        ages.append(HORIZON)
    ages = np.array(ages)
    ml   = ml_curve(ages, coating_life, offset)
    te   = np.maximum(0.0, ages - coating_life)
    df = pd.DataFrame({
        "Vessel age (yr)":   ages,
        "ML model (mm)":     ml,
        "Paik average (mm)": c1_avg(coating_life) * te,
        "Paik severe (mm)":  c1_severe(coating_life) * te,
        "Status":            [get_status(v)[0] for v in ml],
    })
    if plate_mm is not None:
        df["Thickness left (mm)"] = np.maximum(plate_mm - ml, 0.0)
    return df


def chart_frame(coating_life, offset):
    """Full 11-27 yr curves for the chart."""
    ages = np.arange(11.0, HORIZON + 0.001, 0.5)
    te   = np.maximum(0.0, ages - coating_life)
    return pd.DataFrame({
        "Age":          ages,
        "ML model":     ml_curve(ages, coating_life, offset),
        "Paik average": c1_avg(coating_life) * te,
        "Paik severe":  c1_severe(coating_life) * te,
    })


def projection_chart(df, current_age, w_now):
    names  = ["ML model", "Paik average", "Paik severe"]
    colors = ["#3b82f6", "#9aa3af", "#f59e0b"]
    long   = df.melt("Age", value_vars=names,
                     var_name="Series", value_name="Wastage")

    x = alt.X("Age:Q", title="Vessel age (years)",
              scale=alt.Scale(domain=[11, 27], nice=False),
              axis=alt.Axis(tickMinStep=1))
    y = alt.Y("Wastage:Q", title="Wastage (mm)",
              scale=alt.Scale(domain=[0, 4]))

    lines = alt.Chart(long).mark_line(clip=True, strokeWidth=2.4).encode(
        x=x, y=y,
        color=alt.Color("Series:N",
                        scale=alt.Scale(domain=names, range=colors),
                        legend=alt.Legend(orient="bottom", title=None)),
        strokeDash=alt.StrokeDash(
            "Series:N",
            scale=alt.Scale(domain=names, range=[[1, 0], [6, 4], [6, 4]]),
            legend=None),
        tooltip=[alt.Tooltip("Series:N"),
                 alt.Tooltip("Age:Q", title="Vessel age", format=".1f"),
                 alt.Tooltip("Wastage:Q", title="Wastage (mm)", format=".3f")],
    )

    lim = pd.DataFrame({
        "y":     [LIMIT_ABS, LIMIT_SEVERE],
        "label": ["1.0 mm ABS average", "3.5 mm ABS severe"],
    })
    rules = alt.Chart(lim).mark_rule(
        strokeDash=[2, 4], color="#ef4444", opacity=0.75
    ).encode(y=alt.Y("y:Q", scale=alt.Scale(domain=[0, 4])))
    tags = alt.Chart(lim).mark_text(
        align="left", dx=6, dy=-7, color="#ef4444", fontSize=11
    ).encode(y=alt.Y("y:Q", scale=alt.Scale(domain=[0, 4])),
             x=alt.value(4), text="label:N")

    cur = pd.DataFrame({"x": [current_age], "y": [w_now]})
    vline = alt.Chart(cur).mark_rule(color="#6b7280", opacity=0.6).encode(
        x=alt.X("x:Q", scale=alt.Scale(domain=[11, 27], nice=False)))
    dot = alt.Chart(cur).mark_point(
        filled=True, size=120, color="#3b82f6", stroke="white", strokeWidth=1.5
    ).encode(
        x=alt.X("x:Q", scale=alt.Scale(domain=[11, 27], nice=False)),
        y=alt.Y("y:Q", scale=alt.Scale(domain=[0, 4])),
        tooltip=[alt.Tooltip("x:Q", title="Vessel age now", format=".1f"),
                 alt.Tooltip("y:Q", title="Wastage now (mm)", format=".3f")])

    chart = alt.layer(lines, rules, tags, vline, dot).properties(height=360)
    return chart.configure_axis(gridOpacity=0.18).configure_view(stroke=None)


def draw_chart(chart):
    try:
        st.altair_chart(chart, use_container_width=True)
    except Exception:
        st.altair_chart(chart, width="stretch")


def inspection_milestones(current_age, coating_life, offset, yrs_to_abs):
    rows = []
    for name, share in (("Half-life", 0.50), ("Warning", 0.75),
                        ("Critical", 0.90), ("Limit reached", 1.00)):
        age = round(current_age + yrs_to_abs * share, 2)
        wastage = float(model.predict(build_feature_row(age, coating_life))[0]) + offset
        rows.append([
            f"<b>{name}</b>",
            f"{int(share * 100)} %",
            f"{age:.2f} yr",
            f"+{yrs_to_abs * share:.2f} yr",
            f"{wastage:.3f} mm",
        ])
    return rows


def build_summary(current_age, coating_life, measured_wastage, plate_mm,
                  ml, paik, status_label, rate):
    clean = status_label.split(" ", 1)[1]
    def life(yrs, age):
        return "beyond 27 yr" if yrs is None else f"{yrs} yr (vessel age {age})"
    lines = [
        "FPSO CORROSION LIFE ASSESSMENT",
        PROJECT_TITLE,
        f"{AUTHOR} | {DEPARTMENT} | Supervisor: {SUPERVISOR}",
        "",
        "INPUTS",
        f"  Vessel age            : {current_age:g} yr",
        f"  Coating life (Tc)     : {coating_life:g} yr",
        "  Measured wastage      : " + (f"{measured_wastage:.2f} mm" if measured_wastage is not None
                                       else "not provided (fleet estimate)"),
        "  Original thickness    : " + (f"{plate_mm:.1f} mm" if plate_mm is not None else "not provided"),
        "",
        "RESULT",
        f"  Status                : {clean}",
        f"  Wastage used          : {ml['wastage_used_mm']} mm",
        f"  Corrosion rate        : " + (f"{rate:.4f} mm/yr" if rate is not None else "n/a"),
        f"  Paik average / severe : {paik['paik_avg_mm']} / {paik['paik_severe_mm']} mm",
        f"  Life to 1.0 mm (ML)   : {life(ml['yrs_to_abs'], ml['age_at_abs'])}",
        f"  Life to 1.0 mm (Paik) : {life(paik['paik_yrs_to_abs'], paik['paik_age_at_abs'])}",
        f"  Life to 3.5 mm (ML)   : {life(ml['yrs_to_severe'], ml['age_at_severe'])}",
        f"  Life to 3.5 mm (Paik) : {life(paik['paik_yrs_to_severe'], paik['paik_age_at_severe'])}",
        "",
        "Valid for vessel ages 11-27 yr. General wastage only (no pitting).",
        "Model: XGBoost regressor trained on Paik et al. (2004), 1,937 observations.",
    ]
    return "\n".join(lines)


# ══════════════════════════════════════════════════════════════════
# APP LAYOUT
# ══════════════════════════════════════════════════════════════════

md(CSS)

# ── Header ────────────────────────────────────────────────────────
md(f"""
<div class="eyebrow">University of Port Harcourt · B.Eng Final Year Project</div>
<div class="hero-title">FPSO Hull Corrosion Life Assessment</div>
<div class="hero-sub">Predicts ballast tank plate wastage and how long the plate has before it reaches the ABS limits.</div>
<div class="meta-card">
  <div class="meta-project">{html.escape(PROJECT_TITLE)}</div>
  <div class="meta-grid">
    {meta("Author", AUTHOR)}
    {meta("Department", DEPARTMENT)}
    {meta("Project supervisor", SUPERVISOR)}
    {meta("Dataset", DATASET_INFO)}
  </div>
</div>
""")

# ── Sidebar — user inputs ─────────────────────────────────────────
st.sidebar.header("Plate inputs")

md('<div class="side-label">Vessel</div>')
current_age = st.sidebar.slider(
    label="Vessel Age (years)",
    min_value=11.0,
    max_value=27.0,
    value=15.0,
    step=0.5,
    help="Current age of the vessel. Range matches Paik dataset (11-27yr)."
)

coating_life = st.sidebar.selectbox(
    label="Coating Life — Tc (years)",
    options=[5.0, 7.5, 10.0],
    index=1,
    format_func=lambda x: {
        5.0:  "5.0 yr — Undesirable",
        7.5:  "7.5 yr — Intermediate",
        10.0: "10.0 yr — Desirable",
    }[x],
    help=(
        "Coating life assumption from Paik et al. (2004) Table 2.\n"
        "5.0 yr: C1 mean = 0.0466 mm/yr\n"
        "7.5 yr: C1 mean = 0.0579 mm/yr\n"
        "10.0 yr: C1 mean = 0.0823 mm/yr"
    )
)

md('<div class="side-label">Inspection data (optional)</div>')
use_measured = st.sidebar.checkbox(
    "I have a measured wastage value", value=False
)

measured_wastage = None
if use_measured:
    measured_wastage = st.sidebar.number_input(
        label="Measured Wastage (mm)",
        min_value=0.0,
        max_value=4.0,
        value=0.5,
        step=0.05,
        help="Corrosion depth from ultrasonic thickness gauge (UTM). "
             "The projection is anchored to this value."
    )

use_thickness = st.sidebar.checkbox(
    "I know the original plate thickness", value=False
)

plate_mm = None
if use_thickness:
    plate_mm = st.sidebar.number_input(
        label="Original Thickness (mm)",
        min_value=6.0,
        max_value=40.0,
        value=15.0,
        step=0.5,
        help="As-built plate thickness. Used to show the thickness left."
    )

st.sidebar.write("")
try:
    run_button = st.sidebar.button(
        "▶  Run Assessment", use_container_width=True, type="primary"
    )
except Exception:
    run_button = st.sidebar.button(
        "▶  Run Assessment", width="stretch", type="primary"
    )

md("""
<div class="tile" style="min-height:0; margin-top:1.2rem;">
  <div class="tile-label">Model</div>
  <div class="tile-sub" style="margin-top:.35rem; opacity:.8;">
    XGBoost regressor<br>1,937 ballast tank readings<br>
    Valid for vessel age 11 – 27 yr
  </div>
</div>
""")

# Results stay on screen after the first run and update as inputs change.
if "ran" not in st.session_state:
    st.session_state["ran"] = False
if run_button:
    st.session_state["ran"] = True

# ── Main panel ────────────────────────────────────────────────────
if not st.session_state["ran"]:
    left, right = st.columns([1.5, 1], gap="large")

    with left:
        sec("How to use it")
        md("""
        <div class="step"><div class="step-n">1</div>
          <div class="step-t">Set the vessel age and coating life in the sidebar.</div></div>
        <div class="step"><div class="step-n">2</div>
          <div class="step-t">If you have a UTM reading, add it. The projection is then anchored to
          that plate. Without one, you get the fleet average for that age.</div></div>
        <div class="step"><div class="step-n">3</div>
          <div class="step-t">Press <b>Run Assessment</b>. After the first run the results update as
          you change the inputs.</div></div>
        """)
        note("Results are shown next to the Paik (2004) trend lines, so the ML projection "
             "can be checked against established practice.")

    with right:
        sec("Reference limits")
        md(tile("ABS average limit", "1.0", "mm", "ABS NDCV · Paik (2004) Eq. 5a"))
        st.write("")
        md(tile("ABS severe limit", "3.5", "mm", "95th percentile bound · Paik (2004) Eq. 5c"))
        st.write("")
        md("""
        <div class="tile" style="min-height:0;">
          <div class="tile-label">Paik trend lines</div>
          <div class="mono" style="margin-top:.3rem;">
            Eq. 3a&nbsp;&nbsp;Tc 5.0&nbsp;&nbsp;avg&nbsp;&nbsp;tr = 0.0466 × (T − Tc)<br>
            Eq. 3b&nbsp;&nbsp;Tc 7.5&nbsp;&nbsp;avg&nbsp;&nbsp;tr = 0.0579 × (T − Tc)<br>
            Eq. 3c&nbsp;&nbsp;Tc 10&nbsp;&nbsp;&nbsp;avg&nbsp;&nbsp;tr = 0.0823 × (T − Tc)<br>
            Eq. 4a&nbsp;&nbsp;Tc 5.0&nbsp;&nbsp;sev&nbsp;&nbsp;tr = 0.1469 × (T − Tc)<br>
            Eq. 4b&nbsp;&nbsp;Tc 7.5&nbsp;&nbsp;sev&nbsp;&nbsp;tr = 0.1938 × (T − Tc)<br>
            Eq. 4c&nbsp;&nbsp;Tc 10&nbsp;&nbsp;&nbsp;sev&nbsp;&nbsp;tr = 0.2894 × (T − Tc)
          </div>
        </div>
        """)

else:
    # ── Run both assessments ──────────────────────────────────────
    ml   = ml_assessment(current_age, coating_life, measured_wastage)
    paik = paik_assessment(current_age, coating_life)
    w      = ml["wastage_used_mm"]
    offset = ml["offset_mm"]
    status_label, status_msg = get_status(w)
    status_cls = STATUS_CLASS[status_label[0]]

    Te   = max(0.0, current_age - coating_life)
    rate = (w / Te) if Te > 0 else None
    source = ("Measured reading (UTM)" if measured_wastage is not None
              else "Model estimate for a typical plate")

    # ── Status card with gauge ────────────────────────────────────
    md(f"""
    <div class="status {status_cls}">
      <div class="status-head">{status_label}</div>
      <div class="status-msg">{status_msg}</div>
      {gauge_html(w)}
      <div class="status-meta">{source} · vessel age {current_age:g} yr · Tc {coating_life:g} yr</div>
    </div>
    """)

    # ── Key figures ───────────────────────────────────────────────
    yrs, age_at = ml["yrs_to_abs"], ml["age_at_abs"]
    if yrs is None:
        life_tile = tile("Life to 1.0 mm", "—", "", f"no breach before {HORIZON:g} yr")
    elif yrs == 0:
        life_tile = tile("Life to 1.0 mm", "0", "yr", "limit already reached")
    else:
        life_tile = tile("Life to 1.0 mm", f"{yrs:.2f}", "yr", f"reached at vessel age {age_at:g}")

    tiles = [
        tile("Wastage now", f"{w:.3f}", "mm",
             "measured" if measured_wastage is not None else "ML estimate"),
        tile("Corrosion rate",
             f"{rate:.4f}" if rate is not None else "—",
             "mm/yr" if rate is not None else "",
             "wastage ÷ (T − Tc)"),
        life_tile,
        tile("Margin used", f"{w / LIMIT_ABS * 100:.0f}", "%", "of the 1.0 mm ABS limit"),
    ]
    if plate_mm is not None:
        tiles.append(tile("Thickness left", f"{max(plate_mm - w, 0.0):.2f}", "mm",
                          f"of {plate_mm:.1f} mm original"))

    for col, t in zip(st.columns(len(tiles)), tiles):
        with col:
            md(t)

    st.write("")
    tab_life, tab_proj, tab_plan, tab_notes = st.tabs(
        ["Remaining life", "Projection", "Inspection plan", "Notes"]
    )

    # ── Tab 1: remaining life ─────────────────────────────────────
    with tab_life:
        sec("Time to the ABS limits")
        rows = [
            ['<b>1.0 mm</b><span class="dim">ABS average limit · Eq. 5a</span>',
             format_years(ml["yrs_to_abs"], ml["age_at_abs"]),
             format_years(paik["paik_yrs_to_abs"], paik["paik_age_at_abs"]),
             fmt_diff(ml["yrs_to_abs"], paik["paik_yrs_to_abs"])],
            ['<b>3.5 mm</b><span class="dim">ABS severe limit · Eq. 5c</span>',
             format_years(ml["yrs_to_severe"], ml["age_at_severe"]),
             format_years(paik["paik_yrs_to_severe"], paik["paik_age_at_severe"]),
             fmt_diff(ml["yrs_to_severe"], paik["paik_yrs_to_severe"])],
        ]
        md(html_table(["Threshold", "ML model", "Paik average", "Difference (ML − Paik)"], rows))

        if ml["yrs_to_abs"] is not None and paik["paik_yrs_to_abs"] is not None:
            diff_abs = round(ml["yrs_to_abs"] - paik["paik_yrs_to_abs"], 2)
            if diff_abs > 0:
                note(f"The ML model gives <b>{diff_abs:.2f} yr more</b> before 1.0 mm than the "
                     f"Paik average. Paik may be conservative for this plate.")
            elif diff_abs < 0:
                note(f"The ML model gives <b>{abs(diff_abs):.2f} yr less</b> before 1.0 mm than "
                     f"the Paik average. Earlier intervention may be warranted.", "warn")
            else:
                note("Both models agree on the remaining life to 1.0 mm.")

        sec("Wastage at the current age")
        d_avg = w - paik["paik_avg_mm"]
        d_sev = w - paik["paik_severe_mm"]
        c1, c2, c3 = st.columns(3)
        with c1:
            md(tile("Plate (used in this assessment)", f"{w:.3f}", "mm",
                    f"offset applied {offset:+.4f} mm" if measured_wastage is not None
                    else "fleet-level model estimate"))
        with c2:
            md(tile(f"Paik 2004 · average · Eq. {EQ_AVG[float(coating_life)]}",
                    f"{paik['paik_avg_mm']}", "mm",
                    f"C1 = {c1_avg(coating_life)} · plate is {abs(d_avg):.3f} mm "
                    f"{'above' if d_avg > 0 else 'below'}"))
        with c3:
            md(tile(f"Paik 2004 · severe · Eq. {EQ_SEVERE[float(coating_life)]}",
                    f"{paik['paik_severe_mm']}", "mm",
                    f"C1 = {c1_severe(coating_life)} · plate is {abs(d_sev):.3f} mm "
                    f"{'above' if d_sev > 0 else 'below'}"))

    # ── Tab 2: projection ─────────────────────────────────────────
    with tab_proj:
        sec("Wastage against vessel age")
        draw_chart(projection_chart(chart_frame(coating_life, offset), current_age, w))

        sec("Year by year")
        tbl = projection_table(current_age, coating_life, offset, plate_mm, step=1.0)
        headers = ["Vessel age", "ML model", "Paik average", "Paik severe", "Status"]
        if plate_mm is not None:
            headers.append("Thickness left")
        rows = []
        for _, r in tbl.iterrows():
            row = [f"{r['Vessel age (yr)']:g} yr",
                   f"{r['ML model (mm)']:.3f} mm",
                   f"{r['Paik average (mm)']:.3f} mm",
                   f"{r['Paik severe (mm)']:.3f} mm",
                   r["Status"]]
            if plate_mm is not None:
                row.append(f"{r['Thickness left (mm)']:.2f} mm")
            rows.append(row)
        md(html_table(headers, rows, highlight_first=True))

        st.write("")
        summary = build_summary(current_age, coating_life, measured_wastage,
                                plate_mm, ml, paik, status_label, rate)
        d1, d2, _ = st.columns([1, 1, 2])
        with d1:
            st.download_button(
                "Download projection (CSV)",
                data=projection_table(current_age, coating_life, offset,
                                      plate_mm, step=0.5).round(4).to_csv(index=False),
                file_name="corrosion_projection.csv",
                mime="text/csv",
            )
        with d2:
            st.download_button(
                "Download summary (TXT)",
                data=summary,
                file_name="corrosion_assessment_summary.txt",
                mime="text/plain",
            )
        with st.expander("Summary text (use the copy icon)"):
            st.code(summary, language=None)

    # ── Tab 3: inspection plan ────────────────────────────────────
    with tab_plan:
        sec("When to look at this plate again")
        R = ml["yrs_to_abs"]
        if R is None:
            note(f"The ML model shows no breach of 1.0 mm before {HORIZON:g} yr. "
                 f"Keep to the normal class survey cycle and update with a new UTM reading "
                 f"at the next survey.")
        elif R == 0:
            note("The plate is already at or above 1.0 mm. Inspect now and plan renewal "
                 "with the classification society.", "bad")
        else:
            md(html_table(
                ["Milestone", "Share of remaining life used", "Vessel age", "From now", "Expected wastage"],
                inspection_milestones(current_age, coating_life, offset, R)))
            half_age = current_age + R * 0.5
            if R < 10:
                note(f"Under 10 years remain to 1.0 mm, so plan the next inspection by the "
                     f"half-life point, at vessel age <b>{half_age:.1f} yr</b>.", "warn")
            else:
                note(f"Half of the remaining life is used at vessel age <b>{half_age:.1f} yr</b>. "
                     f"Re-run with a fresh UTM reading before then.")
            st.caption("Planning aid only. Class survey requirements always apply.")

    # ── Tab 4: notes ──────────────────────────────────────────────
    with tab_notes:
        sec("Limitations")
        st.markdown(
            "- General wastage only. Pitting is not modelled, so a plate with a good average "
            "can still need renewal.\n"
            "- Valid for vessel age 11 to 27 years. Tree-based models do not extrapolate "
            "beyond the training range.\n"
            "- The projection assumes conditions stay as they are. A measured value shifts "
            "the curve by a fixed offset, which is a first-order correction.\n"
            "- Small negative ML estimates mean no measurable corrosion yet. They are left "
            "unclipped to show the model output as it is."
        )
        sec("References")
        st.markdown(
            "- Paik, J.K., Thayamballi, A.K., Park, Y.I. & Hwang, J.S. (2004). A time-dependent "
            "corrosion wastage model for seawater ballast tank structures of ships. "
            "*Corrosion Science*, 46(3), 471–486.\n"
            "- Ebunuoha, C., Saturday, E.G. & Ebieto, C.E. (2025). Life Extension of Ageing FPSO "
            "and SPM Calm Buoy Hulls and Deck Plates Under Corrosion Effects — Part 1A. "
            "*International Journal of Engineering and Advanced Technology Studies*, 13(3), 42–70.\n"
            "- Still, J., Speck, J. & Pereira, A. (2004). Quality Requirements for an FPSO Hull "
            "and Marine Piping Fabrication. *OMAE-FPSO'04-0088*, TWI Global."
        )

# ── Footer ────────────────────────────────────────────────────────
st.divider()
st.caption(
    f"{AUTHOR}  |  {DEPARTMENT}  |  Supervisor: {SUPERVISOR}  \n"
    "Valid for vessel ages 11–27 yr. General wastage only (no pitting). "
    "Assumes constant conditions going forward.  \n"
    "Model: XGBoost regressor trained on Paik et al. (2004), "
    "1,937 observations."
)