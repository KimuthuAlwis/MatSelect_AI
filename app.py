"""
MatSelect AI — Stage 3 & 4: Web Interface with AI Query + Loading Case PI
Author: Kimuthu
Version: 2.1 — Fixed PI HTML rendering
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from io import BytesIO
import sys, os, json
from google import genai
from google.genai import types

sys.path.insert(0, os.path.dirname(__file__))
from matselect_engine import (
    load_database, filter_materials, score_and_rank,
    calc_performance_indices, rank_by_loading_case,
    COL, CAT_COLORS, QUAL_RANK, LOADING_CASES
)

# ── Page config ──
st.set_page_config(page_title="MatSelect AI", page_icon="🔬",
                   layout="wide", initial_sidebar_state="expanded")

# ── Gemini client ──
try:
    client = genai.Client()
except Exception:
    client = None

def parse_with_gemini(user_prompt: str) -> dict:
    if not client:
        return {}
    system_instruction = """
    You are an expert Material Selection System API. Convert the user's requirement into numerical constraint thresholds.
    You must return ONLY a flat valid JSON object with these exact keys:
    {
        "density_max": float or null,
        "uts_min": float or null,
        "modulus_min": float or null,
        "cost_max": int (1 to 5) or null,
        "explanation": "1-sentence engineering reason"
    }
    Guidelines:
    - 'Lightweight' or 'low density' -> density_max: 2.5 to 3.0
    - 'Strong' or 'high strength' -> uts_min: 300 to 500
    - 'Stiff' or 'rigid' -> modulus_min: 70 to 150
    - 'Cheap' or 'low-cost' -> cost_max: 2
    """
    try:
        response = client.models.generate_content(
            model='gemini-2.5-flash',
            contents=f"Extract metrics for: '{user_prompt}'",
            config=types.GenerateContentConfig(
                system_instruction=system_instruction,
                response_mime_type="application/json",
                temperature=0.1,
            ),
        )
        return json.loads(response.text)
    except Exception as e:
        return {"explanation": f"API Error: {str(e)}"}

# ── Load data ──
@st.cache_data
def get_data():
    return load_database("MatSelect_Database_v1.xlsx")

df = get_data()

# ── Session state defaults ──
if "max_density"    not in st.session_state: st.session_state["max_density"]    = 5.0
if "min_uts"        not in st.session_state: st.session_state["min_uts"]        = 100.0
if "min_E"          not in st.session_state: st.session_state["min_E"]          = 1.0
if "max_cost"       not in st.session_state: st.session_state["max_cost"]       = 5
if "ai_explanation" not in st.session_state: st.session_state["ai_explanation"] = ""
if "ai_caption"     not in st.session_state: st.session_state["ai_caption"]     = ""
if "last_ai_input"  not in st.session_state: st.session_state["last_ai_input"]  = ""

# ── CSS ──
st.markdown("""
<style>
    .main { background-color: #0f1117; }
    .block-container { padding-top: 1.5rem; }
    .mat-card {
        background: linear-gradient(135deg, #1e2130, #252a3d);
        border: 1px solid #2e3450; border-radius: 12px;
        padding: 1.2rem 1.5rem; margin-bottom: 0.4rem;
    }
    .mat-card h3 { margin: 0 0 4px 0; font-size: 1.05rem; color: #e0e6ff; }
    .mat-card .badge {
        display: inline-block; padding: 2px 10px; border-radius: 20px;
        font-size: 0.72rem; font-weight: 600; margin-right: 6px; margin-bottom: 8px;
    }
    .badge-Metal     { background:#1a4a6e; color:#7ec8f4; }
    .badge-Polymer   { background:#1a4e2e; color:#6fcf97; }
    .badge-Ceramic   { background:#4e3a0e; color:#f2c94c; }
    .badge-Composite { background:#4e1a1a; color:#f2776c; }
    .score-bar-bg { background:#2a2d3e; border-radius:6px; height:10px; margin: 6px 0 10px 0; }
    .score-bar-fill { height:10px; border-radius:6px;
                      background: linear-gradient(90deg,#4f8ef7,#9b59f7); }
    .pi-bar-fill { height:10px; border-radius:6px;
                   background: linear-gradient(90deg,#f7c94f,#f76f4f); }
    .prop-grid { display:grid; grid-template-columns:repeat(3,1fr); gap:6px; margin-top:8px; }
    .prop-item { background:#1a1d2e; border-radius:8px; padding:6px 10px; }
    .prop-item .label { font-size:0.68rem; color:#888; }
    .prop-item .value { font-size:0.92rem; color:#e0e6ff; font-weight:600; }
    .pi-highlight {
        background: linear-gradient(135deg, #1a2e1a, #1e3a1e);
        border: 1px solid #2e6e2e; border-radius:8px; padding:10px 14px;
        margin-bottom: 0.8rem;
    }
    .pi-highlight .pi-label { font-size:0.72rem; color:#6fcf97; font-weight:600; margin-bottom:4px; }
    .pi-highlight .pi-value { font-size:1.15rem; color:#a8f0a8; font-weight:700; }
    .pi-highlight .pi-unit  { font-size:0.72rem; color:#6fcf97; font-weight:400; margin-left:6px; }
    .metric-box {
        background:#1e2130; border:1px solid #2e3450;
        border-radius:10px; padding:14px 18px; text-align:center;
    }
    .metric-box .mval { font-size:1.6rem; font-weight:700; color:#4f8ef7; }
    .metric-box .mlbl { font-size:0.75rem; color:#888; margin-top:2px; }
    .section-title {
        font-size:1.15rem; font-weight:700; color:#e0e6ff;
        border-left:3px solid #4f8ef7; padding-left:10px;
        margin: 1.2rem 0 0.8rem 0;
    }
    .lc-card {
        background: linear-gradient(135deg, #1a1e30, #1e2540);
        border: 2px solid #4f8ef7; border-radius:12px;
        padding: 1rem 1.2rem; margin-bottom: 1rem;
    }
    .lc-card h4 { color:#7ec8f4; margin:0 0 4px 0; font-size:0.95rem; }
    .lc-card p  { color:#aaa; font-size:0.8rem; margin:0; }
    .lc-pi-formula {
        background:#0d1020; border-radius:8px; padding:8px 14px;
        font-family: monospace; font-size:1rem; color:#f7c94f;
        margin: 8px 0; display:inline-block;
    }
    div[data-testid="stSidebar"] { background:#0d1020; }
</style>
""", unsafe_allow_html=True)

# ════════════════════════════════════════
# HEADER
# ════════════════════════════════════════
st.markdown("""
<h1 style='margin-bottom:0; color:#e0e6ff;'>🔬 MatSelect <span style='color:#4f8ef7;'>AI</span></h1>
<p style='color:#888; margin-top:4px;'>Intelligent Material Selection System — Ashby PI Edition</p>
""", unsafe_allow_html=True)
st.divider()

# ════════════════════════════════════════
# AI QUERY (before sidebar so session state updates first)
# ════════════════════════════════════════
st.markdown("<div class='section-title'>🤖 AI Materials Assistant</div>", unsafe_allow_html=True)
ai_input = st.text_input("Describe what you are building:",
    placeholder="e.g. lightweight strong bracket for an aircraft that won't corrode",
    key="ai_text_input")

if ai_input and ai_input != st.session_state["last_ai_input"]:
    with st.spinner("Gemini is extracting engineering parameters..."):
        ai_limits = parse_with_gemini(ai_input)
    if ai_limits.get("density_max") is not None:
        st.session_state["max_density"] = float(ai_limits["density_max"])
    if ai_limits.get("uts_min") is not None:
        st.session_state["min_uts"]     = float(ai_limits["uts_min"])
    if ai_limits.get("modulus_min") is not None:
        st.session_state["min_E"]       = float(ai_limits["modulus_min"])
    if ai_limits.get("cost_max") is not None:
        st.session_state["max_cost"]    = int(ai_limits["cost_max"])
    st.session_state["last_ai_input"]  = ai_input
    st.session_state["ai_explanation"] = ai_limits.get("explanation", "")
    st.session_state["ai_caption"] = (
        f"🔧 Max Density: `{ai_limits.get('density_max','Any')}` g/cm³ | "
        f"Min UTS: `{ai_limits.get('uts_min','Any')}` MPa | "
        f"Min E: `{ai_limits.get('modulus_min','Any')}` GPa | "
        f"Max Cost: `{ai_limits.get('cost_max','Any')}/5`"
    )
    st.rerun()

if st.session_state["ai_explanation"]:
    st.info(f"**Gemini Analysis:** {st.session_state['ai_explanation']}")
    st.caption(st.session_state["ai_caption"])

st.divider()

# ════════════════════════════════════════
# LOADING CASE SELECTOR
# ════════════════════════════════════════
st.markdown("<div class='section-title'>⚙️ Step 1 — Select Your Loading Case</div>",
            unsafe_allow_html=True)
st.caption("This determines which Ashby Performance Index (PI) is used to rank materials. "
           "This is the correct engineering approach — different load types require different indices.")

lc_options = {"None — use manual weights": None}
for key, lc in LOADING_CASES.items():
    lc_options[lc["label"]] = key

lc_choice_label = st.selectbox(
    "Loading / Failure Mode:",
    options=list(lc_options.keys()),
    index=0,
    help="Select the primary loading condition your component will experience."
)
loading_case_key = lc_options[lc_choice_label]

if loading_case_key:
    lc = LOADING_CASES[loading_case_key]
    st.markdown(f"""
    <div class='lc-card'>
        <h4>📐 {lc['label']}</h4>
        <p>{lc['description']}</p>
        <div style='margin-top:10px;'>
            <span style='color:#888; font-size:0.75rem;'>PERFORMANCE INDEX (Ashby):</span><br>
            <span class='lc-pi-formula'>{lc['pi_name']}</span>
            <span style='color:#666; font-size:0.75rem; margin-left:8px;'>[{lc['pi_unit']}]</span>
        </div>
        <div style='margin-top:8px;'>
            <span style='color:#888; font-size:0.72rem;'>🎯 Objective: </span>
            <span style='color:#ccc; font-size:0.75rem;'>{lc['objective']}</span>
        </div>
        <div style='margin-top:4px;'>
            <span style='color:#888; font-size:0.72rem;'>📌 Typical applications: </span>
            <span style='color:#aaa; font-size:0.75rem;'>{lc['typical_apps']}</span>
        </div>
    </div>
    """, unsafe_allow_html=True)

st.divider()

# ════════════════════════════════════════
# SIDEBAR
# ════════════════════════════════════════
with st.sidebar:
    st.image("https://img.icons8.com/fluency/96/test-tube.png", width=52)
    st.markdown("## ⚙️ MatSelect AI")
    st.caption("Ashby PI Edition | v2.1")
    st.divider()

    st.markdown("### 🔒 Hard Constraints")
    st.caption("Auto-set by AI. Adjust manually if needed.")

    max_density  = st.slider("Max Density (g/cm³)",        0.5,  10.0,
                              float(st.session_state["max_density"]), 0.1, key="max_density")
    min_uts      = st.slider("Min Tensile Strength (MPa)",  0.0, 2000.0,
                              float(st.session_state["min_uts"]),    10.0, key="min_uts")
    min_E        = st.slider("Min Young's Modulus (GPa)",   0.0,  500.0,
                              float(st.session_state["min_E"]),       1.0, key="min_E")
    max_cost     = st.slider("Max Relative Cost (1–5)",     1,    5,
                              int(st.session_state["max_cost"]),      1,   key="max_cost")
    min_max_temp = st.slider("Min Service Temp (°C)",       0,    1000,   0, 10)

    min_corr = st.selectbox("Min Corrosion Resistance",
                            ["Any","Poor","Moderate","Good","Excellent"])
    category_filter = st.multiselect("Category Filter",
                                     ["Metal","Polymer","Ceramic","Composite"],
                                     default=["Metal","Polymer","Ceramic","Composite"])
    st.divider()

    st.markdown("### ⚖️ Manual Priority Weights")
    st.caption("Only used when no loading case is selected above.")
    w_lightness  = st.slider("Lightness",        0, 10, 5)
    w_strength   = st.slider("Strength (UTS)",   0, 10, 5)
    w_stiffness  = st.slider("Stiffness (E)",    0, 10, 3)
    w_corrosion  = st.slider("Corrosion Res.",   0, 10, 3)
    w_cost       = st.slider("Cost Efficiency",  0, 10, 4)
    w_ductility  = st.slider("Ductility",        0, 10, 2)
    w_high_temp  = st.slider("High Temp Perf.",  0, 10, 2)
    w_electrical = st.slider("Electrical Cond.", 0, 10, 0)
    st.divider()

    if st.button("🔄 Reset to Defaults", use_container_width=True):
        for k, v in [("max_density",5.0),("min_uts",100.0),("min_E",1.0),
                     ("max_cost",5),("ai_explanation",""),
                     ("ai_caption",""),("last_ai_input","")]:
            st.session_state[k] = v
        st.rerun()

# ════════════════════════════════════════
# RUN ENGINE
# ════════════════════════════════════════
constraints = {
    "max_density":  float(max_density),
    "min_uts":      float(min_uts),
    "min_E":        float(min_E),
    "max_cost":     float(max_cost),
    "min_max_temp": min_max_temp,
    "min_corr":     None if min_corr == "Any" else min_corr,
    "category":     category_filter if category_filter else None,
}
manual_weights = {
    "lightness": w_lightness, "strength": w_strength,
    "stiffness": w_stiffness, "corrosion": w_corrosion,
    "cost_efficiency": w_cost, "ductility": w_ductility,
    "high_temp": w_high_temp, "electrical": w_electrical,
}

filtered = filter_materials(df, constraints)

if loading_case_key:
    ranked = rank_by_loading_case(filtered, loading_case_key)
else:
    ranked = score_and_rank(filtered, manual_weights)
    ranked = calc_performance_indices(ranked)

has_pi = "PI_PRIMARY" in ranked.columns and not ranked["PI_PRIMARY"].isna().all()
pi_max = ranked["PI_PRIMARY"].max() if has_pi else 1

# ════════════════════════════════════════
# TABS
# ════════════════════════════════════════
tabs = st.tabs(["🏆 Results & PI", "📊 Ashby Chart",
                "📐 PI Reference Table", "📋 Full Database"])

# ════════════════════════════════════════
# TAB 1 — RESULTS
# ════════════════════════════════════════
with tabs[0]:
    c1, c2, c3, c4 = st.columns(4)
    for col, val, lbl in [
        (c1, len(df),          "Materials in DB"),
        (c2, len(filtered),    "Passed Filters"),
        (c3, ranked["Score"].iloc[0] if not ranked.empty else 0, "Top Score/100"),
        (c4, ranked[COL["category"]].nunique() if not ranked.empty else 0, "Categories"),
    ]:
        with col:
            st.markdown(f"""<div class='metric-box'>
                <div class='mval'>{val}</div>
                <div class='mlbl'>{lbl}</div></div>""", unsafe_allow_html=True)

    # ranking mode banner
    if loading_case_key:
        lc = LOADING_CASES[loading_case_key]
        st.markdown(f"""
        <div style='background:#0d1a0d; border:1px solid #2e6e2e; border-radius:10px;
                    padding:12px 16px; margin:12px 0;'>
            <span style='color:#6fcf97; font-weight:700; font-size:0.85rem;'>
                ✅ RANKING BY ASHBY PI:
            </span>
            <span style='color:#a8f0a8; font-family:monospace; font-size:0.95rem; margin-left:8px;'>
                {lc['pi_name']}
            </span>
            <span style='color:#666; font-size:0.75rem; margin-left:8px;'>
                [{lc['pi_unit']}]
            </span><br>
            <span style='color:#555; font-size:0.75rem;'>{lc['objective']}</span>
        </div>""", unsafe_allow_html=True)
    else:
        st.markdown("""
        <div style='background:#1a1a0d; border:1px solid #6e6e2e; border-radius:10px;
                    padding:10px 16px; margin:12px 0;'>
            <span style='color:#f2c94c; font-size:0.82rem;'>
                ⚠️ No loading case selected — ranking by manual weights.
                Select a loading case above for academically rigorous Ashby PI ranking.
            </span>
        </div>""", unsafe_allow_html=True)

    st.markdown("<div class='section-title'>Top Recommended Materials</div>",
                unsafe_allow_html=True)

    if ranked.empty:
        st.warning("⚠️ No materials match constraints. Try relaxing the filters.")
    else:
        for _, row in ranked.head(8).iterrows():
            rank     = int(row["Rank"])
            cat      = row[COL["category"]]
            score    = row["Score"]
            rank_sym = ["🥇","🥈","🥉"][rank-1] if rank <= 3 else f"#{rank}"

            # ── Main material card ──
            st.markdown(f"""
            <div class='mat-card'>
              <div style='display:flex; align-items:center; margin-bottom:6px;'>
                <span style='font-size:1.4rem; margin-right:10px;'>{rank_sym}</span>
                <div>
                  <h3>{row[COL['name']]}</h3>
                  <span class='badge badge-{cat}'>{cat}</span>
                  <span style='color:#666; font-size:0.75rem;'>{row[COL['subcat']]}</span>
                </div>
                <div style='margin-left:auto; text-align:right;'>
                  <span style='font-size:1.5rem; font-weight:700; color:#4f8ef7;'>{score}</span>
                  <span style='color:#666; font-size:0.8rem;'>/100</span>
                </div>
              </div>
              <div class='score-bar-bg'>
                <div class='score-bar-fill' style='width:{int(score)}%;'></div>
              </div>
              <div class='prop-grid'>
                <div class='prop-item'><div class='label'>Density</div>
                  <div class='value'>{row[COL['density']]} g/cm³</div></div>
                <div class='prop-item'><div class='label'>Tensile Strength</div>
                  <div class='value'>{row[COL['uts']]} MPa</div></div>
                <div class='prop-item'><div class='label'>Young's Modulus</div>
                  <div class='value'>{row[COL['E']]} GPa</div></div>
                <div class='prop-item'><div class='label'>Max Temp</div>
                  <div class='value'>{row[COL['max_temp']]}°C</div></div>
                <div class='prop-item'><div class='label'>Corrosion Res.</div>
                  <div class='value'>{row[COL['corr']]}</div></div>
                <div class='prop-item'><div class='label'>Relative Cost</div>
                  <div class='value'>{row[COL['cost']]}/5</div></div>
              </div>
              <div style='margin-top:8px; color:#555; font-size:0.73rem;'>
                📌 {row[COL['apps']]}
              </div>
            </div>""", unsafe_allow_html=True)

            # ── PI highlight — separate st.markdown call (fixes rendering bug) ──
            if has_pi and pd.notna(row.get("PI_PRIMARY")):
                pi_val  = float(row["PI_PRIMARY"])
                pi_name = str(row.get("PI_PRIMARY_NAME", "PI"))
                pi_unit = str(row.get("PI_PRIMARY_UNIT", ""))
                pi_pct  = int(min(pi_val / pi_max * 100, 100)) if pi_max > 0 else 0
                st.markdown(f"""
                <div class='pi-highlight'>
                    <div class='pi-label'>▶ Ashby PI: {pi_name}</div>
                    <div class='pi-value'>{pi_val:.3f}
                        <span class='pi-unit'>{pi_unit}</span>
                    </div>
                    <div class='score-bar-bg' style='margin-top:6px;'>
                        <div class='pi-bar-fill' style='width:{pi_pct}%;'></div>
                    </div>
                </div>""", unsafe_allow_html=True)

    # Export
    if not ranked.empty:
        st.markdown("<div class='section-title'>Export</div>", unsafe_allow_html=True)
        export_cols = [COL["id"], COL["name"], COL["category"], COL["subcat"],
                       COL["density"], COL["uts"], COL["E"], COL["cost"],
                       COL["corr"], COL["max_temp"], "Score", "Rank", COL["apps"]]
        if has_pi:
            export_cols += ["PI_PRIMARY", "PI_PRIMARY_NAME", "PI_PRIMARY_UNIT"]
        export_df = ranked[[c for c in export_cols if c in ranked.columns]]
        buf = BytesIO()
        with pd.ExcelWriter(buf, engine="openpyxl") as w:
            export_df.to_excel(w, index=False, sheet_name="MatSelect Results")
        buf.seek(0)
        st.download_button("⬇️ Download Results as Excel", data=buf,
                           file_name="MatSelect_Results.xlsx",
                           mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                           use_container_width=True)

# ════════════════════════════════════════
# TAB 2 — ASHBY CHART
# ════════════════════════════════════════
with tabs[1]:
    st.markdown("<div class='section-title'>Interactive Ashby Chart</div>",
                unsafe_allow_html=True)

    if loading_case_key:
        lc = LOADING_CASES[loading_case_key]
        default_x = lc["x_label"].split("(")[0].strip()
        default_y = lc["y_label"].split("(")[0].strip()
        st.info(f"📐 Axes auto-set for **{lc['label']}** — "
                f"correct Ashby space for index **{lc['pi_name']}**")
    else:
        default_x = "Density"
        default_y = "Tensile Strength"

    axis_options = [
        "Density (g/cm³)", "Tensile Strength (MPa)", "Young's Modulus (GPa)",
        "Max Service Temp (°C)", "Relative Cost (1-5)",
        "Thermal Conductivity (W/m·K)", "Hardness (HV)"
    ]
    axis_to_col = {
        "Density (g/cm³)":             COL["density"],
        "Tensile Strength (MPa)":      COL["uts"],
        "Young's Modulus (GPa)":       COL["E"],
        "Max Service Temp (°C)":       COL["max_temp"],
        "Relative Cost (1-5)":         COL["cost"],
        "Thermal Conductivity (W/m·K)":COL["k"],
        "Hardness (HV)":               COL["hardness"],
    }

    col_a, col_b, col_c = st.columns([2,2,1])
    with col_a:
        x_choice = st.selectbox("X Axis", axis_options,
            index=next((i for i,o in enumerate(axis_options)
                        if default_x.lower() in o.lower()), 0))
    with col_b:
        y_choice = st.selectbox("Y Axis", axis_options,
            index=next((i for i,o in enumerate(axis_options)
                        if default_y.lower() in o.lower()), 1))
    with col_c:
        log_axes = st.checkbox("Log Scale", value=True)

    x_col = axis_to_col[x_choice]
    y_col = axis_to_col[y_choice]
    plot_df = df.dropna(subset=[x_col, y_col]).copy()
    top_ids = ranked.head(3)[COL["id"]].tolist() if not ranked.empty else []
    plot_df["highlight"] = plot_df[COL["id"]].isin(top_ids)
    plot_df["size"]   = plot_df["highlight"].map({True: 20, False: 10})
    plot_df["symbol"] = plot_df["highlight"].map({True: "star", False: "circle"})

    fig = px.scatter(plot_df, x=x_col, y=y_col,
                     color=COL["category"], color_discrete_map=CAT_COLORS,
                     hover_name=COL["name"],
                     hover_data={COL["id"]:True, COL["subcat"]:True,
                                 x_col:":.3f", y_col:":.3f", COL["apps"]:True,
                                 "highlight":False,"size":False,"symbol":False},
                     size="size", symbol="symbol",
                     symbol_map={"circle":"circle","star":"star"},
                     log_x=log_axes, log_y=log_axes,
                     title=f"<b>Ashby Chart: {y_choice} vs {x_choice}</b>",
                     template="plotly_dark")

    if loading_case_key and x_col == COL["density"] and y_col in [COL["uts"], COL["E"]]:
        lc = LOADING_CASES[loading_case_key]
        x_line = np.logspace(
            np.log10(max(plot_df[x_col].min()*0.5, 0.1)),
            np.log10(plot_df[x_col].max()*2), 100)
        slope  = lc["slope"]
        y_line = (x_line ** slope) * (
            plot_df[y_col].median() / (plot_df[x_col].median() ** slope))
        fig.add_trace(go.Scatter(x=x_line, y=y_line, mode="lines",
                                 line=dict(color="#f7c94f", width=1.5, dash="dash"),
                                 name=f"PI line: {lc['slope_label']}",
                                 showlegend=True))

    fig.update_layout(plot_bgcolor="#0f1117", paper_bgcolor="#0f1117",
                      font=dict(color="#e0e6ff"), height=540,
                      legend=dict(bgcolor="#1e2130", bordercolor="#2e3450"))
    fig.update_xaxes(gridcolor="#1e2130")
    fig.update_yaxes(gridcolor="#1e2130")
    st.plotly_chart(fig, use_container_width=True)
    st.caption("⭐ Stars = Top 3 ranked materials | Dashed line = Ashby PI selection line")

# ════════════════════════════════════════
# TAB 3 — PI REFERENCE TABLE
# ════════════════════════════════════════
with tabs[2]:
    st.markdown("<div class='section-title'>📐 Ashby PI Reference — All Loading Cases</div>",
                unsafe_allow_html=True)
    st.caption("Source: Ashby, M.F. — Materials Selection in Mechanical Design.")

    ref_data = []
    for key, lc in LOADING_CASES.items():
        ref_data.append({
            "Loading Case":         lc["label"],
            "Design Objective":     lc["objective"],
            "Performance Index":    lc["pi_name"],
            "Units":                lc["pi_unit"],
            "Typical Applications": lc["typical_apps"],
        })
    st.dataframe(pd.DataFrame(ref_data), use_container_width=True, height=420,
                 column_config={
                     "Loading Case":      st.column_config.TextColumn(width="medium"),
                     "Performance Index": st.column_config.TextColumn(width="small"),
                 })

    st.divider()
    st.markdown("<div class='section-title'>All PI Values — Current Filtered Materials</div>",
                unsafe_allow_html=True)

    if not ranked.empty:
        all_pi = calc_performance_indices(ranked)
        pi_display_cols = [
            COL["name"], COL["category"],
            "PI: σf/ρ — Specific Strength (Tie)",
            "PI: σf^(2/3)/ρ — Beam in Bending",
            "PI: σf^(1/2)/ρ — Panel in Bending",
            "PI: E^(1/2)/ρ — Column Buckling",
            "PI: E^(1/3)/ρ — Plate Buckling",
            "PI: σf²/(E·ρ) — Spring / Energy Storage",
            "PI: 0.45·σf/ρ — Fatigue Endurance (approx)",
            "PI: HV — Wear / Hardness",
        ]
        show_cols = [c for c in pi_display_cols if c in all_pi.columns]
        st.dataframe(all_pi[show_cols].reset_index(drop=True),
                     use_container_width=True, height=380)

        if loading_case_key and "PI_PRIMARY" in ranked.columns:
            lc = LOADING_CASES[loading_case_key]
            pi_bar = ranked.dropna(subset=["PI_PRIMARY"]).head(10)
            fig3 = px.bar(pi_bar, x=COL["name"], y="PI_PRIMARY",
                          color=COL["category"], color_discrete_map=CAT_COLORS,
                          template="plotly_dark",
                          title=f"<b>Ranking by Ashby PI: {lc['pi_name']}</b>",
                          labels={COL["name"]:"",
                                  "PI_PRIMARY": f"{lc['pi_name']} [{lc['pi_unit']}]"})
            fig3.update_layout(plot_bgcolor="#0f1117", paper_bgcolor="#0f1117",
                               font=dict(color="#e0e6ff"), height=380,
                               xaxis_tickangle=-25)
            st.plotly_chart(fig3, use_container_width=True)

# ════════════════════════════════════════
# TAB 4 — FULL DATABASE
# ════════════════════════════════════════
with tabs[3]:
    st.markdown("<div class='section-title'>Full Material Database</div>",
                unsafe_allow_html=True)
    search = st.text_input("🔎 Search", placeholder="e.g. aluminium, rubber, ceramic...")
    view_df = df.copy()
    if search:
        mask = view_df.apply(lambda r: r.astype(str).str.contains(
            search, case=False).any(), axis=1)
        view_df = view_df[mask]
    display_cols = [COL["id"], COL["name"], COL["category"], COL["subcat"],
                    COL["density"], COL["uts"], COL["E"], COL["hardness"],
                    COL["elong"], COL["max_temp"], COL["corr"], COL["cost"]]
    st.dataframe(view_df[display_cols].reset_index(drop=True),
                 use_container_width=True, height=420,
                 column_config={
                     COL["density"]:  st.column_config.NumberColumn("Density(g/cm³)", format="%.2f"),
                     COL["uts"]:      st.column_config.NumberColumn("UTS(MPa)",       format="%.0f"),
                     COL["E"]:        st.column_config.NumberColumn("E(GPa)",          format="%.1f"),
                     COL["hardness"]: st.column_config.NumberColumn("HV",             format="%.0f"),
                     COL["elong"]:    st.column_config.NumberColumn("Elong(%)",        format="%.1f"),
                     COL["max_temp"]: st.column_config.NumberColumn("MaxTemp(°C)",    format="%.0f"),
                     COL["cost"]:     st.column_config.NumberColumn("Cost(1-5)",       format="%.0f"),
                 })
    st.caption(f"Showing {len(view_df)} of {len(df)} materials")

# ── Footer ──
st.divider()
st.markdown("""
<p style='text-align:center; color:#444; font-size:0.75rem;'>
MatSelect AI v2.1 &nbsp;|&nbsp; Ashby PI Edition &nbsp;|&nbsp;
PI Reference: Ashby MF, Materials Selection in Mechanical Design &nbsp;|&nbsp;
Data: ASM Handbook, MatWeb, CES EduPack &nbsp;|&nbsp; Built by Kimuthu
</p>""", unsafe_allow_html=True)