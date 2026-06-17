"""
MatSelect AI — Stage 3 & 4: Web Interface with AI Query Layer
Author: Kimuthu
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from io import BytesIO
import sys, os, json
from google import genai
from google.genai import types

sys.path.insert(0, os.path.dirname(__file__))
from matselect_engine import (
    load_database, filter_materials, score_and_rank,
    calc_performance_indices, COL, CAT_COLORS, QUAL_RANK
)

# ── Page config (MUST be first Streamlit call) ──
st.set_page_config(
    page_title="MatSelect AI",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ── Gemini client ──
try:
    client = genai.Client()
except Exception:
    client = None

import time  # Make sure to import time at the top of your file if not present

def parse_with_gemini(user_prompt: str) -> dict:
    """Translates regular language into exact numbers matching your database limits with automatic retry logic."""
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
    
    max_retries = 3
    for attempt in range(max_retries):
        try:
            response = client.models.generate_content(
                model='gemini-2.5-flash',  # Back to the correct model name
                contents=f"Extract metrics for: '{user_prompt}'",
                config=types.GenerateContentConfig(
                    system_instruction=system_instruction,
                    response_mime_type="application/json",
                    temperature=0.1,
                ),
            )
            return json.loads(response.text)
            
        except Exception as e:
            # If it's a 503 traffic error and we have remaining attempts, wait and retry
            if "503" in str(e) and attempt < max_retries - 1:
                time.sleep(0.5)  # Wait 500ms before trying again
                continue
            return {"explanation": f"API Error: {str(e)}"}

# ── Load data ──
@st.cache_data
def get_data():
    return load_database("MatSelect_Database_v1.xlsx")

df = get_data()

# ════════════════════════════════════════
# STEP 1 — Initialize session state defaults
# (must happen before ANY widget is created)
# ════════════════════════════════════════
if "max_density" not in st.session_state: st.session_state["max_density"] = 5.0
if "min_uts"     not in st.session_state: st.session_state["min_uts"]     = 100.0
if "min_E"       not in st.session_state: st.session_state["min_E"]       = 1.0
if "max_cost"    not in st.session_state: st.session_state["max_cost"]    = 5
if "ai_explanation" not in st.session_state: st.session_state["ai_explanation"] = ""
if "ai_caption"     not in st.session_state: st.session_state["ai_caption"]     = ""
if "last_ai_input"  not in st.session_state: st.session_state["last_ai_input"]  = ""

# ── Custom CSS ──
st.markdown("""
<style>
    .main { background-color: #0f1117; }
    .block-container { padding-top: 1.5rem; }
    .mat-card {
        background: linear-gradient(135deg, #1e2130, #252a3d);
        border: 1px solid #2e3450;
        border-radius: 12px;
        padding: 1.2rem 1.5rem;
        margin-bottom: 0.8rem;
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
    .score-bar-wrap { margin: 6px 0 10px 0; }
    .score-bar-bg { background:#2a2d3e; border-radius:6px; height:10px; }
    .score-bar-fill { height:10px; border-radius:6px;
                      background: linear-gradient(90deg,#4f8ef7,#9b59f7); }
    .prop-grid { display:grid; grid-template-columns:repeat(3,1fr); gap:6px; margin-top:8px; }
    .prop-item { background:#1a1d2e; border-radius:8px; padding:6px 10px; }
    .prop-item .label { font-size:0.68rem; color:#888; }
    .prop-item .value { font-size:0.92rem; color:#e0e6ff; font-weight:600; }
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
    div[data-testid="stSidebar"] { background:#0d1020; }
</style>
""", unsafe_allow_html=True)

# ════════════════════════════════════════
# STEP 2 — HEADER + AI QUERY
# (runs before sidebar so session_state updates
#  before sliders are rendered)
# ════════════════════════════════════════
st.markdown("""
<h1 style='margin-bottom:0; color:#e0e6ff;'>🔬 MatSelect <span style='color:#4f8ef7;'>AI</span></h1>
<p style='color:#888; margin-top:4px;'>Intelligent Material Selection System for Engineering Design</p>
""", unsafe_allow_html=True)
st.divider()

st.markdown("<div class='section-title'>🤖 AI Materials Assistant Query</div>",
            unsafe_allow_html=True)

ai_input = st.text_input(
    "Describe what you are building in plain English:",
    placeholder="e.g. I need a lightweight strong material for a drone frame that won't corrode",
    key="ai_text_input"
)

# Only call Gemini when the input actually changes
if ai_input and ai_input != st.session_state["last_ai_input"]:
    with st.spinner("Gemini is extracting engineering parameters..."):
        ai_limits = parse_with_gemini(ai_input)

    # Update session state BEFORE sliders render
    if ai_limits.get("density_max") is not None:
        st.session_state["max_density"] = float(ai_limits["density_max"])
    if ai_limits.get("uts_min") is not None:
        st.session_state["min_uts"] = float(ai_limits["uts_min"])
    if ai_limits.get("modulus_min") is not None:
        st.session_state["min_E"] = float(ai_limits["modulus_min"])
    if ai_limits.get("cost_max") is not None:
        st.session_state["max_cost"] = int(ai_limits["cost_max"])

    # Save explanation and caption to session state for display
    st.session_state["last_ai_input"] = ai_input
    st.session_state["ai_explanation"] = ai_limits.get("explanation", "")
    st.session_state["ai_caption"] = (
        f"🔧 **Target Thresholds Extracted:** "
        f"Max Density: `{ai_limits.get('density_max', 'Any')}` g/cm³ | "
        f"Min UTS: `{ai_limits.get('uts_min', 'Any')}` MPa | "
        f"Min Modulus: `{ai_limits.get('modulus_min', 'Any')}` GPa | "
        f"Max Cost: `{ai_limits.get('cost_max', 'Any')}/5`"
    )
    st.rerun()  # rerun so sliders pick up new session_state values

# Show AI analysis output (from session state, persists across reruns)
if st.session_state["ai_explanation"]:
    st.info(f"**Gemini Analysis:** {st.session_state['ai_explanation']}")
    st.caption(st.session_state["ai_caption"])

st.markdown("<br>", unsafe_allow_html=True)

# ════════════════════════════════════════
# STEP 3 — SIDEBAR (sliders now read updated session_state)
# ════════════════════════════════════════
with st.sidebar:
    st.image("https://img.icons8.com/fluency/96/test-tube.png", width=52)
    st.markdown("## ⚙️ MatSelect AI")
    st.caption("Stage 4 — AI Layer Interface | v1.0")
    st.divider()

    st.markdown("### 🔒 Hard Constraints")
    st.caption("Auto-set by AI query. You can also adjust manually.")

    # Sliders use session_state values as defaults — they're already updated above
    max_density = st.slider("Max Density (g/cm³)",       0.5,  10.0,
                            float(st.session_state["max_density"]), 0.1,  key="max_density")
    min_uts     = st.slider("Min Tensile Strength (MPa)", 0.0, 2000.0,
                            float(st.session_state["min_uts"]),    10.0, key="min_uts")
    min_E       = st.slider("Min Young's Modulus (GPa)",  0.0,  500.0,
                            float(st.session_state["min_E"]),       1.0,  key="min_E")
    max_cost    = st.slider("Max Relative Cost (1–5)",    1,    5,
                            int(st.session_state["max_cost"]),      1,    key="max_cost")
    min_max_temp= st.slider("Min Service Temp (°C)",      0,    1000,   0, 10)

    min_corr = st.selectbox("Min Corrosion Resistance",
                            ["Any", "Poor", "Moderate", "Good", "Excellent"])
    category_filter = st.multiselect("Category Filter",
                                     ["Metal","Polymer","Ceramic","Composite"],
                                     default=["Metal","Polymer","Ceramic","Composite"])
    st.divider()

    st.markdown("### ⚖️ Priority Weights")
    st.caption("Set what matters most (0 = ignore).")
    w_lightness  = st.slider("Lightness (low density)",  0, 10, 5)
    w_strength   = st.slider("Strength (UTS)",           0, 10, 5)
    w_stiffness  = st.slider("Stiffness (Young's mod)",  0, 10, 3)
    w_corrosion  = st.slider("Corrosion Resistance",     0, 10, 3)
    w_cost       = st.slider("Cost Efficiency",          0, 10, 4)
    w_ductility  = st.slider("Ductility (elongation)",   0, 10, 2)
    w_high_temp  = st.slider("High Temp Performance",    0, 10, 2)
    w_electrical = st.slider("Electrical Conductivity",  0, 10, 0)
    st.divider()

    if st.button("🔄 Reset to Defaults", use_container_width=True):
        st.session_state["max_density"]    = 5.0
        st.session_state["min_uts"]        = 100.0
        st.session_state["min_E"]          = 1.0
        st.session_state["max_cost"]       = 5
        st.session_state["ai_explanation"] = ""
        st.session_state["ai_caption"]     = ""
        st.session_state["last_ai_input"]  = ""
        st.rerun()

# ════════════════════════════════════════
# STEP 4 — RUN ENGINE
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
weights = {
    "lightness":       w_lightness,
    "strength":        w_strength,
    "stiffness":       w_stiffness,
    "corrosion":       w_corrosion,
    "cost_efficiency": w_cost,
    "ductility":       w_ductility,
    "high_temp":       w_high_temp,
    "electrical":      w_electrical,
}

filtered  = filter_materials(df, constraints)
ranked    = score_and_rank(filtered, weights)
ranked_pi = calc_performance_indices(ranked)

# ════════════════════════════════════════
# TABS
# ════════════════════════════════════════
tabs = st.tabs(["🏆 Results", "📊 Ashby Charts", "📋 Full Database", "📈 Performance Indices"])

# ── TAB 1 — RESULTS ──
with tabs[0]:
    c1, c2, c3, c4 = st.columns(4)
    for col, val, lbl in [
        (c1, len(df),    "Materials in Database"),
        (c2, len(filtered), "Passed Constraints"),
        (c3, ranked["Score"].iloc[0] if not ranked.empty else 0, "Top Score / 100"),
        (c4, ranked[COL["category"]].nunique() if not ranked.empty else 0, "Categories Represented"),
    ]:
        with col:
            st.markdown(f"""<div class='metric-box'>
                <div class='mval'>{val}</div>
                <div class='mlbl'>{lbl}</div></div>""", unsafe_allow_html=True)

    st.markdown("<div class='section-title'>Top Recommended Materials</div>",
                unsafe_allow_html=True)

    if ranked.empty:
        st.warning("⚠️ No materials match your constraints. Try relaxing the filters.")
    else:
        for _, row in ranked.head(8).iterrows():
            rank     = int(row["Rank"])
            cat      = row[COL["category"]]
            score    = row["Score"]
            rank_sym = ["🥇","🥈","🥉"][rank-1] if rank <= 3 else f"#{rank}"
            bar_w    = int(score)

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
              <div class='score-bar-wrap'>
                <div class='score-bar-bg'>
                  <div class='score-bar-fill' style='width:{bar_w}%;'></div>
                </div>
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
              <div style='margin-top:10px; color:#666; font-size:0.75rem;'>
                📌 {row[COL['apps']]}
              </div>
            </div>""", unsafe_allow_html=True)

    if not ranked.empty:
        st.markdown("<div class='section-title'>Export Results</div>",
                    unsafe_allow_html=True)
        export_cols = [COL["id"], COL["name"], COL["category"], COL["subcat"],
                       COL["density"], COL["uts"], COL["E"], COL["cost"],
                       COL["corr"], COL["max_temp"], "Score", "Rank", COL["apps"]]
        export_df = ranked[[c for c in export_cols if c in ranked.columns]]
        buf = BytesIO()
        with pd.ExcelWriter(buf, engine="openpyxl") as writer:
            export_df.to_excel(writer, index=False, sheet_name="MatSelect Results")
        buf.seek(0)
        st.download_button("⬇️ Download Results as Excel", data=buf,
                           file_name="MatSelect_Results.xlsx",
                           mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                           use_container_width=True)

# ── TAB 2 — ASHBY CHARTS ──
with tabs[1]:
    st.markdown("<div class='section-title'>Interactive Ashby Charts</div>",
                unsafe_allow_html=True)
    col_a, col_b, col_c = st.columns([2, 2, 1])
    with col_a:
        x_choice = st.selectbox("X Axis", [
            "Density (g/cm³)", "Tensile Strength (MPa)",
            "Young's Modulus (GPa)", "Max Service Temp (°C)",
            "Relative Cost (1-5)", "Thermal Conductivity (W/m·K)"])
    with col_b:
        y_choice = st.selectbox("Y Axis", [
            "Tensile Strength (MPa)", "Young's Modulus (GPa)",
            "Density (g/cm³)", "Max Service Temp (°C)",
            "Relative Cost (1-5)", "Thermal Conductivity (W/m·K)"])
    with col_c:
        log_axes = st.checkbox("Log Scale", value=True)

    axis_to_col = {
        "Density (g/cm³)":             COL["density"],
        "Tensile Strength (MPa)":      COL["uts"],
        "Young's Modulus (GPa)":       COL["E"],
        "Max Service Temp (°C)":       COL["max_temp"],
        "Relative Cost (1-5)":         COL["cost"],
        "Thermal Conductivity (W/m·K)":COL["k"],
    }
    x_col = axis_to_col[x_choice]
    y_col = axis_to_col[y_choice]
    plot_df = df.dropna(subset=[x_col, y_col]).copy()
    plot_df["highlight"] = plot_df[COL["id"]].isin(
        ranked.head(3)[COL["id"]].tolist() if not ranked.empty else [])
    plot_df["size"]   = plot_df["highlight"].map({True: 18, False: 10})
    plot_df["symbol"] = plot_df["highlight"].map({True: "star", False: "circle"})

    fig = px.scatter(plot_df, x=x_col, y=y_col,
                     color=COL["category"], color_discrete_map=CAT_COLORS,
                     hover_name=COL["name"],
                     hover_data={COL["id"]: True, COL["subcat"]: True,
                                 x_col: ":.2f", y_col: ":.2f", COL["apps"]: True,
                                 "highlight": False, "size": False, "symbol": False},
                     size="size", symbol="symbol",
                     symbol_map={"circle": "circle", "star": "star"},
                     log_x=log_axes, log_y=log_axes,
                     title=f"<b>Ashby Chart: {y_choice} vs {x_choice}</b>",
                     template="plotly_dark")
    fig.update_layout(plot_bgcolor="#0f1117", paper_bgcolor="#0f1117",
                      font=dict(color="#e0e6ff"), height=520,
                      legend=dict(title="Category", bgcolor="#1e2130", bordercolor="#2e3450"))
    fig.update_xaxes(gridcolor="#1e2130")
    fig.update_yaxes(gridcolor="#1e2130")
    st.plotly_chart(fig, use_container_width=True)
    st.caption("⭐ Stars = Top 3 ranked materials from your current filter settings")

# ── TAB 3 — FULL DATABASE ──
with tabs[2]:
    st.markdown("<div class='section-title'>Full Material Database</div>",
                unsafe_allow_html=True)
    search = st.text_input("🔎 Search materials",
                           placeholder="e.g. aluminium, rubber, ceramic...")
    view_df = df.copy()
    if search:
        mask = view_df.apply(lambda row: row.astype(str).str.contains(
            search, case=False).any(), axis=1)
        view_df = view_df[mask]
    display_cols = [COL["id"], COL["name"], COL["category"], COL["subcat"],
                    COL["density"], COL["uts"], COL["E"], COL["hardness"],
                    COL["elong"], COL["max_temp"], COL["corr"], COL["cost"]]
    st.dataframe(view_df[display_cols].reset_index(drop=True),
                 use_container_width=True, height=420,
                 column_config={
                     COL["density"]:  st.column_config.NumberColumn("Density (g/cm³)", format="%.2f"),
                     COL["uts"]:      st.column_config.NumberColumn("UTS (MPa)",       format="%.0f"),
                     COL["E"]:        st.column_config.NumberColumn("E (GPa)",          format="%.1f"),
                     COL["hardness"]: st.column_config.NumberColumn("Hardness (HV)",   format="%.0f"),
                     COL["elong"]:    st.column_config.NumberColumn("Elong (%)",        format="%.1f"),
                     COL["max_temp"]: st.column_config.NumberColumn("Max Temp (°C)",   format="%.0f"),
                     COL["cost"]:     st.column_config.NumberColumn("Cost (1-5)",       format="%.0f"),
                 })
    st.caption(f"Showing {len(view_df)} of {len(df)} materials")

# ── TAB 4 — PERFORMANCE INDICES ──
with tabs[3]:
    st.markdown("<div class='section-title'>Ashby Performance Indices</div>",
                unsafe_allow_html=True)
    st.caption("Engineering performance metrics derived from material properties.")
    if ranked_pi.empty:
        st.warning("No materials match current constraints.")
    else:
        pi_cols = [COL["name"], COL["category"],
                   "PI: UTS/ρ (Specific Strength)", "PI: E/ρ (Specific Stiffness)",
                   "PI: UTS^(2/3)/ρ (Beam Lightness)", "PI: E^(1/2)/ρ (Plate Stiffness)", "Score"]
        show_pi = ranked_pi[[c for c in pi_cols if c in ranked_pi.columns]]
        st.dataframe(show_pi.reset_index(drop=True), use_container_width=True, height=360)

        st.markdown("<div class='section-title'>Specific Strength Ranking</div>",
                    unsafe_allow_html=True)
        pi_bar = ranked_pi.dropna(subset=["PI: UTS/ρ (Specific Strength)"]).head(10)
        fig2 = px.bar(pi_bar, x=COL["name"], y="PI: UTS/ρ (Specific Strength)",
                      color=COL["category"], color_discrete_map=CAT_COLORS,
                      template="plotly_dark",
                      title="<b>Specific Strength (UTS/ρ) — Top Materials</b>",
                      labels={COL["name"]: "", "PI: UTS/ρ (Specific Strength)": "UTS/ρ (MPa·cm³/g)"})
        fig2.update_layout(plot_bgcolor="#0f1117", paper_bgcolor="#0f1117",
                           font=dict(color="#e0e6ff"), height=380, xaxis_tickangle=-30)
        st.plotly_chart(fig2, use_container_width=True)

# ── Footer ──
st.divider()
st.markdown("""
<p style='text-align:center; color:#444; font-size:0.75rem;'>
MatSelect AI v1.0 &nbsp;|&nbsp; Stage 4: AI Query Integration &nbsp;|&nbsp;
Data: ASM Handbook, MatWeb, CES EduPack &nbsp;|&nbsp; Built by Kimuthu
</p>""", unsafe_allow_html=True)