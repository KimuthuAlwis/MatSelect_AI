"""
MatSelect AI — Stage 2: Engineering Logic Engine
Author: Kimuthu
Version: 1.0

Modules:
  1. Data loader
  2. Hard constraint filter
  3. Weighted scoring & ranking
  4. Ashby chart generator
  5. Performance index calculator
  6. CLI demo
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import warnings
warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────
# COLUMN MAP  (short aliases → actual col names)
# ─────────────────────────────────────────────
COL = {
    "id":       "ID",
    "name":     "Material Name",
    "category": "Category",
    "subcat":   "Sub-Category",
    "density":  "Density\n(g/cm³)",
    "uts":      "Tensile Strength\n(MPa)",
    "ys":       "Yield Strength\n(MPa)",
    "E":        "Young's Modulus\n(GPa)",
    "hardness": "Hardness\n(HV)",
    "elong":    "Elongation\n(%)",
    "k":        "Thermal Conductivity\n(W/m·K)",
    "cte":      "CTE\n(µm/m·°C)",
    "max_temp": "Max Service Temp\n(°C)",
    "elec":     "Electrical Conductivity\n(MS/m)",
    "corr":     "Corrosion Resistance",
    "mach":     "Machinability",
    "weld":     "Weldability",
    "cost":     "Relative Cost\n(1-5)",
    "apps":     "Typical Applications",
    "notes":    "Notes",
}

QUAL_RANK = {"Excellent": 4, "Good": 3, "Moderate": 2, "Poor": 1, "N/A": 0}

CAT_COLORS = {
    "Metal":     "#2E86C1",
    "Polymer":   "#1E8449",
    "Ceramic":   "#B7950B",
    "Composite": "#A93226",
}

DB_PATH = "MatSelect_Database_v1.xlsx"


# ══════════════════════════════════════════════
# 1. DATA LOADER
# ══════════════════════════════════════════════
def load_database(path=DB_PATH):
    df = pd.read_excel(path, sheet_name="Material Database")
    # add numeric version of qualitative cols for scoring
    for col in ["corr", "mach", "weld"]:
        df[COL[col] + "_num"] = df[COL[col]].map(QUAL_RANK).fillna(0)
    return df


# ══════════════════════════════════════════════
# 2. HARD CONSTRAINT FILTER
# ══════════════════════════════════════════════
def filter_materials(df, constraints: dict):
    """
    constraints dict keys (all optional):
      max_density        float  g/cm³
      min_uts            float  MPa
      min_ys             float  MPa
      min_E              float  GPa
      min_elong          float  %
      max_cost           int    1-5
      min_max_temp       float  °C
      min_corr           str    'Poor'/'Moderate'/'Good'/'Excellent'
      min_mach           str    same
      min_weld           str    same
      category           str/list  'Metal','Polymer','Ceramic','Composite'
    """
    mask = pd.Series([True] * len(df), index=df.index)

    num_map = {
        "max_density":  (COL["density"],  "<="),
        "min_uts":      (COL["uts"],      ">="),
        "min_ys":       (COL["ys"],       ">="),
        "min_E":        (COL["E"],        ">="),
        "min_elong":    (COL["elong"],    ">="),
        "max_cost":     (COL["cost"],     "<="),
        "min_max_temp": (COL["max_temp"], ">="),
    }
    for key, (col, op) in num_map.items():
        if key in constraints and constraints[key] is not None:
            val = constraints[key]
            if op == "<=":
                mask &= df[col].fillna(9999) <= val
            else:
                mask &= df[col].fillna(0) >= val

    qual_map = {
        "min_corr": COL["corr"] + "_num",
        "min_mach": COL["mach"] + "_num",
        "min_weld": COL["weld"] + "_num",
    }
    for key, col in qual_map.items():
        if key in constraints and constraints[key] is not None:
            threshold = QUAL_RANK.get(constraints[key], 0)
            mask &= df[col] >= threshold

    if "category" in constraints and constraints["category"] is not None:
        cats = constraints["category"]
        if isinstance(cats, str):
            cats = [cats]
        mask &= df[COL["category"]].isin(cats)

    result = df[mask].copy()
    return result


# ══════════════════════════════════════════════
# 3. WEIGHTED SCORING & RANKING
# ══════════════════════════════════════════════
def score_and_rank(df, weights: dict):
    """
    weights dict — keys from scoring_cols below, values 0-10
    Higher weight = more important to the user.

    Available weight keys:
      strength, stiffness, lightness, ductility,
      corrosion, cost_efficiency, machinability,
      high_temp, electrical
    """
    if df.empty:
        return df

    scored = df.copy()

    def norm(series, invert=False):
        """Normalize a series 0→1. invert=True for 'lower is better'."""
        mn, mx = series.min(), series.max()
        if mx == mn:
            return pd.Series([0.5] * len(series), index=series.index)
        n = (series - mn) / (mx - mn)
        return 1 - n if invert else n

    scoring_map = {
        "strength":       norm(scored[COL["uts"]].fillna(0)),
        "stiffness":      norm(scored[COL["E"]].fillna(0)),
        "lightness":      norm(scored[COL["density"]].fillna(99), invert=True),
        "ductility":      norm(scored[COL["elong"]].fillna(0)),
        "corrosion":      norm(scored[COL["corr"] + "_num"]),
        "cost_efficiency":norm(scored[COL["cost"]].fillna(5), invert=True),
        "machinability":  norm(scored[COL["mach"] + "_num"]),
        "high_temp":      norm(scored[COL["max_temp"]].fillna(0)),
        "electrical":     norm(scored[COL["elec"]].fillna(0)),
    }

    total_weight = sum(weights.get(k, 0) for k in scoring_map)
    if total_weight == 0:
        scored["Score"] = 0
        scored["Rank"] = range(1, len(scored) + 1)
        return scored

    score_series = pd.Series([0.0] * len(scored), index=scored.index)
    for key, norm_series in scoring_map.items():
        w = weights.get(key, 0)
        score_series += (w / total_weight) * norm_series

    scored["Score"] = (score_series * 100).round(1)
    scored = scored.sort_values("Score", ascending=False).reset_index(drop=True)
    scored["Rank"] = range(1, len(scored) + 1)
    return scored


# ══════════════════════════════════════════════
# 4. PERFORMANCE INDEX CALCULATOR
# ══════════════════════════════════════════════
def calc_performance_indices(df):
    """
    Adds common Ashby performance indices as new columns.
    """
    d = df.copy()
    uts = d[COL["uts"]].fillna(0)
    ys  = d[COL["ys"]].fillna(0)
    E   = d[COL["E"]].fillna(0)
    rho = d[COL["density"]].replace(0, np.nan)

    d["PI: UTS/ρ (Specific Strength)"]    = (uts / rho).round(1)
    d["PI: E/ρ (Specific Stiffness)"]     = (E / rho).round(2)
    d["PI: UTS^(2/3)/ρ (Beam Lightness)"] = ((uts ** (2/3)) / rho).round(2)
    d["PI: E^(1/2)/ρ (Plate Stiffness)"]  = ((E ** 0.5) / rho).round(3)
    return d


# ══════════════════════════════════════════════
# 5. ASHBY CHART GENERATOR
# ══════════════════════════════════════════════
def ashby_chart(df, x_key="density", y_key="uts", 
                x_label=None, y_label=None,
                title=None, log_x=True, log_y=True,
                highlight_ids=None, save_path=None):
    """
    Generates an Ashby-style property chart.

    x_key, y_key: keys from COL dict
    highlight_ids: list of material IDs to highlight
    """
    x_col = COL[x_key]
    y_col = COL[y_key]

    plot_df = df.dropna(subset=[x_col, y_col]).copy()
    if plot_df.empty:
        print("No data to plot.")
        return

    fig, ax = plt.subplots(figsize=(12, 8))
    fig.patch.set_facecolor("#F8F9FA")
    ax.set_facecolor("#F8F9FA")

    legend_handles = []
    for cat, color in CAT_COLORS.items():
        subset = plot_df[plot_df[COL["category"]] == cat]
        if subset.empty:
            continue
        ax.scatter(subset[x_col], subset[y_col],
                   c=color, s=120, alpha=0.85, edgecolors="white",
                   linewidths=1.2, zorder=3, label=cat)
        for _, row in subset.iterrows():
            mid = highlight_ids and row[COL["id"]] in highlight_ids
            ax.annotate(
                row[COL["name"]].split(" ")[0] + "\n" + " ".join(row[COL["name"]].split(" ")[1:3]),
                xy=(row[x_col], row[y_col]),
                fontsize=6.5,
                ha="center", va="bottom",
                color="#222222",
                fontweight="bold" if mid else "normal",
                xytext=(0, 6), textcoords="offset points"
            )
        patch = mpatches.Patch(color=color, label=cat)
        legend_handles.append(patch)

    if highlight_ids:
        hi = plot_df[plot_df[COL["id"]].isin(highlight_ids)]
        ax.scatter(hi[x_col], hi[y_col],
                   s=280, facecolors="none", edgecolors="#FF0000",
                   linewidths=2.5, zorder=5, label="Top Ranked")
        legend_handles.append(mpatches.Patch(facecolor="none",
                                              edgecolor="#FF0000",
                                              label="Top Ranked",
                                              linewidth=2))

    if log_x: ax.set_xscale("log")
    if log_y: ax.set_yscale("log")

    ax.set_xlabel(x_label or x_col, fontsize=12, fontweight="bold")
    ax.set_ylabel(y_label or y_col, fontsize=12, fontweight="bold")
    ax.set_title(title or f"Ashby Chart: {y_col} vs {x_col}",
                 fontsize=14, fontweight="bold", pad=15)

    ax.grid(True, which="both", linestyle="--", alpha=0.4, color="#AAAAAA")
    ax.legend(handles=legend_handles, loc="upper left",
              framealpha=0.9, fontsize=10)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"  Chart saved → {save_path}")
    else:
        plt.show()
    plt.close()


# ══════════════════════════════════════════════
# 6. RESULTS PRINTER
# ══════════════════════════════════════════════
def print_results(ranked_df, top_n=5):
    cols_show = ["Rank", COL["id"], COL["name"], COL["category"],
                 COL["density"], COL["uts"], COL["E"],
                 COL["cost"], "Score"]
    available = [c for c in cols_show if c in ranked_df.columns]
    top = ranked_df.head(top_n)[available]

    print("\n" + "═"*70)
    print("  MatSelect AI — Top Recommendations")
    print("═"*70)
    print(top.to_string(index=False))
    print("═"*70)

    print("\n  Detailed View (Top 3):")
    for _, row in ranked_df.head(3).iterrows():
        print(f"\n  #{int(row['Rank'])}  {row[COL['name']]}  [{row[COL['id']]}]")
        print(f"      Category : {row[COL['category']]} / {row[COL['subcat']]}")
        print(f"      Density  : {row[COL['density']]} g/cm³")
        print(f"      UTS      : {row[COL['uts']]} MPa")
        print(f"      E        : {row[COL['E']]} GPa")
        print(f"      Cost     : {row[COL['cost']]}/5")
        print(f"      Score    : {row['Score']}/100")
        print(f"      Apps     : {row[COL['apps']]}")


# ══════════════════════════════════════════════
# 7. MAIN DEMO — CASE STUDY
# ══════════════════════════════════════════════
if __name__ == "__main__":

    print("\n" + "█"*70)
    print("  MatSelect AI — Stage 2: Engineering Logic Engine")
    print("  Case Study: Lightweight Structural Material for a Drone Frame")
    print("█"*70)

    # ── Load ──
    df = load_database()
    print(f"\n  Database loaded: {len(df)} materials\n")

    # ── Step 1: Hard Constraints ──
    print("  STEP 1 — Applying Hard Constraints:")
    print("    max density  : 4.0 g/cm³")
    print("    min UTS      : 150 MPa")
    print("    min E        : 5 GPa")
    print("    max cost     : 4")
    print("    min corrosion: Moderate")

    constraints = {
        "max_density": 4.0,
        "min_uts":     150,
        "min_E":       5,
        "max_cost":    4,
        "min_corr":    "Moderate",
    }
    filtered = filter_materials(df, constraints)
    print(f"\n  → {len(filtered)} materials passed the filter:")
    print("    " + ", ".join(filtered[COL["name"]].tolist()))

    # ── Step 2: Weighted Scoring ──
    print("\n  STEP 2 — Weighted Scoring:")
    weights = {
        "lightness":       9,   # top priority for drone
        "strength":        8,
        "stiffness":       6,
        "corrosion":       5,
        "cost_efficiency": 4,
        "ductility":       2,
    }
    for k, v in weights.items():
        print(f"    {k:<18}: {v}/10")

    ranked = score_and_rank(filtered, weights)
    print_results(ranked, top_n=5)

    # ── Step 3: Performance Indices ──
    print("\n  STEP 3 — Performance Indices (Top 5):")
    ranked_pi = calc_performance_indices(ranked)
    pi_cols = ["Rank", COL["name"],
               "PI: UTS/ρ (Specific Strength)",
               "PI: E/ρ (Specific Stiffness)"]
    print(ranked_pi.head(5)[pi_cols].to_string(index=False))

    # ── Step 4: Ashby Charts ──
    print("\n  STEP 4 — Generating Ashby Charts...")

    top_ids = ranked.head(3)[COL["id"]].tolist()

    ashby_chart(
        df,
        x_key="density", y_key="uts",
        x_label="Density (g/cm³)",
        y_label="Tensile Strength (MPa)",
        title="Ashby Chart: Strength vs Density\n(Drone Frame Case Study)",
        highlight_ids=top_ids,
        save_path="ashby_strength_density.png"
    )

    ashby_chart(
        df,
        x_key="density", y_key="E",
        x_label="Density (g/cm³)",
        y_label="Young's Modulus (GPa)",
        title="Ashby Chart: Stiffness vs Density\n(Drone Frame Case Study)",
        highlight_ids=top_ids,
        save_path="ashby_stiffness_density.png"
    )

    print("\n  ✅ Stage 2 Complete.")
    print("  Files generated:")
    print("    ashby_strength_density.png")
    print("    ashby_stiffness_density.png")
    print("█"*70 + "\n")
