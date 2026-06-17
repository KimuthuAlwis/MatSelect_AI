# 🔬 MatSelect AI — Intelligent Materials Selection System

MatSelect AI is a computational engineering platform that bridges **Materials Science with Artificial Intelligence** to automate structural design constraint matching. Utilizing **Ashby’s Materials Selection Methodology**, the system programmatically parses natural language requirements into multi-variant quantitative constraint thresholds across metals, polymers, technical ceramics, and composites.

Live Web Application Link: `https://matselectai-ndkwffrkkr8mbw3sypkfvb.streamlit.app/`

---

## 🛠️ Engineering Architecture & Logic Flow

The application bypasses manual property chart lookups by executing a real-time programmatic filtering and ranking pipeline:

$$\text{Natural Language Query} \longrightarrow \text{Gemini 2.5 Flash (JSON Parameter Extraction)} \longrightarrow \text{Pandas Constraint Engine} \longrightarrow \text{Multi-Attribute Performance Scoring}$$

1. **AI Semantic Parser:** Maps qualitative engineering terms (e.g., *"ultra-lightweight"*, *"high-temperature structural fatigue resistance"*) into quantitative physical limits.
2. **Hard Constraint Engine:** Executes a vector pass over a 20-property material database, filtering candidate material indices.
3. **Multi-Attribute Utility Theory (MAUT):** Calculates normalized merit scores ($S$) based on user-defined priority weighting values ($w_i$).

---

## 📈 Implemented Engineering Performance Indices (Ashby Framework)

To evaluate structural efficiency accurately under specific loading states, the platform calculates and charts performance metrics derived from physical principles:

### 1. Specific Strength (Centrifugal/Tension Optimization)
For components undergoing high inertial or tensile stresses where mass drives failure (e.g., rotating machinery, high-RPM flywheels):
$$M_1 = \frac{\sigma_{ts}}{\rho}$$
Where $\sigma_{ts}$ is Tensile Strength (MPa) and $\rho$ is Density ($\text{g/cm}^3$).

### 2. Specific Stiffness (Lightweight Minimal-Flex Deflection)
For optimizing structural beams where weight reduction must be achieved while maintaining rigidity under bending loads (e.g., aerospace structural panels, robotics arms, vehicle frames):
$$M_2 = \frac{E^{1/2}}{\rho}$$
Where $E$ is Young's Modulus (GPa).

---

## 🎯 Validation Case Studies

### Case Study A: Aerospace Drone Frame Optimization
* **Input Query:** *"I need an ultra-lightweight material for a drone frame arm that won't flex under high motor torque loads."*
* **Programmatic Filter Target:** Low Density ($\rho \le 2.5 \text{ g/cm}^3$), High Modulus ($E \ge 70 \text{ GPa}$).
* **Engine Output:** Programmatically prioritizes **Carbon Fiber Reinforced Polymers (CFRP)** and specialized advanced composites on interactive Plotly Ashby charts.

### Case Study B: Severe Environment Marine Structural Components
* **Input Query:** *"High-strength structural fastners for load-bearing marine structures exposed to saltwater environment."*
* **Programmatic Filter Target:** High UTS ($\sigma_{ts} \ge 400 \text{ MPa}$), High Corrosion Resistance (`Excellent`).
* **Engine Output:** Filters out structural carbon steels; ranks **Titanium Alloys (Ti-6Al-4V)** and Marine-Grade Stainless Steels (316L) to the peak of the performance hierarchy.

---

## 🗂️ Database Coverage & Properties Tracked
The system evaluates technical material properties across 4 primary engineering families (**Metals, Polymers, Ceramics, Composites**) using standardized properties from reference indices (ASM Handbook, MatWeb, CES EduPack):
* **Physical Properties:** Density ($\rho$)
* **Mechanical Properties:** Tensile Strength ($\sigma_{ts}$), Young's Modulus ($E$), Hardness (HV), Elongation ($\%$)
* **Thermal Properties:** Maximum Service Temperature ($^\circ\text{C}$), Thermal Conductivity ($k$)
* **Economic/Environmental:** Relative Cost Index (1–5 Scale), Local Environmental Degradation Ratings

---

## 🚀 Future Engineering Roadmap
* **Multi-Objective Pareto Optimality:** Implementation of 2D frontier plotting to identify exact non-dominated trade-off spaces between competing properties (e.g., Strength vs. Cost).
* **Chemical Environment Degradation Matrices:** Expanding the AI parser to predict stress-corrosion cracking (SCC) risk under specific acidic, marine, or highly oxidizing operating variables.
* **Eco-Auditing & LCA Tracking:** Integration of embodied energy (MJ/kg) and $\text{CO}_2$ primary production carbon tracking indices.
