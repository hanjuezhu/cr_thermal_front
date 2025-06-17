This repository contains code and data for reproducing figures and results from:

> **Zhu, H., Zweibel, E. G., & Gnedin, N. Y.** (2025).  
> *Cosmic Ray Mediated Thermal Fronts in the Warm-Hot Circumgalactic Medium*  
> **The Astrophysical Journal**.

---

## 📂 Repository Structure

```
cr_thermal_front/
├── data/                  # ion fraction table and observational line ratio data
│   ├── tab2.txt
│   └── wakker.txt
├── scripts/               # Plotting scripts for profiles and diagnostics
│   ├── plot_profiles.py
│   ├── plot_components.py
│   └── plot_line_ratios.py
├── src/                   # Core numerical solver and physical functions
│   ├── solver.py
│   └── physics.py
├── display_plots.ipynb    # Jupyter notebook for interactive plot rendering
└── README.md              # This file
```

---

## 📈 Usage

### 1. Solve and Visualize Thermal Front Profiles

```bash
python scripts/plot_profiles.py
```

This generates figures for number density, temperature, and pressure across static and evaporative fronts (as in Figures 1–4 of the paper).

### 2. Generate Line Ratio Diagnostics

```bash
python scripts/plot_line_ratios.py
```

This will produce a two-panel figure comparing model predictions of:
- **log [N(SiIV)/N(CIV)] vs log [N(CIV)/N(OVI)]**
- **log [N(NV)/N(OVI)] vs log [N(CIV)/N(OVI)]**

with observations from Wakker et al. (2009).

### 3. Interactive Exploration

Open the Jupyter notebook:

```bash
jupyter notebook display_plots.ipynb
```

to re-generate and explore figures interactively.

---

## 📄 Data Sources

- `tab2.txt`: ion fraction table
- `wakker.txt`: Line ratio measurements compiled from Wakker et al. (2009), used for observational comparison

---
