# bezier-circle-approximation
Python reproduction package for an IB Math AA HL IA on numerical study of polynomial Bézier circle-arc approximation: least-squares control point fitting, max radial error evaluation, and semi-log decay modelling.

# Bézier Arc Radial Error (IB Math AA HL IA)

This repository contains the reproducibility code and generated outputs for my IB Mathematics: Analysis & Approaches HL Internal Assessment.

**Topic:** Approximating a quarter-circle arc using a single degree-\(n\) polynomial Bézier curve and measuring the maximum radial error  

The script fits control points (under endpoint, tangent, and symmetry constraints) via linear least squares and then evaluates \(E_n\) on a dense parameter grid. A semi-log plot is used to test the near-exponential decay of error with increasing degree.

---

## Files
- `bezier_script.py` — main reproduction script (generates tables + figures)
- `table1_E_n_results.csv` — results table (if included/exported)
- `outputs/` — generated figures and CSV outputs (created after running)

---

## Requirements
- Python 3.10+ (recommended)
- `numpy`
- `pandas`
- `matplotlib`

Install dependencies:
```bash
pip install numpy pandas matplotlib
