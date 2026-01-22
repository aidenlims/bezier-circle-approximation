
import os, math
from dataclasses import dataclass
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

DEG_MIN, DEG_MAX = 3, 12
M_SAMPLES = 600
N_DENSE   = 4000
OUTDIR    = "outputs"
os.makedirs(OUTDIR, exist_ok=True)

def bernstein_basis(n, t):
    t = np.asarray(t)
    B = np.empty((t.size, n+1), dtype=float)
    for k in range(n+1):
        B[:, k] = math.comb(n, k) * (t**k) * ((1 - t)**(n - k))
    return B

def eval_curve_from_controls(n, xs, ys, t):
    B = bernstein_basis(n, t)
    x = B @ xs
    y = B @ ys
    r = np.sqrt(x*x + y*y)
    return x, y, r

def max_radial_error(n, xs, ys, t_dense):
    _, _, r = eval_curve_from_controls(n, xs, ys, t_dense)
    e = np.abs(r - 1.0)
    idx = int(np.argmax(e))
    return float(e[idx]), float(t_dense[idx]), float(r[idx]-1.0)

from dataclasses import dataclass
@dataclass
class FixedSpec:
    n: int
    P0: np.ndarray
    P1: np.ndarray
    Pn_1: np.ndarray
    Pn: np.ndarray
    P2: np.ndarray = None
    Pn_2: np.ndarray = None

def fixed_spec(n, curvature_match=False):
    P0 = np.array([1.0, 0.0])
    Pn = np.array([0.0, 1.0])
    P1 = np.array([1.0, math.pi/(2*n)])
    Pn_1 = np.array([math.pi/(2*n), 1.0])
    P2 = None; Pn_2 = None
    if curvature_match and n >= 4:
        rpp0 = np.array([-(math.pi**2)/4.0, 0.0])
        rpp1 = np.array([0.0, -(math.pi**2)/4.0])
        P2 = 2*P1 - P0 + rpp0/(n*(n-1))
        Pn_2 = rpp1/(n*(n-1)) - Pn + 2*Pn_1
    return FixedSpec(n, P0, P1, Pn_1, Pn, P2, Pn_2)

def build_linear_system(n, t_samples, curvature_match=False):
    B = bernstein_basis(n, t_samples)
    m = t_samples.size
    theta = (math.pi/2) * t_samples
    tx = np.cos(theta); ty = np.sin(theta)
    spec = fixed_spec(n, curvature_match)
    xs_fixed = np.zeros(n+1); ys_fixed = np.zeros(n+1)
    xs_fixed[0], ys_fixed[0] = spec.P0
    xs_fixed[1], ys_fixed[1] = spec.P1
    xs_fixed[n-1], ys_fixed[n-1] = spec.Pn_1
    xs_fixed[n], ys_fixed[n] = spec.Pn
    fixed_idx = {0,1,n-1,n}
    if curvature_match and (spec.P2 is not None):
        xs_fixed[2], ys_fixed[2] = spec.P2
        xs_fixed[n-2], ys_fixed[n-2] = spec.Pn_2
        fixed_idx.update({2, n-2})
    var_list = []
    for k in range(2, n-1):
        if k in fixed_idx or (n-k) in fixed_idx:
            continue
        if k > n-k:
            continue
        if k < n-k:
            var_list.append(('pair', k))
        else:
            var_list.append(('mid', k))
    num_vars = sum(2 if tag=='pair' else 1 for tag,_ in var_list)
    M = np.zeros((2*m, num_vars), dtype=float)
    rhs = np.zeros(2*m, dtype=float)
    known_x = B @ xs_fixed
    known_y = B @ ys_fixed
    rhs[:m] = tx - known_x
    rhs[m:] = ty - known_y
    col = 0
    for tag, k in var_list:
        if tag == 'pair':
            a_col = np.zeros(2*m); b_col = np.zeros(2*m)
            a_col[:m] = B[:, k];   b_col[:m] = B[:, n-k]
            a_col[m:] = B[:, n-k]; b_col[m:] = B[:, k]
            M[:, col] = a_col; col += 1
            M[:, col] = b_col; col += 1
        else:
            c_col = np.zeros(2*m)
            c_col[:m] = B[:, k]
            c_col[m:] = B[:, k]
            M[:, col] = c_col; col += 1
    return M, rhs, var_list, xs_fixed, ys_fixed

def solve_controls(n, t_samples, curvature_match=False):
    M, rhs, var_list, xs_fixed, ys_fixed = build_linear_system(n, t_samples, curvature_match)
    sol, *_ = np.linalg.lstsq(M, rhs, rcond=None)
    xs = xs_fixed.copy(); ys = ys_fixed.copy()
    si = 0
    for tag, k in var_list:
        if tag == 'pair':
            a = sol[si]; b = sol[si+1]; si += 2
            xs[k] = a; ys[k] = b
            xs[n-k] = b; ys[n-k] = a
        else:
            c = sol[si]; si += 1
            xs[k] = c; ys[k] = c
    return xs, ys

def main():
    t_samples = np.linspace(0, 1, M_SAMPLES)
    t_dense   = np.linspace(0, 1, N_DENSE)

    records_plain = []
    records_curv  = []
    examples = {}

    for n in range(DEG_MIN, DEG_MAX+1):
        xs, ys = solve_controls(n, t_samples, curvature_match=False)
        E, t_star, sgn = max_radial_error(n, xs, ys, t_dense)
        records_plain.append({"n": n, "E_n": E, "t_at_max": t_star, "signed_e": sgn})
        if n in (3,5,8):
            examples[n] = (xs, ys)

        if n >= 4:
            xs2, ys2 = solve_controls(n, t_samples, curvature_match=True)
            E2, t_star2, sgn2 = max_radial_error(n, xs2, ys2, t_dense)
            records_curv.append({"n": n, "E_n_curv": E2})

    df_plain = pd.DataFrame(records_plain)
    df_curv  = pd.DataFrame(records_curv)
    df = pd.merge(df_plain, df_curv, on="n", how="left")
    df["E_mm_R10"] = 10.0 * df["E_n"]

    csv_path = os.path.join(OUTDIR, "table1_E_n_results.csv")
    df.to_csv(csv_path, index=False)

    tex_path = os.path.join(OUTDIR, "table1_E_n_results.tex")
    lines = []
    lines.append("\\begin{tabular}{r r r r}\\toprule\n")
    lines.append("$n$ & $E_n$ & $t_{\\max}$ & $10\\,\\mathrm{mm}\\cdot E_n$ (mm) \\\\\\midrule\n")
    for _, row in df.sort_values("n").iterrows():
        lines.append(f"{int(row['n'])} & {row['E_n']:.3e} & {row['t_at_max']:.3f} & {row['E_mm_R10']:.3e} \\\\\n")
    lines.append("\\bottomrule\\end{tabular}\n")
    with open(tex_path, "w") as f:
        f.writelines(lines)

    # Semi-log decay plot (clean labels + readable fit box)
    n_vals = df_plain["n"].values.astype(float)
    E_vals = df_plain["E_n"].values.astype(float)

    A = np.vstack([n_vals, np.ones_like(n_vals)]).T
    beta, *_ = np.linalg.lstsq(A, np.log(E_vals), rcond=None)
    b = float(beta[0])   # slope
    a = float(beta[1])   # intercept

    # 95% CI for slope b (same math as before)
    yhat = A @ beta
    residuals = np.log(E_vals) - yhat
    sigma2 = (residuals**2).sum() / (len(E_vals) - A.shape[1])
    cov = sigma2 * np.linalg.inv(A.T @ A)
    se_b = math.sqrt(cov[0, 0])
    ci_low = b - 1.96 * se_b
    ci_high = b + 1.96 * se_b

    plt.figure()

    # Data + fit with human-readable legend labels
    plt.semilogy(n_vals, E_vals, 'o', label=r"Computed $E_n$ (least-squares fit)")

    n_fit = np.linspace(n_vals.min(), n_vals.max(), 200)
    E_fit = np.exp(a + b * n_fit)
    plt.semilogy(n_fit, E_fit, '-', label="Exponential model")

    plt.xlabel("Degree $n$")
    plt.ylabel(r"Maximum radial error $E_n$")
    plt.title(r"Semi-log plot of $E_n$ vs degree")

    # Put the detailed equation in a compact annotation box (NOT in the legend)
    fit_text = (
        r"$\ln(E_n)=a+bn$" "\n"
        rf"$a={a:.2f}$" "\n"
        rf"$b={b:.3f}$" "\n"
        rf"$95\%~CI~for~b:[{ci_low:.3f},{ci_high:.3f}]$"
    )
    plt.text(
    0.95, 0.95, fit_text,
    transform=plt.gca().transAxes,
    va="top", ha="right",
    bbox=dict(boxstyle="round", facecolor="white", alpha=0.85, edgecolor="0.7")
    )

    plt.legend()
    semi_path = os.path.join(OUTDIR, "semilog_En_vs_n.png")
    plt.savefig(semi_path, dpi=220, bbox_inches="tight")
    plt.close()


    # Log–log view
    plt.figure()
    plt.loglog(n_vals, E_vals, 'o', label="data")
    A2 = np.vstack([np.log(n_vals), np.ones_like(n_vals)]).T
    beta2, *_ = np.linalg.lstsq(A2, np.log(E_vals), rcond=None)
    alpha = beta2[0]; logC = beta2[1]
    n_fit2 = np.linspace(n_vals.min(), n_vals.max(), 200)
    E_fit2 = np.exp(logC) * (n_fit2**alpha)
    plt.loglog(n_fit2, E_fit2, label=f"fit: log E = {alpha:.3f} log n + {logC:.2f}")
    plt.xlabel("Degree n (log)")
    plt.ylabel("Max radial error $E_n$ (log)")
    plt.title("Log–log view of $E_n$ vs $n$")
    plt.legend()
    loglog_path = os.path.join(OUTDIR, "loglog_En_vs_n.png")
    plt.savefig(loglog_path, dpi=220, bbox_inches="tight")
    plt.close()

    # Radial error illustration for n=8
    n_ex = 8
    xs_ex, ys_ex = examples[n_ex]
    t_dense = np.linspace(0, 1, N_DENSE)
    x, y, r = eval_curve_from_controls(n_ex, xs_ex, ys_ex, t_dense)
    theta = (math.pi/2) * t_dense
    cx = np.cos(theta); cy = np.sin(theta)
    E_ex, t_star_ex, signed_ex = max_radial_error(n_ex, xs_ex, ys_ex, t_dense)
    j = int(round(t_star_ex*(len(t_dense)-1)))
    px, py = x[j], y[j]

    plt.figure()
    plt.plot(cx, cy, label="Quarter circle")
    plt.plot(x, y, label=f"Bézier (n={n_ex})")
    plt.scatter([px], [py])
    plt.plot([0, px], [0, py])
    plt.axis('equal')
    plt.xlabel("x"); plt.ylabel("y")
    plt.title(f"Radial error illustration (n={n_ex})\\nMax |e_r| ≈ {E_ex:.3e} at t ≈ {t_star_ex:.3f}")
    radial_fig = os.path.join(OUTDIR, "radial_error_n8.png")
    plt.savefig(radial_fig, dpi=220, bbox_inches="tight")
    plt.close()

    # Control polygons & curves for n=3,5,8
    plt.figure()
    theta_full = np.linspace(0, math.pi/2, 500)
    plt.plot(np.cos(theta_full), np.sin(theta_full), label="Quarter circle")
    for n_plot in (3,5,8):
        xs_p, ys_p = examples[n_plot]
        plt.plot(xs_p, ys_p, '--', label=f"Control polygon (n={n_plot})")
        t = np.linspace(0,1,800)
        x_p, y_p, _ = eval_curve_from_controls(n_plot, xs_p, ys_p, t)
        plt.plot(x_p, y_p, label=f"Bézier (n={n_plot})")
    plt.axis('equal')
    plt.xlabel("x"); plt.ylabel("y")
    plt.title("Control polygons and Bézier curves")
    polygons_fig = os.path.join(OUTDIR, "control_polygons.png")
    plt.savefig(polygons_fig, dpi=220, bbox_inches="tight")
    plt.close()

    # Plain vs curvature-matched comparison
    plt.figure()
    plt.semilogy(df["n"], df["E_n"], 'o-', label="Plain (endpoints+tangents)")
    plt.semilogy(df["n"], df["E_n_curv"], 's--', label="With endpoint curvature match")
    plt.xlabel("Degree n")
    plt.ylabel("Max radial error $E_n")
    plt.title("Plain vs curvature-matched Bézier approximation")
    plt.legend()
    compare_fig = os.path.join(OUTDIR, "compare_plain_vs_curv.png")
    plt.savefig(compare_fig, dpi=220, bbox_inches="tight")
    plt.close()

    print("Saved outputs in", OUTDIR)

if __name__ == "__main__":
    main()
