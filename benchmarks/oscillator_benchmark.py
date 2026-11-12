"""Integrator benchmark on the harmonic oscillator (exact solution known).

Studies:
1. Error vs step size for Euler and RK4 -> measured order of accuracy.
2. Energy history over several periods -> Euler's energy growth.
3. solve_ivp tolerance sweep, RK45 vs DOP853 -> error vs cost.

Run from the repo root:  python -m benchmarks.oscillator_benchmark
"""

import csv

import matplotlib.pyplot as plt
import numpy as np
from scipy.integrate import solve_ivp

from propagator.constants import FIGURES_DIR, OUTPUT_DIR
from propagator.integrators import euler_step, integrate_fixed, rk4_step
from propagator.reference_problems import (
    oscillator_derivative, oscillator_energy, oscillator_exact,
)

# --- Settings ------------------------------------------------------------------

Y0 = np.array([1.0, 0.0])
ONE_PERIOD_S = 2 * np.pi
ENERGY_RUN_S = 5 * ONE_PERIOD_S
STEP_SIZES = [0.1, 0.05, 0.025, 0.0125]
ENERGY_STEP = 0.1
TOLERANCES = [1e-3, 1e-6, 1e-9, 1e-12]
ATOL_FACTOR = 1e-3                     # atol = rtol * ATOL_FACTOR
FIXED_METHODS = {"Euler": euler_step, "RK4": rk4_step}
CALLS_PER_STEP = {"Euler": 1, "RK4": 4}
SOLVE_IVP_METHODS = ["RK45", "DOP853"]


# --- Helpers -------------------------------------------------------------------

def final_error(y_end, t_end):
    return float(np.linalg.norm(y_end - oscillator_exact(t_end)))


def measured_order(step_sizes, errors):
    slope, _ = np.polyfit(np.log(step_sizes), np.log(errors), 1)
    return float(slope)


# --- 2. Process ----------------------------------------------------------------

def run_step_sweep():
    """Error after one period for each fixed-step method and step size."""
    rows = []
    for name, step in FIXED_METHODS.items():
        for h in STEP_SIZES:
            t, Y = integrate_fixed(oscillator_derivative, (0.0, ONE_PERIOD_S), Y0, h, step)
            rows.append({
                "method": name,
                "h_s": h,
                "n_calls": CALLS_PER_STEP[name] * (len(t) - 1),
                "final_error": final_error(Y[-1], t[-1]),
            })
    return rows


def run_energy_history():
    """Energy vs time for each fixed-step method at one step size."""
    histories = {}
    for name, step in FIXED_METHODS.items():
        t, Y = integrate_fixed(oscillator_derivative, (0.0, ENERGY_RUN_S), Y0, ENERGY_STEP, step)
        histories[name] = (t, oscillator_energy(Y))
    return histories


def run_tolerance_sweep():
    """Error and cost of solve_ivp over the tolerance range."""
    rows = []
    for method in SOLVE_IVP_METHODS:
        for rtol in TOLERANCES:
            sol = solve_ivp(oscillator_derivative, (0.0, ENERGY_RUN_S), Y0,
                            method=method, rtol=rtol, atol=rtol * ATOL_FACTOR)
            if not sol.success:
                raise RuntimeError(f"{method} failed at rtol={rtol}: {sol.message}")
            rows.append({
                "method": method,
                "rtol": rtol,
                "n_calls": sol.nfev,
                "final_error": final_error(sol.y[:, -1], sol.t[-1]),
            })
    return rows


def orders_from(step_rows):
    orders = {}
    for name in FIXED_METHODS:
        rows = [r for r in step_rows if r["method"] == name]
        orders[name] = measured_order([r["h_s"] for r in rows], [r["final_error"] for r in rows])
    return orders


# --- 3. Output -----------------------------------------------------------------

def write_csv(rows, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def plot_error_vs_step(step_rows, orders, path):
    fig, ax = plt.subplots(figsize=(7, 5))
    for name in FIXED_METHODS:
        rows = [r for r in step_rows if r["method"] == name]
        ax.loglog([r["h_s"] for r in rows], [r["final_error"] for r in rows],
                  "o-", label=f"{name} (measured order {orders[name]:.2f})")
    ax.set_xlabel("Step size h (s)")
    ax.set_ylabel("Error after one period")
    ax.set_title("Fixed-step integrators on the harmonic oscillator")
    ax.grid(True, which="both", alpha=0.3)
    ax.legend()
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_energy(histories, path):
    fig, ax = plt.subplots(figsize=(7, 5))
    for name, (t, energy) in histories.items():
        ax.plot(t, energy, label=name)
    ax.axhline(oscillator_energy(Y0)[0], color="k", ls="--", lw=1, label="Exact (constant)")
    ax.set_yscale("log")
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Energy 0.5 (x² + v²)")
    ax.set_title(f"Energy over 5 periods, h = {ENERGY_STEP} s")
    ax.grid(True, which="both", alpha=0.3)
    ax.legend()
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_tolerance_sweep(tol_rows, path):
    fig, ax = plt.subplots(figsize=(7, 5))
    for method in SOLVE_IVP_METHODS:
        rows = [r for r in tol_rows if r["method"] == method]
        ax.loglog([r["n_calls"] for r in rows], [r["final_error"] for r in rows],
                  "o-", label=method)
    ax.set_xlabel("Calls to f (cost)")
    ax.set_ylabel("Error after 5 periods")
    ax.set_title("solve_ivp: error vs cost over rtol = 1e-3 ... 1e-12")
    ax.grid(True, which="both", alpha=0.3)
    ax.legend()
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def print_summary(step_rows, orders, histories, tol_rows):
    print("Error after one period (fixed step)")
    print(f"{'method':<7}{'h (s)':>9}{'calls':>8}{'error':>12}")
    for r in step_rows:
        print(f"{r['method']:<7}{r['h_s']:>9.4f}{r['n_calls']:>8}{r['final_error']:>12.2e}")
    print("\nMeasured order: " + ", ".join(f"{k} {v:.2f}" for k, v in orders.items()))
    print(f"\nEnergy after 5 periods, h = {ENERGY_STEP} s (exact 0.5)")
    for name, (_, energy) in histories.items():
        print(f"  {name:<6}{energy[-1]:.6f}")
    print("\nsolve_ivp tolerance sweep (5 periods)")
    print(f"{'method':<8}{'rtol':>8}{'calls':>8}{'error':>12}")
    for r in tol_rows:
        print(f"{r['method']:<8}{r['rtol']:>8.0e}{r['n_calls']:>8}{r['final_error']:>12.2e}")


# --- Main ----------------------------------------------------------------------

def main():
    step_rows = run_step_sweep()
    orders = orders_from(step_rows)
    histories = run_energy_history()
    tol_rows = run_tolerance_sweep()

    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    write_csv(step_rows, OUTPUT_DIR / "integrator_step_sweep.csv")
    write_csv(tol_rows, OUTPUT_DIR / "solve_ivp_tolerance_sweep.csv")
    plot_error_vs_step(step_rows, orders, FIGURES_DIR / "integrator_error_vs_step.png")
    plot_energy(histories, FIGURES_DIR / "integrator_energy_drift.png")
    plot_tolerance_sweep(tol_rows, FIGURES_DIR / "solve_ivp_tolerance_sweep.png")
    print_summary(step_rows, orders, histories, tol_rows)


if __name__ == "__main__":
    main()