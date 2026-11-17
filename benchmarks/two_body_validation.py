"""Validation of the two-body propagator against the exact Kepler solution.

For each orbit in data/initial_orbits.csv:
1. Position error vs the exact solution over 10 days, split into radial,
   in-track and cross-track (RIC).
2. Conservation: relative drift of energy and |h| over the same run.
3. Tolerance sweep: error after one day and calls to f, rtol 1e-6 ... 1e-12.
The printed summary compares the 10-day in-track error with the value
predicted from the energy drift (0.75 * v * k * t^2).

Run from the repo root:  python -m benchmarks.two_body_validation
"""

import csv

import matplotlib.pyplot as plt
import numpy as np

from propagator.analysis import angular_momentum, ric_components, specific_energy
from propagator.constants import DATA_DIR, DAY_S, FIGURES_DIR, OUTPUT_DIR
from propagator.orbits import initial_state, read_initial_orbits
from propagator.propagate import RTOL, propagate
from propagator.reference_problems import two_body_exact

# --- Settings ------------------------------------------------------------------

ORBITS_CSV = DATA_DIR / "initial_orbits.csv"
RUN_DAYS = 10
STEP_S = 120.0
SWEEP_TOLERANCES = [1e-6, 1e-7, 1e-8, 1e-9, 1e-10, 1e-11, 1e-12]
ATOL_FACTOR = 1e-2                     # atol = rtol * ATOL_FACTOR in the sweep
LOG_FLOOR_M = 1e-6                     # errors of exactly 0 cannot go on a log axis


# --- Helpers -------------------------------------------------------------------

def relative_series(values):
    """(value - first) / |first| at every time."""
    values = np.asarray(values, dtype=float)
    return (values - values[0]) / abs(values[0])


# --- Process -------------------------------------------------------------------

def run_error_history(orbit):
    """Propagate RUN_DAYS and compare with the exact solution at every output time."""
    y0 = initial_state(orbit)
    t, Y = propagate(y0, RUN_DAYS * DAY_S, STEP_S)
    Y_exact = two_body_exact(orbit, t)
    ric_m = ric_components(Y_exact, Y) * 1000
    energy_drift = relative_series(specific_energy(Y))
    h_drift = relative_series(np.linalg.norm(angular_momentum(Y), axis=1))
    k_per_s = np.polyfit(t, energy_drift, 1)[0]
    speed_km_s = np.linalg.norm(y0[3:])
    return {
        "t": t,
        "error_m": np.linalg.norm(ric_m, axis=1),
        "ric_m": ric_m,
        "energy_drift": energy_drift,
        "h_drift": h_drift,
        "predicted_in_track_m": 0.75 * speed_km_s * abs(k_per_s) * t[-1]**2 * 1000,
    }


def run_tolerance_sweep(orbit):
    """Error after one day and calls to f for each tolerance."""
    y0 = initial_state(orbit)
    exact_end = two_body_exact(orbit, [DAY_S])[0]
    rows = []
    for rtol in SWEEP_TOLERANCES:
        t, Y, nfev = propagate(y0, DAY_S, DAY_S, rtol=rtol, atol=rtol * ATOL_FACTOR,
                               return_nfev=True)
        rows.append({
            "orbit": orbit["name"],
            "rtol": rtol,
            "n_calls": nfev,
            "error_1_day_m": float(np.linalg.norm(Y[-1, :3] - exact_end[:3]) * 1000),
        })
    return rows


def summarise(orbit, history):
    final = history["ric_m"][-1]
    return {
        "orbit": orbit["name"],
        "days": RUN_DAYS,
        "error_m": float(history["error_m"][-1]),
        "radial_m": float(final[0]),
        "in_track_m": float(final[1]),
        "cross_track_m": float(final[2]),
        "predicted_in_track_m": float(history["predicted_in_track_m"]),
        "max_energy_drift": float(np.max(np.abs(history["energy_drift"]))),
        "max_h_drift": float(np.max(np.abs(history["h_drift"]))),
    }


# --- Output --------------------------------------------------------------------

def write_csv(rows, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def plot_error_vs_time(histories, path):
    names = list(histories)
    fig, axes = plt.subplots(1, len(names), figsize=(6.5 * len(names), 4.8),
                             layout="constrained", squeeze=False)
    for ax, name in zip(axes[0], names):
        h = histories[name]
        t_days = h["t"] / DAY_S
        ax.semilogy(t_days, np.maximum(h["error_m"], LOG_FLOOR_M), "k", lw=1.6, label="total")
        for col, label in enumerate(["radial", "in-track", "cross-track"]):
            ax.semilogy(t_days, np.maximum(np.abs(h["ric_m"][:, col]), LOG_FLOOR_M),
                        lw=1, label=label)
        ax.set_title(f"{name}: error vs exact Kepler solution")
        ax.set_xlabel("Time (days)")
        ax.set_ylabel("Position error (m)")
        ax.grid(True, which="both", alpha=0.3)
        ax.legend(fontsize=8)
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_conservation(histories, path):
    fig, ax = plt.subplots(figsize=(8, 4.8), layout="constrained")
    for name, h in histories.items():
        t_days = h["t"] / DAY_S
        ax.semilogy(t_days, np.abs(h["energy_drift"]) + 1e-16, lw=1, label=f"{name}: energy")
        ax.semilogy(t_days, np.abs(h["h_drift"]) + 1e-16, lw=1, ls="--", label=f"{name}: |h|")
    ax.set_title("Conservation over 10 days (two-body: both should be constant)")
    ax.set_xlabel("Time (days)")
    ax.set_ylabel("Relative drift")
    ax.grid(True, which="both", alpha=0.3)
    ax.legend(fontsize=8)
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_tolerance_sweep(sweep_rows, path):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 4.8), layout="constrained")
    for name in dict.fromkeys(r["orbit"] for r in sweep_rows):
        rows = [r for r in sweep_rows if r["orbit"] == name]
        errors = [r["error_1_day_m"] for r in rows]
        ax1.loglog([r["rtol"] for r in rows], errors, "o-", label=name)
        ax2.loglog([r["n_calls"] for r in rows], errors, "o-", label=name)
    ax1.invert_xaxis()
    ax1.axvline(RTOL, color="gray", ls=":", label=f"default rtol {RTOL:.0e}")
    ax1.set(title="Error after one day vs tolerance", xlabel="rtol (tighter →)",
            ylabel="Position error (m)")
    ax2.set(title="Cost of accuracy", xlabel="Calls to f", ylabel="Position error (m)")
    for ax in (ax1, ax2):
        ax.grid(True, which="both", alpha=0.3)
        ax.legend(fontsize=8)
    fig.savefig(path, dpi=150)
    plt.close(fig)


def print_summary(summaries, sweep_rows):
    print(f"Error vs exact Kepler solution after {RUN_DAYS} days (rtol {RTOL:.0e})")
    print(f"{'orbit':<14}{'total m':>10}{'radial':>10}{'in-track':>10}{'cross':>10}"
          f"{'predicted':>11}{'energy':>10}{'|h|':>10}")
    for s in summaries:
        print(f"{s['orbit']:<14}{s['error_m']:>10.3f}{s['radial_m']:>10.3f}{s['in_track_m']:>10.3f}"
              f"{s['cross_track_m']:>10.1e}{s['predicted_in_track_m']:>11.3f}"
              f"{s['max_energy_drift']:>10.1e}{s['max_h_drift']:>10.1e}")
    print("\nTolerance sweep: error after 1 day (m) and calls to f")
    for r in sweep_rows:
        print(f"  {r['orbit']:<14} rtol {r['rtol']:.0e}  {r['error_1_day_m']:>12.4f} m  {r['n_calls']:>6} calls")


# --- Main ----------------------------------------------------------------------

def main():
    orbits = read_initial_orbits(ORBITS_CSV)
    histories = {o["name"]: run_error_history(o) for o in orbits}
    summaries = [summarise(o, histories[o["name"]]) for o in orbits]
    sweep_rows = [row for o in orbits for row in run_tolerance_sweep(o)]

    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    write_csv(summaries, OUTPUT_DIR / "validation_summary.csv")
    write_csv(sweep_rows, OUTPUT_DIR / "validation_tolerance_sweep.csv")
    plot_error_vs_time(histories, FIGURES_DIR / "validation_error_vs_time.png")
    plot_conservation(histories, FIGURES_DIR / "validation_conservation.png")
    plot_tolerance_sweep(sweep_rows, FIGURES_DIR / "validation_tolerance_sweep.png")
    print_summary(summaries, sweep_rows)


if __name__ == "__main__":
    main()