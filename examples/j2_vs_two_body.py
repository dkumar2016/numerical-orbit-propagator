"""J2 perturbation: two-body vs two-body + J2 from the same initial state.

For each orbit in data/initial_orbits.csv, propagate RUN_DAYS with
forces=("two_body",) and forces=("two_body", "j2"), then:
  - fit the RAAN drift rate and compare with the analytic J2 rate,
  - measure how far apart the two predictions drift (RIC split),
  - draw the orbit plane rotating under J2.

Run from the repo root:  python -m examples.j2_vs_two_body
"""

import csv

import matplotlib.pyplot as plt
import numpy as np

from propagator.analysis import (
    drift_rate_per_day, j2_raan_rate_deg_per_day, per_orbit_max, raan_deg, ric_components,
)
from propagator.constants import DATA_DIR, DAY_S, FIGURES_DIR, OUTPUT_DIR, R_EARTH_KM
from propagator.orbits import initial_state, read_initial_orbits
from propagator.propagate import propagate
from toolkit.kepler import period_s

# --- Settings ------------------------------------------------------------------

ORBITS_CSV = DATA_DIR / "initial_orbits.csv"
RUN_DAYS = 10
STEP_S = 120.0
TWO_BODY = ("two_body",)
WITH_J2 = ("two_body", "j2")
RING_DAYS = [0, 5, 10]                 # days at which to draw one orbit in 3D


# --- Process -------------------------------------------------------------------

def compare_force_models(orbit):
    y0 = initial_state(orbit)
    t, Y_2b = propagate(y0, RUN_DAYS * DAY_S, STEP_S, forces=TWO_BODY)
    _, Y_j2 = propagate(y0, RUN_DAYS * DAY_S, STEP_S, forces=WITH_J2)
    raan_2b, raan_j2 = raan_deg(Y_2b), raan_deg(Y_j2)
    measured = drift_rate_per_day(t, raan_j2)
    analytic = j2_raan_rate_deg_per_day(orbit["a_km"], orbit["e"], orbit["i_deg"])
    ric_km = ric_components(Y_2b, Y_j2)
    return {
        "orbit": orbit, "t": t, "Y_2b": Y_2b, "Y_j2": Y_j2,
        "raan_2b": raan_2b, "raan_j2": raan_j2,
        "difference_km": np.linalg.norm(ric_km, axis=1), "ric_km": ric_km,
        "summary": {
            "orbit": orbit["name"],
            "days": RUN_DAYS,
            "raan_rate_measured_deg_day": measured,
            "raan_rate_analytic_deg_day": analytic,
            "difference_percent": 100 * (measured - analytic) / analytic,
            "raan_rate_two_body_deg_day": drift_rate_per_day(t, raan_2b),
            "separation_after_1_day_km": float(np.interp(DAY_S, t, np.linalg.norm(ric_km, axis=1))),
            "max_separation_km": float(np.linalg.norm(ric_km, axis=1).max()),
        },
    }


# --- Output --------------------------------------------------------------------

def write_csv(rows, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def plot_raan(results, path):
    fig, axes = plt.subplots(1, len(results), figsize=(6.5 * len(results), 4.6),
                             layout="constrained", squeeze=False)
    for ax, res in zip(axes[0], results):
        s, t_days = res["summary"], res["t"] / DAY_S
        ax.plot(t_days, res["raan_2b"], label="two-body (no drift)")
        ax.plot(t_days, res["raan_j2"], lw=1, label=f"with J2: {s['raan_rate_measured_deg_day']:.3f} °/day")
        ax.plot(t_days, res["raan_j2"][0] + s["raan_rate_analytic_deg_day"] * t_days, "k--",
                label=f"analytic J2: {s['raan_rate_analytic_deg_day']:.3f} °/day")
        ax.set(title=f"{s['orbit']}: RAAN drift", xlabel="Time (days)", ylabel="RAAN (deg)")
        ax.grid(alpha=0.3)
        ax.legend(fontsize=8)
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_separation(results, path):
    fig, axes = plt.subplots(1, len(results), figsize=(6.5 * len(results), 4.6),
                             layout="constrained", squeeze=False)
    for ax, res in zip(axes[0], results):
        t, T = res["t"], period_s(res["orbit"]["a_km"])
        ends, total = per_orbit_max(t, res["difference_km"], T)
        ax.semilogy(ends / DAY_S, total, "k", lw=2, label="total")
        for col, label in enumerate(["radial", "in-track", "cross-track"]):
            ends, largest = per_orbit_max(t, res["ric_km"][:, col], T)
            ax.semilogy(ends / DAY_S, largest, lw=1.4, label=label)
        ax.set(title=f"{res['summary']['orbit']}: two-body prediction vs J2 prediction",
               xlabel="Time (days)", ylabel="Largest separation in each orbit (km)")
        ax.grid(True, which="both", alpha=0.3)
        ax.legend(fontsize=8)
    fig.suptitle("Same initial state, propagated with and without J2: "
                 "how far off a two-body prediction would be", fontsize=11)
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_plane_rotation(res, path):
    t, T = res["t"], period_s(res["orbit"]["a_km"])
    n_per_orbit = int(round(T / STEP_S)) + 1
    u, w = np.linspace(0, 2 * np.pi, 40), np.linspace(0, np.pi, 20)
    earth = (R_EARTH_KM * np.outer(np.cos(u), np.sin(w)),
             R_EARTH_KM * np.outer(np.sin(u), np.sin(w)),
             R_EARTH_KM * np.outer(np.ones_like(u), np.cos(w)))
    panels = [("two-body only", res["Y_2b"], res["raan_2b"]),
              ("two-body + J2", res["Y_j2"], res["raan_j2"])]
    fig = plt.figure(figsize=(13, 6.5), layout="constrained")
    for col, (label, Y, raan) in enumerate(panels, start=1):
        ax = fig.add_subplot(1, 2, col, projection="3d")
        ax.plot_surface(*earth, color="tab:blue", alpha=0.25, linewidth=0)
        for day in RING_DAYS:
            k0 = min(int(np.searchsorted(t, day * DAY_S)), len(t) - n_per_orbit)
            ax.plot(*Y[k0:k0 + n_per_orbit, :3].T, lw=1.5, label=f"day {day}: RAAN {raan[k0]:.1f}°")
        ax.set(xlabel="x (km)", ylabel="y (km)", zlabel="z (km)")
        ax.set_aspect("equal")
        ax.view_init(elev=25, azim=-60)
        ax.set_title(label)
        ax.legend(fontsize=8, loc="upper left")
    fig.suptitle(f"{res['summary']['orbit']}: one orbit drawn on days {RING_DAYS}, same initial state")
    fig.savefig(path, dpi=150)
    plt.close(fig)


def print_summary(rows):
    print(f"RAAN drift over {RUN_DAYS} days: measured (fit) vs analytic J2 rate")
    print(f"{'orbit':<14}{'measured':>11}{'analytic':>11}{'diff %':>9}{'two-body':>11}"
          f"{'sep 1 day km':>14}{'max sep km':>12}")
    for s in rows:
        print(f"{s['orbit']:<14}{s['raan_rate_measured_deg_day']:>11.4f}{s['raan_rate_analytic_deg_day']:>11.4f}"
              f"{s['difference_percent']:>9.2f}{s['raan_rate_two_body_deg_day']:>11.1e}"
              f"{s['separation_after_1_day_km']:>14.1f}{s['max_separation_km']:>12.1f}")


# --- Main ----------------------------------------------------------------------

def main():
    results = [compare_force_models(o) for o in read_initial_orbits(ORBITS_CSV)]
    rows = [r["summary"] for r in results]

    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    write_csv(rows, OUTPUT_DIR / "j2_raan_rates.csv")
    plot_raan(results, FIGURES_DIR / "j2_raan_drift.png")
    plot_separation(results, FIGURES_DIR / "two_body_error_from_ignoring_j2.png")
    plot_plane_rotation(results[0], FIGURES_DIR / f"{rows[0]['orbit']}_j2_plane_rotation.png")
    print_summary(rows)


if __name__ == "__main__":
    main()