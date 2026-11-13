"""First numerical orbit: from orbital elements to a propagated trajectory.

For each orbit in data/initial_orbits.csv:
  elements -> coe_to_rv (toolkit) -> y0 -> propagate one period and one day
  -> ephemeris CSVs, three figures, and a printed table of physics checks.

Run from the repo root:  python -m examples.first_orbit
"""

import csv

import matplotlib.pyplot as plt
import numpy as np

from propagator.analysis import (
    angular_momentum, closure_error_km, max_relative_drift, plane_shift_deg,
    radius_km, specific_energy,
)
from propagator.constants import FIGURES_DIR, OUTPUT_DIR, R_EARTH_KM, REPO_DIR
from propagator.propagate import propagate
from toolkit.conversions import coe_to_rv
from toolkit.kepler import period_s

# --- Settings ------------------------------------------------------------------

ORBITS_CSV = REPO_DIR / "data" / "initial_orbits.csv"
ELEMENT_FIELDS = ["a_km", "e", "i_deg", "raan_deg", "argp_deg", "nu_deg"]
STATE_FIELDS = ["t_s", "x_km", "y_km", "z_km", "vx_km_s", "vy_km_s", "vz_km_s"]
STEP_S = 60.0
ONE_DAY_S = 86400.0


# --- 1. Read -------------------------------------------------------------------

def read_initial_orbits(path):
    """Return a list of {'name': str, 'a_km': float, ...} from the CSV."""
    with open(path, newline="") as f:
        return [{"name": row["name"], **{k: float(row[k]) for k in ELEMENT_FIELDS}}
                for row in csv.DictReader(f)]


# --- 2. Process ----------------------------------------------------------------

def initial_state(orbit):
    r, v = coe_to_rv(*(orbit[k] for k in ELEMENT_FIELDS))
    return np.concatenate((r, v))


def physics_checks(orbit, Y_period, Y_day):
    a, e = orbit["a_km"], orbit["e"]
    r_day = radius_km(Y_day)
    return {
        "closure_mm": closure_error_km(Y_period) * 1e6,
        "r_min_km": r_day.min(),
        "r_max_km": r_day.max(),
        "r_perigee_hand_km": a * (1 - e),
        "r_apogee_hand_km": a * (1 + e),
        "energy_drift": max_relative_drift(specific_energy(Y_day)),
        "h_drift": max_relative_drift(np.linalg.norm(angular_momentum(Y_day), axis=1)),
        "plane_shift_deg": plane_shift_deg(Y_day),
    }


def run_orbit(orbit):
    y0 = initial_state(orbit)
    T = period_s(orbit["a_km"])
    t_period, Y_period = propagate(y0, T, STEP_S)
    t_day, Y_day = propagate(y0, ONE_DAY_S, STEP_S)
    return {
        "orbit": orbit, "y0": y0, "period_s": T,
        "t_period": t_period, "Y_period": Y_period,
        "t_day": t_day, "Y_day": Y_day,
        "checks": physics_checks(orbit, Y_period, Y_day),
    }


# --- 3. Output -----------------------------------------------------------------

def write_ephemeris(t, Y, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(STATE_FIELDS)
        for t_s, state in zip(t, Y):
            writer.writerow([f"{t_s:.3f}"] + [f"{x:.9f}" for x in state])


def earth_sphere(n_lon=60, n_lat=30):
    u = np.linspace(0, 2 * np.pi, n_lon)
    w = np.linspace(0, np.pi, n_lat)
    return (R_EARTH_KM * np.outer(np.cos(u), np.sin(w)),
            R_EARTH_KM * np.outer(np.sin(u), np.sin(w)),
            R_EARTH_KM * np.outer(np.ones_like(u), np.cos(w)))


def plot_orbit_3d(result, path):
    Y, name = result["Y_period"], result["orbit"]["name"]
    fig = plt.figure(figsize=(7, 7))
    ax = fig.add_subplot(projection="3d")
    ax.plot_surface(*earth_sphere(), color="tab:blue", alpha=0.3, linewidth=0)
    ax.plot(Y[:, 0], Y[:, 1], Y[:, 2], color="tab:red", lw=1.5, label="orbit (1 period)")
    ax.scatter(*Y[0, :3], color="k", s=30, label="start")
    ax.set_xlabel("x (km)")
    ax.set_ylabel("y (km)")
    ax.set_zlabel("z (km)")
    ax.set_aspect("equal")
    ax.set_title(f"{name}: orbit in ECI")
    ax.legend()
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_orbit_projections(result, path):
    Y, name = result["Y_period"], result["orbit"]["name"]
    planes = [(0, 1, "x", "y"), (0, 2, "x", "z"), (1, 2, "y", "z")]
    fig, axes = plt.subplots(1, 3, figsize=(13, 4.5), layout="constrained")
    for ax, (i, j, li, lj) in zip(axes, planes):
        ax.add_patch(plt.Circle((0, 0), R_EARTH_KM, color="tab:blue", alpha=0.3))
        ax.plot(Y[:, i], Y[:, j], color="tab:red", lw=1.5)
        ax.plot(Y[0, i], Y[0, j], "ko", ms=5)
        ax.set_xlabel(f"{li} (km)")
        ax.set_ylabel(f"{lj} (km)")
        ax.set_title(f"{li}-{lj} plane")
        ax.set_aspect("equal")
        ax.grid(alpha=0.3)
    fig.suptitle(f"{name}: orbit projected onto the ECI planes")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_altitude(result, path):
    c, name = result["checks"], result["orbit"]["name"]
    t_h = result["t_day"] / 3600
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(t_h, radius_km(result["Y_day"]) - R_EARTH_KM, lw=1)
    ax.axhline(c["r_perigee_hand_km"] - R_EARTH_KM, ls="--", color="gray", label="perigee (hand)")
    ax.axhline(c["r_apogee_hand_km"] - R_EARTH_KM, ls=":", color="gray", label="apogee (hand)")
    ax.set_xlabel("Time (h)")
    ax.set_ylabel("Altitude (km)")
    ax.set_title(f"{name}: altitude over one day ({ONE_DAY_S / result['period_s']:.2f} orbits)")
    ax.grid(alpha=0.3)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.15), ncol=2)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def print_checks(result):
    o, c = result["orbit"], result["checks"]
    print(f"\n{o['name']}  (a = {o['a_km']} km, e = {o['e']}, i = {o['i_deg']} deg)")
    print(f"  y0                 {np.array2string(result['y0'], precision=6, suppress_small=True)}")
    print(f"  period             {result['period_s']:.1f} s ({result['period_s'] / 60:.2f} min)")
    print(f"  closure, 1 period  {c['closure_mm']:.2f} mm")
    print(f"  r range, 1 day     {c['r_min_km']:.3f} - {c['r_max_km']:.3f} km"
          f"   (hand {c['r_perigee_hand_km']:.3f} - {c['r_apogee_hand_km']:.3f})")
    print(f"  energy drift       {c['energy_drift']:.1e}")
    print(f"  |h| drift          {c['h_drift']:.1e}")
    print(f"  plane shift        {c['plane_shift_deg']:.1e} deg")


# --- Main ----------------------------------------------------------------------

def main():
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    for orbit in read_initial_orbits(ORBITS_CSV):
        result = run_orbit(orbit)
        name = orbit["name"]
        write_ephemeris(result["t_period"], result["Y_period"], OUTPUT_DIR / f"{name}_1_period.csv")
        write_ephemeris(result["t_day"], result["Y_day"], OUTPUT_DIR / f"{name}_1_day.csv")
        plot_orbit_3d(result, FIGURES_DIR / f"{name}_orbit_3d.png")
        plot_orbit_projections(result, FIGURES_DIR / f"{name}_orbit_projections.png")
        plot_altitude(result, FIGURES_DIR / f"{name}_altitude.png")
        print_checks(result)


if __name__ == "__main__":
    main()