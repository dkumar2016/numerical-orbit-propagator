"""Equations of motion: accelerations (forces) and the state derivative.

The state is y = [x, y, z, vx, vy, vz] in km and km/s (ECI frame).
The derivative is  y_dot = [v, a],  where a is the sum of the chosen forces.
Forces are plug-ins: each is listed by name in FORCES.
"""

import numpy as np

from propagator.constants import J2, MU_KM3_S2, R_EARTH_KM


# --- Accelerations -----------------------------------------------------------

def two_body_acceleration(r, mu=MU_KM3_S2):
    """Point-mass gravity acceleration (km/s^2) at position r (km)."""
    r_mag = np.linalg.norm(r)
    if r_mag == 0.0:
        raise ValueError("Position is at Earth's centre: acceleration undefined.")
    return -mu * r / r_mag**3

def j2_acceleration(r, mu=MU_KM3_S2, j2=J2, r_eq_km=R_EARTH_KM):
    """Acceleration (km/s^2) from Earth's equatorial bulge (J2), ECI position r (km)."""
    x, y, z = r
    r_mag = np.linalg.norm(r)
    factor = -1.5 * j2 * mu * r_eq_km**2 / r_mag**5
    z2_r2 = z**2 / r_mag**2
    return factor * np.array([x * (1 - 5 * z2_r2),
                              y * (1 - 5 * z2_r2),
                              z * (3 - 5 * z2_r2)])


# --- Force registry ------------------------------------------------------------
# Every entry has the same call signature (t, r, v, mu) -> acceleration (km/s^2),
# so forces that need time or velocity (drag, Sun/Moon) can be added later
# without changing the derivative.

FORCES = {
    "two_body": lambda t, r, v, mu: two_body_acceleration(r, mu),
    "j2": lambda t, r, v, mu: j2_acceleration(r, mu),
}
DEFAULT_FORCES = ("two_body",)


def check_forces(forces):
    """Raise ValueError if no force is given or a name is not in FORCES."""
    if len(forces) == 0:
        raise ValueError("Give at least one force, e.g. ('two_body',).")
    unknown = [name for name in forces if name not in FORCES]
    if unknown:
        raise ValueError(f"Unknown force(s) {unknown}; choose from {sorted(FORCES)}")


# --- State derivative --------------------------------------------------------

def derivative(t, y, forces=DEFAULT_FORCES, mu=MU_KM3_S2):
    """Return y_dot = [v, sum of the chosen accelerations] for solve_ivp."""
    r = y[:3]
    v = y[3:]
    a = sum(FORCES[name](t, r, v, mu) for name in forces)
    return np.concatenate((v, a))


def two_body_derivative(t, y, mu=MU_KM3_S2):
    """Two-body only. Kept for the integrator benchmark and Day 11 tests."""
    return derivative(t, y, DEFAULT_FORCES, mu)


# --- Main: hand check --------------------------------------------------------

def main():
    np.set_printoptions(precision=6, suppress=True)
    r_iss_km = R_EARTH_KM + 400.0
    v_circ_km_s = np.sqrt(MU_KM3_S2 / r_iss_km)
    y = np.array([r_iss_km, 0.0, 0.0, 0.0, v_circ_km_s, 0.0])
    y_dot = derivative(0.0, y)
    a_m_s2 = np.linalg.norm(y_dot[3:]) * 1000.0
    print(f"State y      : {y}")
    print(f"Derivative   : {y_dot}")
    print(f"|a| at 400 km: {a_m_s2:.3f} m/s^2")


if __name__ == "__main__":
    main()