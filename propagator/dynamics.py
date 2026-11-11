"""Equations of motion for the two-body problem.

The state is y = [x, y, z, vx, vy, vz] in km and km/s (ECI frame).
The derivative is  y_dot = [v, a],  with  a = -mu * r / |r|^3.
"""


import numpy as np
from propagator.constants import MU_KM3_S2, R_EARTH_KM


#---Acceleration-----------------------------------------------

def two_body_acceleration(r, mu=MU_KM3_S2):
    """Point-mass gravity acceleration (km/s^2) at position r (km)."""
    r_mag = np.linalg.norm(r)
    if r_mag == 0.0:
        raise ValueError("Position is at Earth's centre: acceleration undefined.")
    return -mu * r / r_mag**3


# --- State derivative --------------------------------------------------------

def two_body_derivative(t, y, mu=MU_KM3_S2):
    """Return y_dot = [v, a] for solve_ivp. t is unused but required."""
    r = y[:3]
    v = y[3:]
    a = two_body_acceleration(r, mu)
    return np.concatenate((v, a))


# --- Main: hand check --------------------------------------------------------

def main():
    r_iss_km = R_EARTH_KM + 400.0
    v_circ_km_s = np.sqrt(MU_KM3_S2 / r_iss_km)
    y = np.array([r_iss_km, 0.0, 0.0, 0.0, v_circ_km_s, 0.0])
    y_dot = two_body_derivative(0.0, y)
    a_m_s2 = np.linalg.norm(y_dot[3:]) * 1000.0
    print(f"State y      : {y}")
    print(f"Derivative   : {y_dot}")
    print(f"|a| at 400 km: {a_m_s2:.3f} m/s^2")


if __name__ == "__main__":
    main()