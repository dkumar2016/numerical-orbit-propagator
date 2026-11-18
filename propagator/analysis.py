"""Checks on a propagated trajectory Y with shape (n, 6), one row per time.

All functions are vectorised: they work on every row at once.
"""

import numpy as np

from propagator.constants import DAY_S, J2, MU_KM3_S2, R_EARTH_KM


def radius_km(Y):
    return np.linalg.norm(Y[:, :3], axis=1)


def speed_km_s(Y):
    return np.linalg.norm(Y[:, 3:], axis=1)


def specific_energy(Y, mu=MU_KM3_S2):
    """epsilon = v^2/2 - mu/r (km^2/s^2) at every time."""
    return speed_km_s(Y)**2 / 2 - mu / radius_km(Y)


def angular_momentum(Y):
    """h = r x v (km^2/s) at every time, shape (n, 3)."""
    return np.cross(Y[:, :3], Y[:, 3:])


def max_relative_drift(values):
    """Largest |value - first| / |first| over the trajectory."""
    values = np.asarray(values, dtype=float)
    return float(np.max(np.abs(values - values[0]) / np.abs(values[0])))


def plane_shift_deg(Y):
    """Largest angle (deg) between the orbit normal h_hat and its initial value."""
    h = angular_momentum(Y)
    h_hat = h / np.linalg.norm(h, axis=1, keepdims=True)
    cos_angle = np.clip(h_hat @ h_hat[0], -1.0, 1.0)
    return float(np.degrees(np.arccos(cos_angle)).max())


def closure_error_km(Y):
    """Distance between the final and initial positions (km)."""
    return float(np.linalg.norm(Y[-1, :3] - Y[0, :3]))


def ric_components(Y_ref, Y):
    """Position of Y relative to Y_ref in the reference's RIC frame (km), shape (n, 3).

    Columns: radial (along r), in-track (completes the set, ~ along v),
    cross-track (along h = r x v). Same frame as RSW / RTN.
    """
    r, v = Y_ref[:, :3], Y_ref[:, 3:]
    R_hat = r / np.linalg.norm(r, axis=1, keepdims=True)
    h = np.cross(r, v)
    C_hat = h / np.linalg.norm(h, axis=1, keepdims=True)
    I_hat = np.cross(C_hat, R_hat)
    d = Y[:, :3] - r
    return np.column_stack((np.sum(d * R_hat, axis=1),
                            np.sum(d * I_hat, axis=1),
                            np.sum(d * C_hat, axis=1)))


def raan_deg(Y):
    """Right ascension of the ascending node (deg) at every time, unwrapped.

    From the orbit normal h = r x v:  RAAN = atan2(h_x, -h_y).
    Unwrapped so it changes smoothly past 0/360 deg.
    """
    h = angular_momentum(Y)
    return np.degrees(np.unwrap(np.arctan2(h[:, 0], -h[:, 1])))


def drift_rate_per_day(t, values):
    """Slope of a straight-line fit of values against t (s), per day."""
    return float(np.polyfit(t, values, 1)[0] * DAY_S)


def j2_raan_rate_deg_per_day(a_km, e, i_deg, mu=MU_KM3_S2, j2=J2, r_eq_km=R_EARTH_KM):
    """Analytic (secular) RAAN drift from J2:  -1.5 n J2 (R/p)^2 cos i, in deg/day."""
    n = np.sqrt(mu / a_km**3)
    p = a_km * (1 - e**2)
    rate_rad_s = -1.5 * n * j2 * (r_eq_km / p)**2 * np.cos(np.radians(i_deg))
    return float(np.degrees(rate_rad_s) * DAY_S)