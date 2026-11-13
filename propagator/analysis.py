"""Checks on a propagated trajectory Y with shape (n, 6), one row per time.

All functions are vectorised: they work on every row at once.
"""

import numpy as np

from propagator.constants import MU_KM3_S2


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