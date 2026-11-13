"""Numerical orbit propagation with scipy's solve_ivp (DOP853).

Settings follow the integrator benchmark: an 8th-order adaptive method with
tight tolerances. Output is returned on a regular time grid, one row per time.
"""

import numpy as np
from scipy.integrate import solve_ivp

from propagator.constants import MU_KM3_S2
from propagator.dynamics import two_body_derivative

METHOD = "DOP853"
RTOL = 1e-10
ATOL = 1e-12


# --- Helpers -------------------------------------------------------------------

def output_times(duration_s, step_s):
    """Regular grid 0, step, 2*step, ... that always ends exactly at duration_s."""
    if duration_s <= 0:
        raise ValueError(f"Duration must be positive, got {duration_s} s.")
    if step_s <= 0:
        raise ValueError(f"Output step must be positive, got {step_s} s.")
    t = np.arange(0.0, duration_s, step_s)
    t = t[t < duration_s - 1e-9 * step_s]       # drop a near-duplicate of the end
    return np.append(t, duration_s)


# --- Propagation ---------------------------------------------------------------

def propagate(y0, duration_s, step_s=60.0, mu=MU_KM3_S2, rtol=RTOL, atol=ATOL):
    """Propagate a state [x, y, z, vx, vy, vz] (km, km/s) under two-body gravity.

    Returns t with shape (n,) in seconds from the start, and Y with shape
    (n, 6): one row per output time (the transpose of solve_ivp's sol.y).
    """
    y0 = np.asarray(y0, dtype=float)
    if y0.shape != (6,):
        raise ValueError(f"State must have 6 elements, got shape {y0.shape}.")
    t_out = output_times(duration_s, step_s)
    sol = solve_ivp(two_body_derivative, (0.0, duration_s), y0, method=METHOD,
                    rtol=rtol, atol=atol, t_eval=t_out, args=(mu,))
    if not sol.success:
        raise RuntimeError(f"Propagation failed: {sol.message}")
    return sol.t, sol.y.T