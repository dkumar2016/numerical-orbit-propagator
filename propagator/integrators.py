"""Fixed-step integrators for y' = f(t, y).

Used to understand and benchmark integration error. Production propagation
uses scipy.integrate.solve_ivp (DOP853) instead.
"""

import numpy as np


# --- Single steps --------------------------------------------------------------

def euler_step(f, t, y, h):
    """One forward-Euler step (order 1, one call to f)."""
    return y + h * f(t, y)


def rk4_step(f, t, y, h):
    """One classical Runge-Kutta step (order 4, four calls to f)."""
    k1 = f(t, y)
    k2 = f(t + h / 2, y + h / 2 * k1)
    k3 = f(t + h / 2, y + h / 2 * k2)
    k4 = f(t + h, y + h * k3)
    return y + h / 6 * (k1 + 2 * k2 + 2 * k3 + k4)


# --- Fixed-step integration ----------------------------------------------------

def integrate_fixed(f, t_span, y0, h, step):
    """Integrate from t_span[0] to t_span[1] with a fixed step.

    The number of steps is rounded so the grid ends exactly at t_span[1].
    Returns t with shape (n + 1,) and Y with shape (n + 1, len(y0)):
    one row per time (note: solve_ivp returns the transpose).
    """
    t0, tf = t_span
    if h <= 0:
        raise ValueError(f"Step size must be positive, got h = {h}.")
    n_steps = max(1, int(round((tf - t0) / h)))
    t = np.linspace(t0, tf, n_steps + 1)
    Y = np.zeros((n_steps + 1, len(y0)))
    Y[0] = y0
    for k in range(n_steps):
        Y[k + 1] = step(f, t[k], Y[k], t[k + 1] - t[k])
    return t, Y