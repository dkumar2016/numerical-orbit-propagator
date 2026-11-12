"""Test problems with known exact solutions, used to validate integrators.

All derivatives use the solve_ivp signature f(t, y).
"""

import numpy as np


# --- Exponential growth:  y' = y,  exact y = y0 * e^t ------------------------

def growth_derivative(t, y):
    return y


def growth_exact(t, y0=1.0):
    return y0 * np.exp(t)


# --- Harmonic oscillator:  x'' = -x,  state [x, v] ---------------------------

def oscillator_derivative(t, y):
    x, v = y
    return np.array([v, -x])


def oscillator_exact(t):
    """Exact state from y0 = [1, 0]: x = cos t, v = -sin t."""
    return np.array([np.cos(t), -np.sin(t)])


def oscillator_energy(Y):
    """Energy 0.5 (x^2 + v^2) for one state (2,) or a history (n, 2)."""
    Y = np.atleast_2d(Y)
    return 0.5 * (Y[:, 0]**2 + Y[:, 1]**2)