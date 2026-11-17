"""Test problems with known exact solutions, used to validate integrators
and the orbit propagator.
"""

import numpy as np
from propagator.orbits import elements_of
from toolkit.kepler import propagate as kepler_propagate


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

# --- Two-body motion:  exact solution from Kepler's equation ----------------

def two_body_exact(orbit, t):
    """Exact two-body states at the times t (s), shape (len(t), 6).

    Uses kepler.propagate from orbit-state-vector-toolkit (Kepler's equation),
    which shares no code with the numerical propagator.
    """
    Y = np.zeros((len(t), 6))
    for k, t_s in enumerate(t):
        r, v = kepler_propagate(*elements_of(orbit), t_s)
        Y[k] = np.concatenate((r, v))
    return Y