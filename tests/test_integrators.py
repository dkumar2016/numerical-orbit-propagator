"""Tests for propagator.integrators, using problems with exact solutions."""

import numpy as np
import pytest

from propagator.dynamics import two_body_derivative
from propagator.integrators import euler_step, integrate_fixed, rk4_step
from propagator.reference_problems import (
    growth_derivative, growth_exact, oscillator_derivative,
    oscillator_energy, oscillator_exact,
)

Y0_OSC = np.array([1.0, 0.0])
ORBIT_STATE = np.array([6778.137, 0.0, 0.0, 0.0, 7.668558, 0.0])


# --- Single steps: hand calculations from Block 1 ------------------------------

def test_euler_step_growth_is_one_plus_h():
    y1 = euler_step(growth_derivative, 0.0, np.array([1.0]), 0.1)
    assert y1[0] == pytest.approx(1.1, abs=1e-15)


def test_rk4_step_growth_matches_exponential():
    h = 0.1
    y1 = rk4_step(growth_derivative, 0.0, np.array([1.0]), h)
    assert y1[0] == pytest.approx(1.1051708, abs=1e-7)          # HC 6
    assert abs(y1[0] - growth_exact(h)) < h**5                  # local error ~ h^5


def test_rk4_step_oscillator_matches_hand_calculation():
    y1 = rk4_step(oscillator_derivative, 0.0, Y0_OSC, 0.1)
    assert y1 == pytest.approx([0.9950042, -0.0998333], abs=1e-7)  # HC 7


@pytest.mark.parametrize("step", [euler_step, rk4_step])
def test_step_works_on_orbit_state_without_mutation(step):
    y = ORBIT_STATE.copy()
    y1 = step(two_body_derivative, 0.0, y, 10.0)
    assert y1.shape == (6,)
    assert np.array_equal(y, ORBIT_STATE)


# --- integrate_fixed --------------------------------------------------------------

def test_integrate_fixed_grid_and_shape():
    t, Y = integrate_fixed(oscillator_derivative, (0.0, 0.3), Y0_OSC, 0.1, euler_step)
    assert t == pytest.approx([0.0, 0.1, 0.2, 0.3])
    assert Y.shape == (4, 2)
    assert np.array_equal(Y[0], Y0_OSC)


def test_integrate_fixed_reproduces_euler_table():
    _, Y = integrate_fixed(oscillator_derivative, (0.0, 0.3), Y0_OSC, 0.1, euler_step)
    expected = [[1.0, 0.0], [1.0, -0.1], [0.99, -0.2], [0.97, -0.299]]   # HC 2
    assert Y.ravel() == pytest.approx(np.ravel(expected), abs=1e-12)


def test_integrate_fixed_rejects_bad_step():
    with pytest.raises(ValueError):
        integrate_fixed(oscillator_derivative, (0.0, 1.0), Y0_OSC, 0.0, euler_step)


# --- Energy behaviour -------------------------------------------------------------

def test_euler_energy_grows_by_one_plus_h_squared_each_step():
    h, n = 0.1, 50
    _, Y = integrate_fixed(oscillator_derivative, (0.0, n * h), Y0_OSC, h, euler_step)
    energy = oscillator_energy(Y)
    assert energy[-1] == pytest.approx(0.5 * (1 + h**2)**n, rel=1e-12)


def test_rk4_energy_nearly_conserved():
    _, Y = integrate_fixed(oscillator_derivative, (0.0, 10 * np.pi), Y0_OSC, 0.1, rk4_step)
    assert oscillator_energy(Y)[-1] == pytest.approx(0.5, abs=1e-5)


# --- Convergence order --------------------------------------------------------------

@pytest.mark.parametrize("step, expected_order, step_sizes", [
    (euler_step, 1.0, [0.01, 0.005, 0.0025]),
    (rk4_step, 4.0, [0.1, 0.05, 0.025]),
])
def test_measured_order(step, expected_order, step_sizes):
    T = 2 * np.pi
    errors = []
    for h in step_sizes:
        _, Y = integrate_fixed(oscillator_derivative, (0.0, T), Y0_OSC, h, step)
        errors.append(np.linalg.norm(Y[-1] - oscillator_exact(T)))
    slope, _ = np.polyfit(np.log(step_sizes), np.log(errors), 1)
    assert slope == pytest.approx(expected_order, abs=0.1)