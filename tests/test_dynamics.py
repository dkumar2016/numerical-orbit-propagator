"""Tests for propagator.dynamics."""

import numpy as np
import pytest

from propagator.constants import MU_KM3_S2, R_EARTH_KM
from propagator.dynamics import two_body_acceleration, two_body_derivative

ISS_STATE = np.array([6774.611, 0.0, 0.0, 0.0, 4.761539, 6.016191])
OFF_AXIS_STATE = np.array([6000.0, 3000.0, 2000.0, 1.0, 2.0, 3.0])


# --- Magnitude ----------------------------

@pytest.mark.parametrize("r_km, expected_m_s2", [
    (R_EARTH_KM, 9.798),            # surface
    (R_EARTH_KM + 400.0, 8.676),    # ISS altitude
    (42164.0, 0.224),               # GEO
])
def test_magnitude_matches_mu_over_r_squared(r_km, expected_m_s2):
    a = two_body_acceleration(np.array([r_km, 0.0, 0.0]))
    assert np.linalg.norm(a) * 1000.0 == pytest.approx(expected_m_s2, abs=1e-3)


def test_magnitude_off_axis_state():
    a = two_body_acceleration(OFF_AXIS_STATE[:3])
    assert np.linalg.norm(a) * 1000.0 == pytest.approx(8.1347, abs=1e-4)


# --- Direction -------------------------------------------------------------

def test_acceleration_points_toward_earth_centre():
    r = OFF_AXIS_STATE[:3]
    a = two_body_acceleration(r)
    cos_angle = np.dot(a, r) / (np.linalg.norm(a) * np.linalg.norm(r))
    assert cos_angle == pytest.approx(-1.0, abs=1e-12)


# --- Inverse square --------------------------------------------------------

@pytest.mark.parametrize("factor", [2.0, 3.0, 10.0])
def test_inverse_square(factor):
    r = OFF_AXIS_STATE[:3]
    a_near = np.linalg.norm(two_body_acceleration(r))
    a_far = np.linalg.norm(two_body_acceleration(factor * r))
    assert a_near / a_far == pytest.approx(factor**2, rel=1e-12)


# --- Derivative structure ------------------------------------------------------

def test_derivative_shape():
    assert two_body_derivative(0.0, ISS_STATE).shape == (6,)


def test_velocity_passes_through():
    y_dot = two_body_derivative(0.0, OFF_AXIS_STATE)
    assert np.array_equal(y_dot[:3], OFF_AXIS_STATE[3:])


def test_derivative_matches_hand_calculation():
    y_dot = two_body_derivative(0.0, ISS_STATE)
    expected = [0.0, 4.761539, 6.016191, -0.0086850, 0.0, 0.0]
    assert y_dot == pytest.approx(expected, abs=1e-7)


def test_derivative_does_not_change_input():
    y = ISS_STATE.copy()
    two_body_derivative(0.0, y)
    assert np.array_equal(y, ISS_STATE)


def test_time_does_not_matter():
    early = two_body_derivative(0.0, ISS_STATE)
    late = two_body_derivative(86400.0, ISS_STATE)
    assert np.array_equal(early, late)


def test_mu_is_passed_through():
    y_dot = two_body_derivative(0.0, ISS_STATE, mu=2 * MU_KM3_S2)
    expected = 2 * two_body_derivative(0.0, ISS_STATE)[3:]
    assert y_dot[3:] == pytest.approx(expected, rel=1e-12)


# --- Errors ----------------------------------------------------------------

def test_zero_position_raises():
    with pytest.raises(ValueError):
        two_body_acceleration(np.zeros(3))