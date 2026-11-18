"""Tests for the plug-in force model and the J2 perturbation."""

import numpy as np
import pytest

from propagator.analysis import drift_rate_per_day, j2_raan_rate_deg_per_day, raan_deg
from propagator.constants import DATA_DIR, DAY_S
from propagator.dynamics import (
    FORCES, derivative, j2_acceleration, two_body_acceleration, two_body_derivative,
)
from propagator.orbits import initial_state, read_initial_orbits
from propagator.propagate import propagate

ISS_STATE = np.array([6774.611, 0.0, 0.0, 0.0, 4.761539368, 6.016190934])
R_ISS_KM = 6778.0
ORBITS = read_initial_orbits(DATA_DIR / "initial_orbits.csv")


# --- The refactor changed nothing for two-body ---------------------------------

def test_two_body_is_registered():
    assert "two_body" in FORCES


def test_derivative_with_two_body_matches_old_function():
    assert np.array_equal(derivative(0.0, ISS_STATE, ("two_body",)),
                          two_body_derivative(0.0, ISS_STATE))


def test_propagate_default_is_two_body():
    _, Y_default = propagate(ISS_STATE, 3600.0)
    _, Y_named = propagate(ISS_STATE, 3600.0, forces=("two_body",))
    assert np.array_equal(Y_default, Y_named)


def test_forces_given_as_a_list_also_work():
    _, Y_tuple = propagate(ISS_STATE, 600.0, forces=("two_body",))
    _, Y_list = propagate(ISS_STATE, 600.0, forces=["two_body"])
    assert np.array_equal(Y_tuple, Y_list)


# --- Bad force lists ----------------------------------------------------------------

@pytest.mark.parametrize("forces", [("drag",), ("two_body", "typo"), ()])
def test_propagate_rejects_bad_force_lists(forces):
    with pytest.raises(ValueError):
        propagate(ISS_STATE, 600.0, forces=forces)


# --- J2 acceleration on known geometry (Block 1, HC 1) ------------------------

def test_j2_is_registered():
    assert "j2" in FORCES


def test_j2_over_equator_points_inward_at_12_5_mm_s2():
    a = j2_acceleration(np.array([R_ISS_KM, 0.0, 0.0]))
    assert a * 1e6 == pytest.approx([-12.48, 0.0, 0.0], abs=0.01)     # mm/s^2


def test_j2_over_pole_points_outward_twice_as_strong():
    a_pole = j2_acceleration(np.array([0.0, 0.0, R_ISS_KM]))
    a_equator = j2_acceleration(np.array([R_ISS_KM, 0.0, 0.0]))
    assert a_pole[2] > 0
    assert np.linalg.norm(a_pole) == pytest.approx(2 * np.linalg.norm(a_equator), rel=1e-12)


def test_j2_is_about_700_times_smaller_than_two_body():
    r = np.array([R_ISS_KM, 0.0, 0.0])
    ratio = np.linalg.norm(two_body_acceleration(r)) / np.linalg.norm(j2_acceleration(r))
    assert 600 < ratio < 800


def test_forces_add_up():
    y_dot_both = derivative(0.0, ISS_STATE, ("two_body", "j2"))
    expected = two_body_acceleration(ISS_STATE[:3]) + j2_acceleration(ISS_STATE[:3])
    assert y_dot_both[3:] == pytest.approx(expected, rel=1e-14)


# --- RAAN drift: the plan's acceptance test ----------------------------------------

def test_analytic_raan_rate_iss_like_is_minus_5_deg_per_day():
    assert j2_raan_rate_deg_per_day(6778.0, 0.0005, 51.64) == pytest.approx(-5.00, abs=0.01)


@pytest.mark.parametrize("orbit", ORBITS, ids=lambda o: o["name"])
def test_measured_raan_drift_matches_analytic_j2_rate(orbit):
    t, Y = propagate(initial_state(orbit), 10 * DAY_S, 600.0, forces=("two_body", "j2"))
    measured = drift_rate_per_day(t, raan_deg(Y))
    analytic = j2_raan_rate_deg_per_day(orbit["a_km"], orbit["e"], orbit["i_deg"])
    assert measured == pytest.approx(analytic, rel=0.02)


@pytest.mark.parametrize("orbit", ORBITS, ids=lambda o: o["name"])
def test_two_body_has_no_raan_drift(orbit):
    t, Y = propagate(initial_state(orbit), 3 * DAY_S, 600.0, forces=("two_body",))
    assert abs(drift_rate_per_day(t, raan_deg(Y))) < 1e-8