"""Tests for propagator.propagate and propagator.analysis."""

import numpy as np
import pytest

from propagator.analysis import (
    angular_momentum, closure_error_km, max_relative_drift, plane_shift_deg,
    radius_km, specific_energy,
)
from propagator.constants import MU_KM3_S2
from propagator.propagate import output_times, propagate

ISS_STATE = np.array([6774.611, 0.0, 0.0, 0.0, 4.761539368, 6.016190934])
ISS_A_KM = 6778.0
ISS_PERIOD_S = 2 * np.pi * np.sqrt(ISS_A_KM**3 / MU_KM3_S2)

R_CIRC_KM = 7000.0
CIRCULAR_STATE = np.array([R_CIRC_KM, 0.0, 0.0, 0.0, np.sqrt(MU_KM3_S2 / R_CIRC_KM), 0.0])


# --- output_times --------------------------------------------------------------

def test_output_times_regular_grid_with_exact_end():
    t = output_times(150.0, 60.0)
    assert t == pytest.approx([0.0, 60.0, 120.0, 150.0])


def test_output_times_no_duplicate_when_duration_is_a_multiple():
    t = output_times(120.0, 60.0)
    assert t == pytest.approx([0.0, 60.0, 120.0])


@pytest.mark.parametrize("duration_s, step_s", [(0.0, 60.0), (-10.0, 60.0), (100.0, 0.0)])
def test_output_times_rejects_bad_input(duration_s, step_s):
    with pytest.raises(ValueError):
        output_times(duration_s, step_s)


# --- propagate: structure ------------------------------------------------------

def test_propagate_shapes_and_first_row():
    t, Y = propagate(ISS_STATE, 600.0, 60.0)
    assert t.shape == (11,)
    assert Y.shape == (11, 6)
    assert Y[0] == pytest.approx(ISS_STATE, abs=1e-12)


def test_propagate_does_not_change_input():
    y0 = ISS_STATE.copy()
    propagate(y0, 600.0, 60.0)
    assert np.array_equal(y0, ISS_STATE)


def test_propagate_rejects_wrong_state_size():
    with pytest.raises(ValueError):
        propagate(ISS_STATE[:3], 600.0)


# --- propagate: physics --------------------------------------------------------

def test_one_period_closes():
    _, Y = propagate(ISS_STATE, ISS_PERIOD_S)
    assert closure_error_km(Y) < 1e-3                     # under 1 m (actual ~1.4 mm)


def test_circular_orbit_keeps_constant_radius():
    _, Y = propagate(CIRCULAR_STATE, 86400.0)
    assert radius_km(Y) == pytest.approx(R_CIRC_KM, rel=1e-9)


def test_radius_stays_between_perigee_and_apogee():
    _, Y = propagate(ISS_STATE, 86400.0)
    r = radius_km(Y)
    e = 0.0005
    assert r.min() >= ISS_A_KM * (1 - e) - 1e-6
    assert r.max() <= ISS_A_KM * (1 + e) + 1e-6


def test_energy_and_angular_momentum_conserved_over_one_day():
    _, Y = propagate(ISS_STATE, 86400.0)
    assert max_relative_drift(specific_energy(Y)) < 1e-8
    assert max_relative_drift(np.linalg.norm(angular_momentum(Y), axis=1)) < 1e-8


def test_orbit_plane_fixed():
    _, Y = propagate(ISS_STATE, 86400.0)
    assert plane_shift_deg(Y) < 1e-5


def test_mu_is_used():
    _, Y_earth = propagate(CIRCULAR_STATE, 600.0)
    _, Y_heavy = propagate(CIRCULAR_STATE, 600.0, mu=2 * MU_KM3_S2)
    assert not np.allclose(Y_earth[-1], Y_heavy[-1])


# --- analysis on known values --------------------------------------------------

def test_specific_energy_of_circular_orbit():
    Y = CIRCULAR_STATE.reshape(1, 6)
    assert specific_energy(Y)[0] == pytest.approx(-MU_KM3_S2 / (2 * R_CIRC_KM), rel=1e-12)


def test_max_relative_drift():
    assert max_relative_drift([2.0, 2.002, 1.999]) == pytest.approx(1e-3)