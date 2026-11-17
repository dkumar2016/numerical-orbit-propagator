"""Validation tests: the propagator against the exact two-body (Kepler) solution.

Every orbit in data/initial_orbits.csv is tested, so adding a row adds tests.
"""

import numpy as np
import pytest

from propagator.analysis import ric_components
from propagator.constants import DATA_DIR, DAY_S
from propagator.orbits import ELEMENT_FIELDS, initial_state, read_initial_orbits
from propagator.propagate import propagate
from propagator.reference_problems import two_body_exact

ORBITS = read_initial_orbits(DATA_DIR / "initial_orbits.csv")
ONE_DAY_LIMIT_M = 1.0          # measured: ~5 mm (ISS-like), ~5 cm (Molniya)


def by_name(orbit):
    return orbit["name"]


# --- Reading orbits ------------------------------------------------------------

def test_orbits_file_has_names_and_numeric_elements():
    assert len(ORBITS) >= 2
    for orbit in ORBITS:
        assert isinstance(orbit["name"], str)
        assert all(isinstance(orbit[k], float) for k in ELEMENT_FIELDS)


# --- The exact reference -------------------------------------------------------

@pytest.mark.parametrize("orbit", ORBITS, ids=by_name)
def test_exact_solution_starts_at_initial_state(orbit):
    Y_exact = two_body_exact(orbit, [0.0])
    assert Y_exact[0] == pytest.approx(initial_state(orbit), abs=1e-9)


# --- Propagator vs exact -------------------------------------------------------

@pytest.mark.parametrize("orbit", ORBITS, ids=by_name)
def test_one_day_error_against_kepler(orbit):
    t, Y = propagate(initial_state(orbit), DAY_S, 600.0)
    error_m = np.linalg.norm(Y[:, :3] - two_body_exact(orbit, t)[:, :3], axis=1) * 1000
    assert error_m.max() < ONE_DAY_LIMIT_M


@pytest.mark.parametrize("orbit", ORBITS, ids=by_name)
def test_error_shrinks_as_tolerance_tightens(orbit):
    y0 = initial_state(orbit)
    exact_end = two_body_exact(orbit, [DAY_S])[0]
    errors = []
    for rtol in [1e-6, 1e-8, 1e-10]:
        _, Y = propagate(y0, DAY_S, DAY_S, rtol=rtol, atol=rtol * 1e-2)
        errors.append(np.linalg.norm(Y[-1, :3] - exact_end[:3]))
    assert errors[0] > errors[1] > errors[2]


@pytest.mark.parametrize("orbit", ORBITS, ids=by_name)
def test_error_is_mostly_in_track(orbit):
    t, Y = propagate(initial_state(orbit), 3 * DAY_S, 3 * DAY_S)
    radial, in_track, cross = np.abs(ric_components(two_body_exact(orbit, t), Y)[-1])
    assert in_track > 3 * radial
    assert in_track > 1000 * cross


# --- RIC decomposition on known geometry --------------------------------------

REF = np.array([[7000.0, 0.0, 0.0, 0.0, 7.5, 0.0]])       # on x axis, moving along +y


@pytest.mark.parametrize("offset, expected", [
    ([1.0, 0.0, 0.0], [1.0, 0.0, 0.0]),        # outward: radial
    ([0.0, 1.0, 0.0], [0.0, 1.0, 0.0]),        # along the velocity: in-track
    ([0.0, 0.0, 1.0], [0.0, 0.0, 1.0]),        # along h = r x v: cross-track
])
def test_ric_directions(offset, expected):
    Y = REF.copy()
    Y[0, :3] += offset
    assert ric_components(REF, Y)[0] == pytest.approx(expected, abs=1e-12)


def test_ric_preserves_error_length():
    rng = np.random.default_rng(1)
    Y_ref = np.hstack((rng.normal(0, 7000, (5, 3)), rng.normal(0, 7, (5, 3))))
    Y = Y_ref.copy()
    Y[:, :3] += rng.normal(0, 1, (5, 3))
    ric = ric_components(Y_ref, Y)
    assert np.linalg.norm(ric, axis=1) == pytest.approx(
        np.linalg.norm(Y[:, :3] - Y_ref[:, :3], axis=1), rel=1e-12)


# --- propagate(..., return_nfev=True) -----------------------------------------

def test_propagate_can_return_call_count():
    t, Y, nfev = propagate(initial_state(ORBITS[0]), 600.0, 60.0, return_nfev=True)
    assert Y.shape == (11, 6)
    assert isinstance(nfev, int) and nfev > 0