"""Tests for the plug-in force model (dynamics.FORCES, derivative, propagate(forces=...))."""

import numpy as np
import pytest

from propagator.dynamics import FORCES, derivative, two_body_derivative
from propagator.propagate import propagate

ISS_STATE = np.array([6774.611, 0.0, 0.0, 0.0, 4.761539368, 6.016190934])


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