"""Initial orbits: reading them from CSV and turning them into state vectors."""

import csv

import numpy as np

from toolkit.conversions import coe_to_rv

ELEMENT_FIELDS = ["a_km", "e", "i_deg", "raan_deg", "argp_deg", "nu_deg"]


def read_initial_orbits(path):
    """Return a list of {'name': str, 'a_km': float, ...} from the CSV."""
    with open(path, newline="") as f:
        return [{"name": row["name"], **{k: float(row[k]) for k in ELEMENT_FIELDS}}
                for row in csv.DictReader(f)]


def elements_of(orbit):
    """The six elements in the order coe_to_rv and kepler.propagate expect."""
    return [orbit[k] for k in ELEMENT_FIELDS]


def initial_state(orbit):
    """State vector [x, y, z, vx, vy, vz] (km, km/s) at the orbit's epoch."""
    r, v = coe_to_rv(*elements_of(orbit))
    return np.concatenate((r, v))