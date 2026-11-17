"""Physical constants and project paths. Units are in the names: km, s."""

from pathlib import Path

MU_KM3_S2 = 398600.4418     # Earth's gravitational parameter, GM
R_EARTH_KM = 6378.137       # Earth's equatorial radius (WGS-84)
G0_M_S2 = 9.80665           # standard gravity, for comparisons only
DAY_S = 86400.0             # seconds in a (solar) day

REPO_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = REPO_DIR / "data"
OUTPUT_DIR = REPO_DIR / "output"
FIGURES_DIR = REPO_DIR / "figures"
DATA_DIR = REPO_DIR / "data"