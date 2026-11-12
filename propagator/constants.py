"""Physical constants and project paths. Units are in the names: km, s."""

from pathlib import Path

MU_KM3_S2 = 398600.4418     # Earth's gravitational parameter, GM
R_EARTH_KM = 6378.137       # Earth's equatorial radius (WGS-84)
G0_M_S2 = 9.80665           # standard gravity, for comparisons only

REPO_DIR = Path(__file__).resolve().parent.parent
OUTPUT_DIR = REPO_DIR / "output"
FIGURES_DIR = REPO_DIR / "figures"