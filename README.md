# numerical-orbit-propagator

Numerical orbit propagator in Python: two-body equations of motion,
integration with SciPy `solve_ivp`, validation against Kepler, and
perturbations (starting with J2) as plug-in forces.

## Roadmap

- [x] Two-body equations of motion as a first-order state derivative, with tests
- [ ] Integrator comparison: Euler, RK4 and SciPy `solve_ivp` on a test problem
- [ ] Orbit propagation from initial orbital elements, with 2-D/3-D plots
- [ ] Validation: energy and angular-momentum drift, comparison with Kepler, tolerance study
- [ ] J2 perturbation as a plug-in force; RAAN drift checked against the analytic rate
- [ ] Continuous integration (GitHub Actions running pytest)

## Equations of motion

Two-body relative motion with the satellite's mass neglected (μ = GM):

    r̈ = −μ r / r³

Rewritten as six first-order equations for the integrator, with state
y = [x, y, z, vx, vy, vz] (km, km/s, ECI):

    ẏ = [ v ,  −μ r / r³ ]

`two_body_acceleration(r)` holds the physics; `two_body_derivative(t, y)`
slices the state, calls the acceleration and stacks [v, a]. Keeping them
separate lets further forces (J2, drag) be added as extra acceleration
functions.

Hand check: |a| = μ/r² = 8.676 m/s² at 400 km altitude (88% of g₀).

## Structure

    propagator/
        constants.py    μ, Earth radius (units in the names)
        dynamics.py     accelerations and the state derivative
    tests/
        test_dynamics.py

## Run

    python -m propagator.dynamics     # hand check
    python -m pytest -v               # 15 tests

## Assumptions

Point-mass Earth, no drag, no Sun/Moon, no solar radiation pressure,
no thrust. Each is added later as a separate force.