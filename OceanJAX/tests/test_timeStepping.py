"""
Tests for OceanJAX.timeStepping
=================================
Two fundamental properties of the time stepper are verified:

  1. Resting ocean stability
       A motionless, horizontally uniform ocean with no surface forcing
       must remain at rest after any number of time steps.  Any nonzero
       tendency would indicate a metric error, mask inconsistency, or
       spurious pressure gradient in the coupled physics.


  2. State shape consistency
       After one step, every field in the returned OceanState must have
       the same shape as in the initial state, and the time counter must
       advance by exactly dt.

Running
-------
    pytest OceanJAX/tests/test_timeStepping.py -v
"""

from __future__ import annotations

import numpy as np
import jax.numpy as jnp
import pytest

from OceanJAX.grid import OceanGrid
from OceanJAX.state import ModelParams, OceanState
from OceanJAX.state import create_rest_state
from OceanJAX.timeStepping import step, SurfaceForcing


# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def flat_grid():
    """Small 4×4×4 flat-bottom open-ocean grid (no land)."""
    z_levels = np.array([5.0, 20.0, 50.0, 100.0], dtype=np.float64)
    return OceanGrid.create(
        lon_bounds=(0.0, 40.0),
        lat_bounds=(10.0, 50.0),
        depth_levels=z_levels,
        Nx=4,
        Ny=4,
    )


@pytest.fixture(scope="module")
def default_params():
    return ModelParams()


@pytest.fixture(scope="module")
def rest_state(flat_grid, default_params):
    """Resting ocean with T=T_ref, S=S_ref, all velocities zero."""
    return create_rest_state(
        flat_grid,
        T_background=default_params.T_ref,
        S_background=default_params.S_ref,
    )


# ---------------------------------------------------------------------------
# 1. Resting ocean stability
# ---------------------------------------------------------------------------

class TestRestingOcean:
    """
    A resting, horizontally uniform ocean should remain at rest indefinitely.

    With u = v = w = 0, eta = 0, T = const, S = const:
      - Coriolis = 0 (v = 0, u = 0)
      - PGF_baroclinic = 0 (uniform T/S → rho' = 0 → p_hyd_prime = 0)
      - PGF_barotropic = 0 (eta = 0)
      - Horizontal viscosity = 0 (uniform velocity)
      - Tracer advection = 0 (u = v = w = 0)
      - Tracer diffusion = 0 (uniform T/S)
      - Surface forcing = 0 (forcing=None)
    Therefore every tendency is zero and the state is a fixed point.
    """

    def test_velocities_remain_zero(self, flat_grid, default_params, rest_state):
        """u, v, w must all remain zero after one step."""
        new_state = step(rest_state, flat_grid, default_params, forcing=None)

        assert jnp.allclose(new_state.u, 0.0, atol=1e-6), (
            f"u should stay zero; max|u| = {jnp.max(jnp.abs(new_state.u)):.3e}"
        )
        assert jnp.allclose(new_state.v, 0.0, atol=1e-6), (
            f"v should stay zero; max|v| = {jnp.max(jnp.abs(new_state.v)):.3e}"
        )
        assert jnp.allclose(new_state.w, 0.0, atol=1e-6), (
            f"w should stay zero; max|w| = {jnp.max(jnp.abs(new_state.w)):.3e}"
        )

    def test_tracers_unchanged(self, flat_grid, default_params, rest_state):
        """T and S must be unchanged after one step with no forcing."""
        new_state = step(rest_state, flat_grid, default_params, forcing=None)

        assert jnp.allclose(new_state.T, rest_state.T, atol=1e-6), (
            f"T changed; max|ΔT| = {jnp.max(jnp.abs(new_state.T - rest_state.T)):.3e}"
        )
        assert jnp.allclose(new_state.S, rest_state.S, atol=1e-6), (
            f"S changed; max|ΔS| = {jnp.max(jnp.abs(new_state.S - rest_state.S)):.3e}"
        )

    def test_eta_remains_zero(self, flat_grid, default_params, rest_state):
        """SSH must remain zero for a resting, divergence-free ocean."""
        new_state = step(rest_state, flat_grid, default_params, forcing=None)

        assert jnp.allclose(new_state.eta, 0.0, atol=1e-8), (
            f"eta should stay zero; max|eta| = {jnp.max(jnp.abs(new_state.eta)):.3e}"
        )

    def test_resting_ocean_multi_step(self, flat_grid, default_params, rest_state):
        """Resting ocean remains at rest after 10 consecutive steps."""
        state = rest_state
        for _ in range(10):
            state = step(state, flat_grid, default_params, forcing=None)

        assert jnp.allclose(state.u, 0.0, atol=1e-5), (
            f"u drifted after 10 steps; max|u| = {jnp.max(jnp.abs(state.u)):.3e}"
        )
        assert jnp.allclose(state.T, rest_state.T, atol=1e-5), (
            f"T drifted after 10 steps; max|ΔT| = "
            f"{jnp.max(jnp.abs(state.T - rest_state.T)):.3e}"
        )


# ---------------------------------------------------------------------------
# 2. State shape and time consistency
# ---------------------------------------------------------------------------

class TestStateConsistency:
    """
    After one step, all output fields must have the correct shapes and
    the time counter must advance by exactly dt.
    """

    def test_output_shapes(self, flat_grid, default_params, rest_state):
        """All fields in the returned state have the expected shapes."""
        grid      = flat_grid
        new_state = step(rest_state, grid, default_params, forcing=None)

        assert new_state.u.shape   == (grid.Nx, grid.Ny, grid.Nz),     "u shape"
        assert new_state.v.shape   == (grid.Nx, grid.Ny, grid.Nz),     "v shape"
        assert new_state.w.shape   == (grid.Nx, grid.Ny, grid.Nz + 1), "w shape"
        assert new_state.T.shape   == (grid.Nx, grid.Ny, grid.Nz),     "T shape"
        assert new_state.S.shape   == (grid.Nx, grid.Ny, grid.Nz),     "S shape"
        assert new_state.eta.shape == (grid.Nx, grid.Ny),              "eta shape"

    def test_time_advances_by_dt(self, flat_grid, default_params, rest_state):
        """time field must increase by exactly dt after one step."""
        new_state = step(rest_state, flat_grid, default_params, forcing=None)
        expected  = float(rest_state.time) + default_params.dt

        assert jnp.allclose(new_state.time, expected, rtol=1e-6), (
            f"Expected time {expected:.1f} s, got {float(new_state.time):.1f} s"
        )

    def test_tendency_history_rotated(self, flat_grid, default_params, rest_state):
        """T_tend_prev2 in the new state must equal T_tend_prev of the old state."""
        new_state = step(rest_state, flat_grid, default_params, forcing=None)

        assert jnp.allclose(
            new_state.T_tend_prev2, rest_state.T_tend_prev, atol=1e-10
        ), "AB3 tendency history not rotated correctly for T"
        assert jnp.allclose(
            new_state.S_tend_prev2, rest_state.S_tend_prev, atol=1e-10
        ), "AB3 tendency history not rotated correctly for S"

    def test_surface_forcing_warms_top_layer(self, flat_grid, default_params, rest_state):
        """
        A positive heat flux must warm the top tracer layer and leave
        deeper layers unchanged after one step.

        Vertical diffusion (kappa_v) and viscosity (nu_v) are set to zero
        so that only the explicit surface forcing tendency reaches k=0;
        no implicit mixing can leak heat downward into k=1.
        """
        grid   = flat_grid
        # Use zero diffusion/viscosity to isolate the surface forcing signal
        params = ModelParams(kappa_v=0.0, nu_v=0.0)

        forcing = SurfaceForcing(
            heat_flux=jnp.full((grid.Nx, grid.Ny), 200.0),   # 200 W m-2 into ocean
            fw_flux  =jnp.zeros((grid.Nx, grid.Ny)),
            tau_x    =jnp.zeros((grid.Nx, grid.Ny)),
            tau_y    =jnp.zeros((grid.Nx, grid.Ny)),
        )

        new_state = step(rest_state, grid, params, forcing=forcing)

        dT = new_state.T - rest_state.T
        assert jnp.all(dT[:, :, 0] > 0.0), (
            "Positive heat flux should warm the surface layer"
        )
        assert jnp.allclose(dT[:, :, 1:], 0.0, atol=1e-7), (
            "Heat flux should not affect layers below k=0"
        )
