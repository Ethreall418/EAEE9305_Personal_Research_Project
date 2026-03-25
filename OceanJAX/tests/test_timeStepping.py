"""
Tests for OceanJAX.timeStepping
=================================
Five groups of properties are verified:

  1. Resting ocean stability
       A motionless, horizontally uniform ocean with no surface forcing
       must remain at rest after any number of time steps.

  2. State shape and time consistency
       After one step, every field must have the correct shape and the
       time counter must advance by exactly dt.

  3. step_count management
       step_count must start at 0 and increment by 1 on every step(),
       including through run().

  4. Bootstrap (AB1 → AB2 → AB3 and leapfrog dt)
       The AB order selection is driven by step_count:
         step_count == 0  →  AB1 (ignores T_tend_prev and T_tend_prev2)
         step_count == 1  →  AB2 (uses T_tend_prev with -1/2 coefficient)
         step_count >= 2  →  AB3 (uses both history fields)
       The leapfrog multiplier is dt at step_count == 0 and 2*dt afterwards.

  5. eta / w kinematic surface BC
       The surface face of w (k=0) must equal deta/dt so that the
       continuity equation is satisfied at the free surface.

Running
-------
    pytest OceanJAX/tests/test_timeStepping.py -v
"""

from __future__ import annotations

import numpy as np
import jax.numpy as jnp
import equinox as eqx
import pytest

from OceanJAX.grid import OceanGrid
from OceanJAX.state import ModelParams, OceanState
from OceanJAX.state import create_rest_state
from OceanJAX.timeStepping import step, run, SurfaceForcing


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
        """All fields of a resting ocean remain at rest after 10 consecutive steps."""
        state = rest_state
        for _ in range(10):
            state = step(state, flat_grid, default_params, forcing=None)

        assert jnp.allclose(state.u, 0.0, atol=1e-5), (
            f"u drifted after 10 steps; max|u| = {jnp.max(jnp.abs(state.u)):.3e}"
        )
        assert jnp.allclose(state.v, 0.0, atol=1e-5), (
            f"v drifted after 10 steps; max|v| = {jnp.max(jnp.abs(state.v)):.3e}"
        )
        assert jnp.allclose(state.w, 0.0, atol=1e-5), (
            f"w drifted after 10 steps; max|w| = {jnp.max(jnp.abs(state.w)):.3e}"
        )
        assert jnp.allclose(state.T, rest_state.T, atol=1e-5), (
            f"T drifted after 10 steps; max|ΔT| = "
            f"{jnp.max(jnp.abs(state.T - rest_state.T)):.3e}"
        )
        assert jnp.allclose(state.S, rest_state.S, atol=1e-5), (
            f"S drifted after 10 steps; max|ΔS| = "
            f"{jnp.max(jnp.abs(state.S - rest_state.S)):.3e}"
        )
        assert jnp.allclose(state.eta, 0.0, atol=1e-7), (
            f"eta drifted after 10 steps; max|eta| = {jnp.max(jnp.abs(state.eta)):.3e}"
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


# ---------------------------------------------------------------------------
# 3. step_count management
# ---------------------------------------------------------------------------

class TestStepCount:
    """step_count must start at 0 and increment by 1 on every step."""

    def test_initial_step_count_zero(self, flat_grid, default_params):
        """create_rest_state must produce step_count = 0."""
        state = create_rest_state(flat_grid)
        assert int(state.step_count) == 0

    def test_step_count_increments_with_step(self, flat_grid, default_params, rest_state):
        """After N consecutive step() calls step_count must equal N."""
        state = rest_state
        for n in range(1, 5):
            state = step(state, flat_grid, default_params, forcing=None)
            assert int(state.step_count) == n, (
                f"Expected step_count {n}, got {int(state.step_count)}"
            )

    def test_run_step_count(self, flat_grid, default_params, rest_state):
        """After run(n_steps=N) the final step_count must equal N."""
        N = 7
        final_state, _ = run(rest_state, flat_grid, default_params, n_steps=N)
        assert int(final_state.step_count) == N

    def test_run_time_final(self, flat_grid, default_params, rest_state):
        """After run(n_steps=N) the elapsed time must equal N * dt."""
        N = 5
        final_state, _ = run(rest_state, flat_grid, default_params, n_steps=N)
        expected_time = float(rest_state.time) + N * default_params.dt
        assert jnp.allclose(final_state.time, expected_time, rtol=1e-6), (
            f"Expected time {expected_time:.1f} s, got {float(final_state.time):.1f} s"
        )


# ---------------------------------------------------------------------------
# 4. Bootstrap — AB order and leapfrog dt selection
# ---------------------------------------------------------------------------

class TestBootstrap:
    """
    The AB scheme and leapfrog multiplier must be selected by step_count.

    Strategy: use a resting ocean so that all explicit tendencies (G_T, G_u …)
    are zero or analytically known.  Then inject non-zero T_tend_prev /
    T_tend_prev2 (via eqx.tree_at) to expose which history coefficients are
    actually applied.

    With G_T = 0 (rest state, no forcing):
      AB1 (step_count=0): T_new = T                          (prev fields ignored)
      AB2 (step_count=1): T_new = T - dt/2 * T_tend_prev    (prev field used)
      AB3 (step_count≥2): T_new = T + dt*(-16/12*T_tend_prev + 5/12*T_tend_prev2)
    """

    # Fake tendency values — uniform so that implicit diffusion leaves them
    # intact (a spatially uniform increment is not diffused).
    _PREV  = 0.1    # °C s-1 injected as T_tend_prev
    _PREV2 = 0.05   # °C s-1 injected as T_tend_prev2

    def _state_with_history(self, rest_state, sc: int):
        """Return rest_state with fake tendency history and the given step_count."""
        fake_prev  = jnp.full_like(rest_state.T, self._PREV)
        fake_prev2 = jnp.full_like(rest_state.T, self._PREV2)
        s = eqx.tree_at(
            lambda s: (s.T_tend_prev, s.T_tend_prev2, s.step_count),
            rest_state,
            (fake_prev, fake_prev2, jnp.array(sc, dtype=jnp.int32)),
        )
        return s

    def test_ab1_ignores_prev_tendencies(self, flat_grid, default_params, rest_state):
        """step_count=0 must use AB1: T_new = T (prev history ignored)."""
        s = self._state_with_history(rest_state, sc=0)
        new = step(s, flat_grid, default_params, forcing=None)
        # G_T = 0 for rest state → AB1 gives T_new = T; AB2/AB3 would change T
        assert jnp.allclose(new.T, rest_state.T, atol=1e-6), (
            "AB1 at step_count=0 must ignore T_tend_prev; T should be unchanged"
        )

    def test_ab2_uses_prev_tendency(self, flat_grid, default_params, rest_state):
        """step_count=1 must use AB2: T_new = T - dt/2 * T_tend_prev."""
        dt = default_params.dt
        s  = self._state_with_history(rest_state, sc=1)
        new = step(s, flat_grid, default_params, forcing=None)
        # G_T = 0, so AB2 → T_new = T + dt * (-1/2) * T_tend_prev
        expected_delta = -0.5 * dt * self._PREV
        actual_delta   = float(jnp.mean(new.T - rest_state.T))
        assert abs(actual_delta - expected_delta) < 1e-4, (
            f"AB2 delta T = {actual_delta:.6f}, expected {expected_delta:.6f}"
        )

    def test_ab3_uses_both_prev_tendencies(self, flat_grid, default_params, rest_state):
        """step_count=2 must use AB3 with coefficients (23/12, -16/12, 5/12)."""
        dt = default_params.dt
        ab3_0, ab3_1, ab3_2 = default_params.ab3_coeffs   # (23/12, -16/12, 5/12)
        s   = self._state_with_history(rest_state, sc=2)
        new = step(s, flat_grid, default_params, forcing=None)
        # G_T = 0 → T_new = T + dt * (ab3_1 * T_tend_prev + ab3_2 * T_tend_prev2)
        expected_delta = dt * (ab3_1 * self._PREV + ab3_2 * self._PREV2)
        actual_delta   = float(jnp.mean(new.T - rest_state.T))
        assert abs(actual_delta - expected_delta) < 1e-4, (
            f"AB3 delta T = {actual_delta:.6f}, expected {expected_delta:.6f}"
        )

    def test_leapfrog_first_step_uses_single_dt(self, flat_grid, default_params, rest_state):
        """
        The leapfrog multiplier is ``dt`` at step_count=0 and ``2*dt`` at
        step_count=1.  With a resting ocean and wind stress the momentum
        increment at step_count=0 must be exactly half that at step_count=1.
        """
        tau_val = 0.1   # N m-2 — constant uniform wind stress
        forcing = SurfaceForcing(
            heat_flux=jnp.zeros((flat_grid.Nx, flat_grid.Ny)),
            fw_flux  =jnp.zeros((flat_grid.Nx, flat_grid.Ny)),
            tau_x    =jnp.full((flat_grid.Nx, flat_grid.Ny), tau_val),
            tau_y    =jnp.zeros((flat_grid.Nx, flat_grid.Ny)),
        )

        # step_count=0 → leapfrog_dt = dt
        s0 = rest_state                                                    # step_count=0
        new0 = step(s0, flat_grid, default_params, forcing=forcing)

        # step_count=1 but otherwise identical input (same u, u_prev, T, S …)
        s1 = eqx.tree_at(
            lambda s: s.step_count, rest_state, jnp.array(1, dtype=jnp.int32)
        )
        new1 = step(s1, flat_grid, default_params, forcing=forcing)

        # After implicit vertical viscosity the factor-of-2 relationship is
        # preserved exactly (linear operator scales with the input).
        # Compare the surface momentum layer (k=0) where wind forcing acts.
        u0 = np.array(new0.u[:, :, 0])
        u1 = np.array(new1.u[:, :, 0])
        np.testing.assert_allclose(u1, 2.0 * u0, rtol=1e-5,
                                   err_msg="leapfrog_dt=2*dt at step_count=1 "
                                           "must give twice the u increment of dt at step_count=0")


# ---------------------------------------------------------------------------
# 5. eta / w kinematic surface BC
# ---------------------------------------------------------------------------

class TestEtaWConsistency:
    """
    After every step, the surface face of w must equal deta/dt = (eta_new -
    eta_old) / dt.  This enforces the kinematic free-surface boundary
    condition and ensures that the free-surface update and the continuity
    diagnosis are consistent.
    """

    def test_w_surface_equals_deta_dt_rest(self, flat_grid, default_params, rest_state):
        """For resting ocean w[0] and deta/dt are both zero — trivial BC holds."""
        new = step(rest_state, flat_grid, default_params, forcing=None)
        # eta_old = 0, eta_new = 0 → deta/dt = 0 → w[0] = 0
        assert jnp.allclose(new.w[:, :, 0], 0.0, atol=1e-9)

    def test_w_surface_equals_deta_dt_with_forcing(self, flat_grid, default_params, rest_state):
        """
        With spatially varying wind stress the horizontal velocity divergence is
        non-zero, so eta and w[:, :, 0] are both non-trivial.  Their relationship
        w[:, :, 0] = (eta_new - eta_old) / dt must hold exactly.
        """
        # Linearly varying tau_x in x → zonal gradient → divergence ≠ 0
        tau_x = (jnp.arange(flat_grid.Nx, dtype=jnp.float32)[:, None]
                 * jnp.ones((flat_grid.Nx, flat_grid.Ny)) * 0.05)
        forcing = SurfaceForcing(
            heat_flux=jnp.zeros((flat_grid.Nx, flat_grid.Ny)),
            fw_flux  =jnp.zeros((flat_grid.Nx, flat_grid.Ny)),
            tau_x    =tau_x,
            tau_y    =jnp.zeros((flat_grid.Nx, flat_grid.Ny)),
        )

        new = step(rest_state, flat_grid, default_params, forcing=forcing)
        dt  = default_params.dt

        # For a full-ocean flat grid, mask_c[:, :, 0] = mask_w[:, :, 0] = 1 everywhere.
        # The kinematic BC then reads: w_new[:, :, 0] = (eta_new - eta_old) / dt
        deta_dt_from_eta = (new.eta - rest_state.eta) / dt
        np.testing.assert_allclose(
            np.array(new.w[:, :, 0]),
            np.array(deta_dt_from_eta),
            atol=1e-6,
            err_msg="w[:, :, 0] must equal (eta_new - eta_old) / dt"
        )

    def test_w_surface_nonzero_after_divergent_forcing(self, flat_grid, default_params, rest_state):
        """With spatially varying wind stress w[:, :, 0] must not be uniformly zero."""
        tau_x = (jnp.arange(flat_grid.Nx, dtype=jnp.float32)[:, None]
                 * jnp.ones((flat_grid.Nx, flat_grid.Ny)) * 0.05)
        forcing = SurfaceForcing(
            heat_flux=jnp.zeros((flat_grid.Nx, flat_grid.Ny)),
            fw_flux  =jnp.zeros((flat_grid.Nx, flat_grid.Ny)),
            tau_x    =tau_x,
            tau_y    =jnp.zeros((flat_grid.Nx, flat_grid.Ny)),
        )
        new = step(rest_state, flat_grid, default_params, forcing=forcing)
        assert not jnp.allclose(new.w[:, :, 0], 0.0, atol=1e-10), (
            "Spatially varying wind stress should produce non-zero w[0]"
        )
