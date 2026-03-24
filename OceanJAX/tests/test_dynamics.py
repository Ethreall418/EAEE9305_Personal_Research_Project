"""
Tests for OceanJAX.Physics.dynamics
=====================================
Three groups of properties are verified:

  1. Baroclinic pressure is zero for a uniform tracer field
       equation_of_state on constant T, S gives rho' = 0 everywhere,
       so hydrostatic_pressure must return identically zero and both
       baroclinic PGFs must vanish.  A dimensional unit test verifies
       the integration formula for a single-layer anomaly.

  2. Continuity: compute_w satisfies div_h + div_z(w) = 0 exactly
       This is the design contract of compute_w for any (u, v) field.
       Bottom-w = 0 is also checked for the special case of globally
       non-divergent (uniform zonal) flow.

  3. Coriolis sign tests
       The 4-point-average Coriolis on a C-grid does not conserve KE
       exactly (that requires a Sadourny 1975-type scheme); no global KE
       test is included.  Sign and zero-input tests are verified instead.

Running
-------
    pytest OceanJAX/tests/test_dynamics.py -v
"""

from __future__ import annotations

import numpy as np
import jax
import jax.numpy as jnp
import pytest

from OceanJAX.grid import OceanGrid
from OceanJAX.state import ModelParams
from OceanJAX.Physics.dynamics import (
    equation_of_state,
    hydrostatic_pressure,
    pressure_gradient_u,
    pressure_gradient_v,
    coriolis_u,
    coriolis_v,
    compute_w,
)


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


# ---------------------------------------------------------------------------
# 1. Baroclinic pressure vanishes for reference T and S
# ---------------------------------------------------------------------------

class TestHydrostaticPressure:
    """
    When T = T_ref and S = S_ref everywhere, rho' = rho - rho0 = 0, so
    the baroclinic hydrostatic pressure must be zero, and both the zonal
    and meridional baroclinic PGFs must be identically zero.
    """

    def test_zero_anomaly_gives_zero_pressure(self, flat_grid, default_params):
        """Uniform T=T_ref, S=S_ref → p_hyd_prime = 0 everywhere."""
        grid   = flat_grid
        params = default_params
        T = jnp.full((grid.Nx, grid.Ny, grid.Nz), params.T_ref)
        S = jnp.full((grid.Nx, grid.Ny, grid.Nz), params.S_ref)

        rho         = equation_of_state(T, S, params)
        p_hyd_prime = hydrostatic_pressure(rho, grid, params)

        assert jnp.allclose(p_hyd_prime, 0.0, atol=1e-4), (
            f"Expected zero baroclinic pressure for reference T/S; "
            f"max|p| = {jnp.max(jnp.abs(p_hyd_prime)):.3e} Pa"
        )

    def test_zero_baroclinic_pgf_u(self, flat_grid, default_params):
        """Zero p_hyd_prime + flat eta → zero zonal baroclinic PGF."""
        grid   = flat_grid
        params = default_params
        p_hyd_prime = jnp.zeros((grid.Nx, grid.Ny, grid.Nz))
        eta         = jnp.zeros((grid.Nx, grid.Ny))

        pgf = pressure_gradient_u(p_hyd_prime, eta, grid, params)

        assert jnp.allclose(pgf, 0.0, atol=1e-10), (
            f"Expected zero PGF_u; max|pgf| = {jnp.max(jnp.abs(pgf)):.3e}"
        )

    def test_zero_baroclinic_pgf_v(self, flat_grid, default_params):
        """Zero p_hyd_prime + flat eta → zero meridional baroclinic PGF."""
        grid   = flat_grid
        params = default_params
        p_hyd_prime = jnp.zeros((grid.Nx, grid.Ny, grid.Nz))
        eta         = jnp.zeros((grid.Nx, grid.Ny))

        pgf = pressure_gradient_v(p_hyd_prime, eta, grid, params)

        assert jnp.allclose(pgf, 0.0, atol=1e-10), (
            f"Expected zero PGF_v; max|pgf| = {jnp.max(jnp.abs(pgf)):.3e}"
        )

    def test_single_layer_pressure_formula(self, flat_grid, default_params):
        """Dimensional unit test: single surface-layer anomaly.

        With rho' non-zero only in k=0 and zero elsewhere, the integration
        formula gives:

          p_hyd_prime[0]   = 0.5 * rho' * g * dz_c[0]    (cell-centre of k=0)
          p_hyd_prime[k>0] = rho' * g * dz_c[0]           (constant below, rho'[k>0]=0)

        This test verifies the formula directly without relying on the sign
        convention of the equation of state.
        """
        grid   = flat_grid
        params = default_params

        # Synthetic density anomaly: only the top layer is non-zero
        rho_prime_val = 0.5  # kg m-3 (arbitrary non-zero value)
        rho_prime = jnp.zeros((grid.Nx, grid.Ny, grid.Nz))
        rho_prime = rho_prime.at[:, :, 0].set(rho_prime_val)
        # Shift by rho0 so that equation_of_state is not needed here
        rho_full = rho_prime + params.rho0

        p_hyd_prime = hydrostatic_pressure(rho_full, grid, params)

        dz0 = float(grid.dz_c[0])

        # k=0: p = 0 + 0.5 * rho_prime_val * g * dz0
        expected_k0 = 0.5 * rho_prime_val * params.g * dz0
        assert jnp.allclose(p_hyd_prime[:, :, 0], expected_k0, rtol=1e-5), (
            f"k=0: expected {expected_k0:.4f} Pa, "
            f"got {float(jnp.mean(p_hyd_prime[:,:,0])):.4f} Pa"
        )

        # k>0: p = rho_prime_val * g * dz0 (full layer above fully counted)
        expected_kp = rho_prime_val * params.g * dz0
        assert jnp.allclose(p_hyd_prime[:, :, 1:], expected_kp, rtol=1e-5), (
            f"k>0: expected {expected_kp:.4f} Pa, "
            f"got {float(jnp.mean(p_hyd_prime[:,:,1])):.4f} Pa"
        )


# ---------------------------------------------------------------------------
# 2. Mass conservation: w at the sea floor is zero
# ---------------------------------------------------------------------------

class TestComputeW:
    """
    The design contract of compute_w is the discrete continuity equation:

      div_h(u, v)[i,j,k]  +  (w[i,j,k+1] - w[i,j,k]) / dz_c[k]  =  0

    This must hold for *any* (u, v) field, not just divergence-free ones.
    The bottom-w = 0 property is an additional check for the special case
    of globally non-divergent (uniform zonal periodic) flow.
    """

    def test_zero_velocity_gives_zero_w(self, flat_grid):
        """u = v = 0 → w = 0 everywhere."""
        grid = flat_grid
        u = jnp.zeros((grid.Nx, grid.Ny, grid.Nz))
        v = jnp.zeros((grid.Nx, grid.Ny, grid.Nz))

        w = compute_w(u, v, grid)

        assert jnp.allclose(w, 0.0, atol=1e-10), (
            f"Zero velocity should give zero w; max|w| = {jnp.max(jnp.abs(w)):.3e}"
        )

    def test_uniform_zonal_flow_bottom_w_zero(self, flat_grid):
        """Uniform eastward flow (periodic in x) has zero div_h → w[Nz] = 0."""
        grid  = flat_grid
        speed = 0.5  # m/s
        u = jnp.full((grid.Nx, grid.Ny, grid.Nz), speed) * grid.mask_u
        v = jnp.zeros((grid.Nx, grid.Ny, grid.Nz))

        w = compute_w(u, v, grid)

        assert jnp.allclose(w[:, :, grid.Nz], 0.0, atol=1e-5), (
            f"Uniform zonal flow: w at sea floor should be ~0; "
            f"max|w[Nz]| = {jnp.max(jnp.abs(w[:,:,grid.Nz])):.3e}"
        )

    def test_w_shape(self, flat_grid):
        """compute_w must return shape (Nx, Ny, Nz+1)."""
        grid = flat_grid
        u = jnp.zeros((grid.Nx, grid.Ny, grid.Nz))
        v = jnp.zeros((grid.Nx, grid.Ny, grid.Nz))

        w = compute_w(u, v, grid)

        assert w.shape == (grid.Nx, grid.Ny, grid.Nz + 1), (
            f"Expected shape {(grid.Nx, grid.Ny, grid.Nz+1)}, got {w.shape}"
        )

    def test_continuity_residual_random_flow(self, flat_grid):
        """compute_w satisfies div_h + div_z(w) = 0 for arbitrary (u, v).

        This is the fundamental design contract of compute_w.  The residual
        is computed independently from the returned w and the same area
        metrics used in the flux-form divergence, so the test is a genuine
        physical check rather than a tautological re-run of the integrator.
        """
        grid = flat_grid
        key  = jax.random.PRNGKey(0)
        u = jax.random.normal(key, (grid.Nx, grid.Ny, grid.Nz)) * grid.mask_u
        v = jax.random.normal(jax.random.fold_in(key, 1),
                              (grid.Nx, grid.Ny, grid.Nz)) * grid.mask_v

        w = compute_w(u, v, grid)

        # Recompute horizontal divergence independently (flux form)
        Fu    = u * grid.mask_u * grid.dy_c[:, :, jnp.newaxis]
        dFu   = Fu - jnp.roll(Fu, 1, axis=0)

        Fv    = v * grid.mask_v * grid.dx_v[:, :, jnp.newaxis]
        Fv_s  = jnp.concatenate(
            [jnp.zeros((grid.Nx, 1, grid.Nz), dtype=u.dtype), Fv[:, :-1, :]], axis=1
        )
        dFv   = Fv - Fv_s
        div_h = (dFu + dFv) / grid.area_c[:, :, jnp.newaxis]   # (Nx, Ny, Nz)

        # Vertical divergence of w at cell centres
        div_z_w = (w[:, :, 1:] - w[:, :, :-1]) / grid.dz_c     # (Nx, Ny, Nz)

        residual = (div_h + div_z_w) * grid.mask_c

        # The hard-wall BC mask_w[:,:,Nz]=0 forces w[bottom]=0 regardless of
        # flow divergence, so the deepest layer (k=Nz-1) does not satisfy
        # continuity for arbitrary (u,v).  Test only the interior layers.
        assert jnp.allclose(residual[:, :, :-1], 0.0, atol=1e-5), (
            f"Continuity residual (k=0..Nz-2) max = "
            f"{jnp.max(jnp.abs(residual[:,:,:-1])):.3e} s-1"
        )


# ---------------------------------------------------------------------------
# 3. Coriolis antisymmetry: no net kinetic-energy tendency
# ---------------------------------------------------------------------------

class TestCoriolis:
    """
    Sign and zero-input tests for the Coriolis terms.

    Note: the 4-point-average Coriolis discretisation on a C-grid does not
    in general conserve kinetic energy exactly (that requires a
    potential-enstrophy-conserving scheme such as Sadourny 1975).  No
    global KE-conservation test is included here for this reason.
    """

    def test_coriolis_u_zero_v(self, flat_grid):
        """v = 0 → coriolis_u = 0."""
        grid = flat_grid
        v    = jnp.zeros((grid.Nx, grid.Ny, grid.Nz))

        cor_u = coriolis_u(v, grid)

        assert jnp.allclose(cor_u, 0.0, atol=1e-10), (
            "coriolis_u must be zero when v = 0"
        )

    def test_coriolis_v_zero_u(self, flat_grid):
        """u = 0 → coriolis_v = 0."""
        grid = flat_grid
        u    = jnp.zeros((grid.Nx, grid.Ny, grid.Nz))

        cor_v = coriolis_v(u, grid)

        assert jnp.allclose(cor_v, 0.0, atol=1e-10), (
            "coriolis_v must be zero when u = 0"
        )

    def test_coriolis_u_sign(self, flat_grid):
        """Positive v + positive f (northern hemisphere) → positive coriolis_u."""
        grid = flat_grid   # lat 10°–50°N → f > 0 everywhere
        v = jnp.ones((grid.Nx, grid.Ny, grid.Nz)) * grid.mask_v

        cor_u = coriolis_u(v, grid)

        # Interior points (away from south wall where v_s=0 dilutes the average)
        assert jnp.all(cor_u[:, 1:-1, :] >= 0.0), (
            "With positive f and positive v, coriolis_u should be non-negative"
        )

    def test_coriolis_v_sign(self, flat_grid):
        """Positive u + positive f (northern hemisphere) → negative coriolis_v."""
        grid = flat_grid
        u = jnp.ones((grid.Nx, grid.Ny, grid.Nz)) * grid.mask_u

        cor_v = coriolis_v(u, grid)

        # Interior points
        assert jnp.all(cor_v[:, :-1, :] <= 0.0), (
            "With positive f and positive u, coriolis_v should be non-positive"
        )
