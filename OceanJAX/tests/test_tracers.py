"""
Tests for OceanJAX.Physics.tracers
====================================
Two fundamental properties of a conservative flux-form discretisation:

  1. Constant-tracer consistency
       For a spatially uniform tracer C and a divergence-free velocity field,
       upwind_advection must return identically zero.  Any nonzero tendency
       would indicate a metric or mask inconsistency in the flux assembly.

  2. Global tracer conservation
       For a closed domain (periodic in x, solid walls in y, hard surface/
       bottom via mask_w_adv), the global integral of the advective tendency
       must vanish exactly (up to floating-point round-off), regardless of
       the tracer distribution or velocity field.

Additional unit tests verify the sign, units, and top-layer thickness
conversion for the surface forcing tendencies.

Running
-------
    pytest OceanJAX/tests/test_tracers.py -v
"""

from __future__ import annotations

import numpy as np
import jax
import jax.numpy as jnp
import pytest

from OceanJAX.grid import OceanGrid
from OceanJAX.state import ModelParams
from OceanJAX.Physics.tracers import (
    upwind_advection,
    centered_advection,
    tracer_tendency,
    surface_layer_tendency,
    heat_surface_tendency,
    salt_surface_tendency,
)

# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def flat_grid():
    """
    Small 4×4×4 flat-bottom open-ocean grid (no land).
    Periodic in x; solid walls in y; hard surface/bottom for advection.
    """
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
# Helper: simple divergence-free velocity
# ---------------------------------------------------------------------------

def _zonal_uniform_flow(speed: float, grid: OceanGrid):
    """u = speed everywhere, v = 0, w = 0.
    This is divergence-free: div_h = (Fu_e - Fu_w)/A = 0 because the east-
    face area dy_c is independent of i, making every column's Fu_e equal to
    the same constant.
    """
    u = jnp.full((grid.Nx, grid.Ny, grid.Nz), speed) * grid.mask_u
    v = jnp.zeros((grid.Nx, grid.Ny, grid.Nz))
    w = jnp.zeros((grid.Nx, grid.Ny, grid.Nz + 1))
    return u, v, w


# ---------------------------------------------------------------------------
# 1. Constant-tracer consistency
# ---------------------------------------------------------------------------

class TestConstantTracer:
    """
    A uniform tracer field advected by any divergence-free velocity must
    produce zero tendency in every wet cell.
    """

    def test_upwind_zero_velocity(self, flat_grid):
        """Trivial case: u = v = w = 0."""
        grid = flat_grid
        phi  = jnp.ones((grid.Nx, grid.Ny, grid.Nz)) * grid.mask_c
        u    = jnp.zeros_like(phi)
        v    = jnp.zeros_like(phi)
        w    = jnp.zeros((grid.Nx, grid.Ny, grid.Nz + 1))

        tend = upwind_advection(phi, u, v, w, grid)

        assert jnp.allclose(tend, 0.0, atol=1e-6), (
            f"Non-zero tendency for zero velocity: max|tend| = {jnp.max(jnp.abs(tend)):.3e}"
        )

    def test_upwind_uniform_eastward_flow(self, flat_grid):
        """
        Uniform eastward flow on periodic grid: dy_c is independent of i,
        so Fu_e[i] = Fu_w[i] everywhere and the zonal tendency is zero.
        """
        grid = flat_grid
        phi  = jnp.ones((grid.Nx, grid.Ny, grid.Nz)) * grid.mask_c
        u, v, w = _zonal_uniform_flow(1.0, grid)

        tend = upwind_advection(phi, u, v, w, grid)

        assert jnp.allclose(tend, 0.0, atol=1e-6), (
            f"Constant tracer + uniform u gave non-zero tendency: "
            f"max|tend| = {jnp.max(jnp.abs(tend)):.3e}"
        )

    def test_upwind_uniform_westward_flow(self, flat_grid):
        """Same as above for u < 0 (upwind direction flips, result unchanged)."""
        grid = flat_grid
        phi  = jnp.ones((grid.Nx, grid.Ny, grid.Nz)) * grid.mask_c
        u, v, w = _zonal_uniform_flow(-0.5, grid)

        tend = upwind_advection(phi, u, v, w, grid)

        assert jnp.allclose(tend, 0.0, atol=1e-6), (
            f"Constant tracer + uniform -u gave non-zero tendency: "
            f"max|tend| = {jnp.max(jnp.abs(tend)):.3e}"
        )

    def test_centered_zero_velocity(self, flat_grid):
        """Centered scheme: constant phi + zero velocity -> zero tendency."""
        grid = flat_grid
        phi  = jnp.ones((grid.Nx, grid.Ny, grid.Nz)) * grid.mask_c
        u    = jnp.zeros_like(phi)
        v    = jnp.zeros_like(phi)
        w    = jnp.zeros((grid.Nx, grid.Ny, grid.Nz + 1))

        tend = centered_advection(phi, u, v, w, grid)

        assert jnp.allclose(tend, 0.0, atol=1e-6), (
            f"Centered: non-zero tendency for zero velocity: "
            f"max|tend| = {jnp.max(jnp.abs(tend)):.3e}"
        )


# ---------------------------------------------------------------------------
# 2. Global tracer conservation
# ---------------------------------------------------------------------------

class TestGlobalConservation:
    """
    For any tracer distribution and any zonal-only flow (v = w = 0),
    the global integral of the advective tendency must vanish:

      sum_ijk  tend[i,j,k] * volume_c[i,j,k] = 0

    Proof: the zonal flux divergence is a telescoping sum over a periodic
    domain, so the east fluxes and west fluxes cancel exactly when summed.
    """

    def test_upwind_random_phi_zonal_flow(self, flat_grid):
        grid = flat_grid
        key  = jax.random.PRNGKey(42)

        phi = jax.random.normal(key, (grid.Nx, grid.Ny, grid.Nz)) * grid.mask_c
        u   = jax.random.normal(key, (grid.Nx, grid.Ny, grid.Nz)) * grid.mask_u
        v   = jnp.zeros((grid.Nx, grid.Ny, grid.Nz))
        w   = jnp.zeros((grid.Nx, grid.Ny, grid.Nz + 1))

        tend          = upwind_advection(phi, u, v, w, grid)
        total_tend    = jnp.sum(tend * grid.volume_c * grid.mask_c)
        # Use a relative tolerance: |global tendency| / |total tracer flux|
        # must be < float32 round-off level (~1e-5).  An absolute tolerance
        # is inappropriate here because volume_c ~ O(1e13) m³, so even exact
        # cancellation accumulates float32 errors of O(10–100).
        total_flux    = jnp.sum(jnp.abs(tend) * grid.volume_c * grid.mask_c)
        rel_error     = jnp.abs(total_tend) / (total_flux + 1e-30)

        assert rel_error < 1e-4, (
            f"Upwind global conservation error (zonal flow): "
            f"abs={total_tend:.3e}, rel={rel_error:.3e}"
        )

    def test_centered_random_phi_zonal_flow(self, flat_grid):
        grid = flat_grid
        key  = jax.random.PRNGKey(7)

        phi = jax.random.normal(key, (grid.Nx, grid.Ny, grid.Nz)) * grid.mask_c
        u   = jax.random.normal(key, (grid.Nx, grid.Ny, grid.Nz)) * grid.mask_u
        v   = jnp.zeros((grid.Nx, grid.Ny, grid.Nz))
        w   = jnp.zeros((grid.Nx, grid.Ny, grid.Nz + 1))

        tend       = centered_advection(phi, u, v, w, grid)
        total_tend = jnp.sum(tend * grid.volume_c * grid.mask_c)
        total_flux = jnp.sum(jnp.abs(tend) * grid.volume_c * grid.mask_c)
        rel_error  = jnp.abs(total_tend) / (total_flux + 1e-30)

        assert rel_error < 1e-4, (
            f"Centered global conservation error (zonal flow): "
            f"abs={total_tend:.3e}, rel={rel_error:.3e}"
        )

    def test_upwind_zero_diffusion_conservation(self, flat_grid, default_params):
        """
        tracer_tendency (adv + kappa_h=0) is also globally conservative.
        """
        grid   = flat_grid
        params = default_params
        key    = jax.random.PRNGKey(13)

        phi = jax.random.normal(key, (grid.Nx, grid.Ny, grid.Nz)) * grid.mask_c
        u   = jax.random.normal(key, (grid.Nx, grid.Ny, grid.Nz)) * grid.mask_u
        v   = jnp.zeros((grid.Nx, grid.Ny, grid.Nz))
        w   = jnp.zeros((grid.Nx, grid.Ny, grid.Nz + 1))

        tend       = tracer_tendency(phi, u, v, w, kappa_h=0.0, grid=grid)
        total_tend = jnp.sum(tend * grid.volume_c * grid.mask_c)
        total_flux = jnp.sum(jnp.abs(tend) * grid.volume_c * grid.mask_c)
        rel_error  = jnp.abs(total_tend) / (total_flux + 1e-30)

        assert rel_error < 1e-4, (
            f"tracer_tendency global conservation error: "
            f"abs={total_tend:.3e}, rel={rel_error:.3e}"
        )


# ---------------------------------------------------------------------------
# 3. Surface forcing tendencies
# ---------------------------------------------------------------------------

class TestSurfaceForcingTendencies:
    """
    Verify sign, units, top-layer thickness conversion, and confinement
    of the surface forcing functions.
    """

    def test_surface_layer_confined_to_k0(self, flat_grid):
        """surface_layer_tendency must be nonzero only at k=0."""
        grid  = flat_grid
        flux  = jnp.ones((grid.Nx, grid.Ny))

        tend  = surface_layer_tendency(flux, grid)

        assert tend.shape == (grid.Nx, grid.Ny, grid.Nz)
        assert jnp.all(tend[:, :, 1:] == 0.0), "Nonzero tendency below surface layer"

    def test_surface_layer_magnitude(self, flat_grid):
        """tend = flux / dz_c[0] at k=0."""
        grid  = flat_grid
        flux  = jnp.full((grid.Nx, grid.Ny), 2.0)

        tend  = surface_layer_tendency(flux, grid)
        expected = 2.0 / float(grid.dz_c[0])

        assert jnp.allclose(tend[:, :, 0], expected, rtol=1e-5), (
            f"Expected {expected:.6f}, got {jnp.mean(tend[:,:,0]):.6f}"
        )

    def test_heat_flux_positive_warms(self, flat_grid, default_params):
        """Downward heat flux (positive) must produce a positive T tendency."""
        grid   = flat_grid
        params = default_params
        q      = jnp.full((grid.Nx, grid.Ny), 100.0)   # 100 W m-2 into ocean

        tend   = heat_surface_tendency(q, grid, params)

        assert jnp.all(tend[:, :, 0] > 0), "Positive heat flux should warm surface layer"
        assert jnp.all(tend[:, :, 1:] == 0.0), "Heat forcing leaked below surface layer"

    def test_heat_flux_units(self, flat_grid, default_params):
        """
        Verify: dT/dt [K s-1] = Q [W m-2] / (rho0 [kg m-3] * cp [J kg-1 K-1] * dz [m])
        """
        from OceanJAX.Physics.tracers import CP_SEAWATER
        grid   = flat_grid
        params = default_params
        q_val  = 200.0
        q      = jnp.full((grid.Nx, grid.Ny), q_val)

        tend   = heat_surface_tendency(q, grid, params)

        expected = q_val / (params.rho0 * CP_SEAWATER * float(grid.dz_c[0]))
        assert jnp.allclose(tend[:, :, 0], expected, rtol=1e-5), (
            f"Heat tendency: expected {expected:.3e} K/s, "
            f"got {jnp.mean(tend[:,:,0]):.3e} K/s"
        )

    def test_evaporation_increases_salinity(self, flat_grid, default_params):
        """Net evaporation (fw_flux > 0) must produce a positive S tendency."""
        grid   = flat_grid
        params = default_params
        ep     = jnp.full((grid.Nx, grid.Ny), 1e-7)   # 0.1 mm/s net evaporation

        tend   = salt_surface_tendency(ep, grid, params)

        assert jnp.all(tend[:, :, 0] > 0), "Net evaporation should increase salinity"
        assert jnp.all(tend[:, :, 1:] == 0.0), "Salt forcing leaked below surface layer"

    def test_freshwater_flux_units(self, flat_grid, default_params):
        """
        Verify: dS/dt [psu s-1] = S_ref [psu] * fw_flux [m s-1] / dz [m]
        """
        grid   = flat_grid
        params = default_params
        ep_val = 5e-8
        ep     = jnp.full((grid.Nx, grid.Ny), ep_val)

        tend     = salt_surface_tendency(ep, grid, params)
        expected = params.S_ref * ep_val / float(grid.dz_c[0])

        assert jnp.allclose(tend[:, :, 0], expected, rtol=1e-5), (
            f"Salt tendency: expected {expected:.3e} psu/s, "
            f"got {jnp.mean(tend[:,:,0]):.3e} psu/s"
        )
