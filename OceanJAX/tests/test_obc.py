"""
Tests for OceanJAX.Physics.obc
================================
Property-based tests for the open-boundary-condition layer.  Covers:

  1.  obc=None is bit-identical to the legacy no-kwarg path
      (already exercised by 128 existing tests; here checked explicitly).

  2.  Shape and dtype contract of OpenBCs and apply_obc.

  3.  eta relaxation with alpha_eta = 1 reproduces eta_ref exactly
      at i=0 and i=Nx-1 columns.

  4.  Closed-equivalent: u_ref = 0, U_col_ref = 0, eta_ref = 0,
      alpha_eta = 0 starting from a resting ocean keeps the state at rest
      (no spurious boundary-driven motion).

  5.  East-boundary inflow override: a state with u[-1] < 0 is overwritten
      to obc.u_ref[-1] both on the current and previous levels after apply_obc.

  6.  JIT compatibility: step(..., obc=obc) is jit-able and produces the
      same numerical result as the eager call.

Running
-------
    pytest OceanJAX/tests/test_obc.py -v
"""

from __future__ import annotations

import numpy as np
import jax
import jax.numpy as jnp
import pytest

from OceanJAX.grid import OceanGrid
from OceanJAX.state import ModelParams, create_rest_state
from OceanJAX.timeStepping import step
from OceanJAX.Physics.obc import OpenBCs, apply_obc


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def flat_grid():
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
    return create_rest_state(
        flat_grid,
        T_background=default_params.T_ref,
        S_background=default_params.S_ref,
    )


def _zero_obc(grid: OceanGrid, params: ModelParams, alpha_eta: float = 0.0) -> OpenBCs:
    """OpenBCs with all reference fields = 0 (closed-walls + no nudging)."""
    Nx, Ny, Nz = grid.Nx, grid.Ny, grid.Nz
    return OpenBCs(
        u_ref     = jnp.zeros((Nx, Ny, Nz), dtype=jnp.float32),
        T_ref     = jnp.full((Nx, Ny, Nz), params.T_ref, dtype=jnp.float32),
        S_ref     = jnp.full((Nx, Ny, Nz), params.S_ref, dtype=jnp.float32),
        eta_ref   = jnp.zeros((Nx, Ny), dtype=jnp.float32),
        U_col_ref = jnp.zeros((Nx, Ny), dtype=jnp.float32),
        alpha_eta = alpha_eta,
    )


# ---------------------------------------------------------------------------
# 1. obc=None bit-identical
# ---------------------------------------------------------------------------

def test_obc_none_matches_default(flat_grid, default_params, rest_state):
    """Passing obc=None explicitly must equal omitting the argument."""
    s1 = step(rest_state, flat_grid, default_params)
    s2 = step(rest_state, flat_grid, default_params, obc=None)
    for f in ("u", "v", "w", "T", "S", "eta", "u_prev", "v_prev", "eta_prev"):
        np.testing.assert_array_equal(getattr(s1, f), getattr(s2, f))


# ---------------------------------------------------------------------------
# 2. Shape / dtype contract
# ---------------------------------------------------------------------------

def test_openbcs_dataclass_shapes(flat_grid, default_params):
    obc = _zero_obc(flat_grid, default_params)
    Nx, Ny, Nz = flat_grid.Nx, flat_grid.Ny, flat_grid.Nz
    assert obc.u_ref.shape     == (Nx, Ny, Nz)
    assert obc.T_ref.shape     == (Nx, Ny, Nz)
    assert obc.S_ref.shape     == (Nx, Ny, Nz)
    assert obc.eta_ref.shape   == (Nx, Ny)
    assert obc.U_col_ref.shape == (Nx, Ny)
    assert isinstance(obc.alpha_eta, float)


def test_step_with_obc_returns_correct_shapes(flat_grid, default_params, rest_state):
    obc = _zero_obc(flat_grid, default_params, alpha_eta=0.5)
    new_state = step(rest_state, flat_grid, default_params, obc=obc)
    assert new_state.u.shape   == rest_state.u.shape
    assert new_state.v.shape   == rest_state.v.shape
    assert new_state.w.shape   == rest_state.w.shape
    assert new_state.T.shape   == rest_state.T.shape
    assert new_state.S.shape   == rest_state.S.shape
    assert new_state.eta.shape == rest_state.eta.shape


# ---------------------------------------------------------------------------
# 3. eta relaxation with alpha_eta = 1
# ---------------------------------------------------------------------------

def test_eta_relaxation_alpha_one(flat_grid, default_params, rest_state):
    """alpha_eta = 1 hard-sets boundary eta to eta_ref."""
    Nx, Ny = flat_grid.Nx, flat_grid.Ny
    # Reference target: non-trivial pattern at boundary columns
    eta_target = jnp.zeros((Nx, Ny), dtype=jnp.float32)
    eta_target = eta_target.at[0,  :].set(0.5)
    eta_target = eta_target.at[-1, :].set(-0.3)

    obc = OpenBCs(
        u_ref     = jnp.zeros((Nx, Ny, flat_grid.Nz), dtype=jnp.float32),
        T_ref     = jnp.full((Nx, Ny, flat_grid.Nz), default_params.T_ref, dtype=jnp.float32),
        S_ref     = jnp.full((Nx, Ny, flat_grid.Nz), default_params.S_ref, dtype=jnp.float32),
        eta_ref   = eta_target,
        U_col_ref = jnp.zeros((Nx, Ny), dtype=jnp.float32),
        alpha_eta = 1.0,
    )

    new_state = apply_obc(rest_state, flat_grid, obc)
    # i=0 and i=Nx-1 columns should equal eta_target after relaxation
    np.testing.assert_allclose(np.asarray(new_state.eta[0,  :]),  np.asarray(eta_target[0,  :]),  atol=1e-7)
    np.testing.assert_allclose(np.asarray(new_state.eta[-1, :]),  np.asarray(eta_target[-1, :]),  atol=1e-7)
    # Interior columns unchanged
    np.testing.assert_array_equal(np.asarray(new_state.eta[1:-1, :]),
                                   np.asarray(rest_state.eta[1:-1, :]))
    # eta_prev must be relaxed identically (leapfrog history consistency)
    np.testing.assert_allclose(np.asarray(new_state.eta_prev[0,  :]),  np.asarray(eta_target[0,  :]),  atol=1e-7)
    np.testing.assert_allclose(np.asarray(new_state.eta_prev[-1, :]),  np.asarray(eta_target[-1, :]),  atol=1e-7)


# ---------------------------------------------------------------------------
# 4. Closed-equivalent: zero references + resting ocean stays at rest
# ---------------------------------------------------------------------------

def test_rest_ocean_with_zero_obc_stays_at_rest(flat_grid, default_params, rest_state):
    """
    With u_ref = 0, U_col_ref = 0, eta_ref = 0, alpha_eta = 0 and a
    resting initial state, the model should not generate boundary motion.

    The OBC pathway replaces the periodic wrap at west with U_col_ref=0
    (i.e. zero transport into i=0 from outside) and clamps east inflow
    to 0 — for a state with u ≡ 0 these are identical to the resting
    state, so the velocity, w and eta fields must all stay at machine zero.
    """
    obc = _zero_obc(flat_grid, default_params, alpha_eta=0.0)
    s = rest_state
    for _ in range(5):
        s = step(s, flat_grid, default_params, obc=obc)

    # Velocities, w, eta should remain at machine zero
    assert float(jnp.max(jnp.abs(s.u)))   < 1e-6
    assert float(jnp.max(jnp.abs(s.v)))   < 1e-6
    assert float(jnp.max(jnp.abs(s.w)))   < 1e-6
    assert float(jnp.max(jnp.abs(s.eta))) < 1e-6
    # T, S should remain very close to background (small float32 drift OK)
    np.testing.assert_allclose(np.asarray(s.T), np.asarray(rest_state.T), atol=1e-3)
    np.testing.assert_allclose(np.asarray(s.S), np.asarray(rest_state.S), atol=1e-3)


# ---------------------------------------------------------------------------
# 5. East-boundary inflow override on u and u_prev
# ---------------------------------------------------------------------------

def test_east_inflow_u_override(flat_grid, default_params, rest_state):
    """
    When u[-1] < 0 (inflow from east), apply_obc replaces both u[-1] and
    u_prev[-1] with obc.u_ref[-1] at every (j, k) where the prognostic
    velocity is negative.  Outflow points are untouched.
    """
    Nx, Ny, Nz = flat_grid.Nx, flat_grid.Ny, flat_grid.Nz

    # Build a state with mixed inflow / outflow at the east face.
    u = np.zeros((Nx, Ny, Nz), dtype=np.float32)
    # Inflow at the first two j's, outflow at the others
    u[-1, :Ny // 2, :] = -0.1
    u[-1,  Ny // 2:, :] = +0.2
    u_jnp = jnp.asarray(u)

    state = type(rest_state)(
        u   = u_jnp,
        v   = rest_state.v,
        w   = rest_state.w,
        T   = rest_state.T,
        S   = rest_state.S,
        eta = rest_state.eta,
        u_prev   = u_jnp,
        v_prev   = rest_state.v_prev,
        eta_prev = rest_state.eta_prev,
        T_tend_prev  = rest_state.T_tend_prev,
        S_tend_prev  = rest_state.S_tend_prev,
        T_tend_prev2 = rest_state.T_tend_prev2,
        S_tend_prev2 = rest_state.S_tend_prev2,
        time       = rest_state.time,
        step_count = rest_state.step_count,
    )

    # Reference u: distinct non-zero pattern so we can detect the override
    u_ref = jnp.zeros((Nx, Ny, Nz), dtype=jnp.float32)
    u_ref = u_ref.at[-1, :, :].set(-0.05)

    obc = OpenBCs(
        u_ref     = u_ref,
        T_ref     = jnp.full((Nx, Ny, Nz), default_params.T_ref, dtype=jnp.float32),
        S_ref     = jnp.full((Nx, Ny, Nz), default_params.S_ref, dtype=jnp.float32),
        eta_ref   = jnp.zeros((Nx, Ny), dtype=jnp.float32),
        U_col_ref = jnp.zeros((Nx, Ny), dtype=jnp.float32),
        alpha_eta = 0.0,
    )

    new_state = apply_obc(state, flat_grid, obc)

    # Inflow rows (j < Ny/2): u_east should be obc.u_ref[-1, j, :] (gated by mask_u)
    expected_inflow = np.asarray(u_ref[-1, :Ny // 2, :]) * np.asarray(flat_grid.mask_u[-1, :Ny // 2, :])
    np.testing.assert_allclose(np.asarray(new_state.u[-1, :Ny // 2, :]),       expected_inflow, atol=1e-7)
    np.testing.assert_allclose(np.asarray(new_state.u_prev[-1, :Ny // 2, :]),  expected_inflow, atol=1e-7)

    # Outflow rows (j >= Ny/2): u unchanged
    expected_outflow = np.asarray(state.u[-1, Ny // 2:, :])
    np.testing.assert_allclose(np.asarray(new_state.u[-1, Ny // 2:, :]),       expected_outflow, atol=1e-7)


# ---------------------------------------------------------------------------
# 6. JIT compatibility
# ---------------------------------------------------------------------------

def test_step_with_obc_jit_matches_eager(flat_grid, default_params, rest_state):
    obc = _zero_obc(flat_grid, default_params, alpha_eta=0.5)
    eager  = step(rest_state, flat_grid, default_params, obc=obc)
    jitted = jax.jit(
        lambda s: step(s, flat_grid, default_params, obc=obc)
    )(rest_state)
    for f in ("u", "v", "w", "T", "S", "eta", "u_prev", "v_prev", "eta_prev"):
        np.testing.assert_allclose(
            np.asarray(getattr(eager, f)), np.asarray(getattr(jitted, f)),
            atol=1e-6, rtol=1e-5,
        )
