"""
OceanJAX Time Stepper
=====================
Coordinates one model time step (and multi-step runs) by orchestrating the
Physics modules in the correct order.

Splitting strategy
------------------
Momentum  — Leapfrog with Asselin-Robert filter (suppresses the computational
             mode that leapfrog generates).  Vertical viscosity is applied
             implicitly after the leapfrog advance.

Tracers   — Adams-Bashforth 3rd order (AB3) for the explicit part; vertical
             diffusion is applied implicitly afterwards.  The two previous
             tendencies stored in OceanState are used for the AB3 weights.
             For the first two steps the history fields are zero, so the
             scheme degrades gracefully to AB1 → AB2; the error is confined
             to the startup transient and does not affect long integrations.

Free surface — Forward Euler update from the depth-integrated divergence.
               The diagnosed w field is then updated so that its surface face
               (k=0) carries the kinematic signal w[0] = deta/dt.

Division of labour with Physics modules
----------------------------------------
  dynamics.py  : equation_of_state, hydrostatic_pressure computed ONCE per
                 step and passed to both momentum tendency functions.
  mixing.py    : implicit_vertical_visc / implicit_vertical_mix applied AFTER
                 explicit advances; not called inside tendency functions.
  tracers.py   : tracer_tendency (advection + horizontal diffusion) and the
                 surface forcing tendencies.

Surface forcing
---------------
All surface fluxes are packaged in ``SurfaceForcing``.  Wind stress enters
the momentum equations as a surface-layer tendency analogous to the tracer
heat / freshwater fluxes.  Passing ``forcing=None`` disables all surface
inputs (useful for spin-up or idealised experiments).

Contents
--------
  SurfaceForcing    – equinox Module holding surface flux fields
  step              – advance OceanState by one dt
  run               – lax.scan wrapper for multi-step integrations
"""

from __future__ import annotations

from typing import Optional

import jax
import jax.numpy as jnp
import equinox as eqx

from OceanJAX.grid import OceanGrid
from OceanJAX.state import OceanState, ModelParams
from OceanJAX.Physics.dynamics import (
    equation_of_state,
    hydrostatic_pressure,
    momentum_tendency_u,
    momentum_tendency_v,
    free_surface_tendency,
    compute_w,
)
from OceanJAX.Physics.tracers import (
    tracer_tendency,
    heat_surface_tendency,
    salt_surface_tendency,
)
from OceanJAX.Physics.mixing import (
    implicit_vertical_visc,
    implicit_vertical_mix,
)


# ---------------------------------------------------------------------------
# Surface forcing container
# ---------------------------------------------------------------------------

class SurfaceForcing(eqx.Module):
    """
    Surface boundary conditions for one time step.

    All fields are (Nx, Ny) arrays in SI units.  Pass ``forcing=None``
    to ``step`` or ``run`` to disable all surface inputs.

    Fields
    ------
    heat_flux : (Nx, Ny) [W m-2]   Net downward heat flux into the ocean.
                                    Positive = ocean gains heat.
    fw_flux   : (Nx, Ny) [m s-1]   Net evaporation minus precipitation (E-P).
                                    Positive = net freshwater loss → salinity increase.
    tau_x     : (Nx, Ny) [N m-2]   Zonal wind stress at the sea surface.
                                    Positive eastward.
    tau_y     : (Nx, Ny) [N m-2]   Meridional wind stress at the sea surface.
                                    Positive northward.
    """
    heat_flux: jnp.ndarray
    fw_flux:   jnp.ndarray
    tau_x:     jnp.ndarray
    tau_y:     jnp.ndarray


# ---------------------------------------------------------------------------
# Internal: wind stress surface tendencies
# ---------------------------------------------------------------------------

def _wind_tendency_u(
    tau_x:  jnp.ndarray,
    grid:   OceanGrid,
    params: ModelParams,
) -> jnp.ndarray:
    """
    Zonal momentum tendency from surface wind stress [m s-2].

    Applies the stress to the top layer (k=0) only:

      du/dt|_wind = tau_x / (rho0 * dz_c[0])   at k = 0
                 = 0                             at k > 0

    The sign convention matches the momentum equation:
    positive tau_x (eastward stress) accelerates u.

    Args:
        tau_x  : (Nx, Ny) [N m-2]
        grid   : OceanGrid
        params : ModelParams  (uses rho0)

    Returns:
        (Nx, Ny, Nz) [m s-2]
    """
    tend_surf = tau_x / (params.rho0 * grid.dz_c[0]) * grid.mask_u[:, :, 0]
    return jnp.zeros(
        (grid.Nx, grid.Ny, grid.Nz), dtype=tau_x.dtype
    ).at[:, :, 0].set(tend_surf)


def _wind_tendency_v(
    tau_y:  jnp.ndarray,
    grid:   OceanGrid,
    params: ModelParams,
) -> jnp.ndarray:
    """
    Meridional momentum tendency from surface wind stress [m s-2].

    Mirrors ``_wind_tendency_u`` for the v-equation.

    Args:
        tau_y  : (Nx, Ny) [N m-2]
        grid   : OceanGrid
        params : ModelParams  (uses rho0)

    Returns:
        (Nx, Ny, Nz) [m s-2]
    """
    tend_surf = tau_y / (params.rho0 * grid.dz_c[0]) * grid.mask_v[:, :, 0]
    return jnp.zeros(
        (grid.Nx, grid.Ny, grid.Nz), dtype=tau_y.dtype
    ).at[:, :, 0].set(tend_surf)


# ---------------------------------------------------------------------------
# Single time step
# ---------------------------------------------------------------------------

def step(
    state:   OceanState,
    grid:    OceanGrid,
    params:  ModelParams,
    forcing: Optional[SurfaceForcing] = None,
) -> OceanState:
    """
    Advance the model state by one time step dt.

    Calling sequence
    ----------------
    1.  Shared diagnostics (EOS + hydrostatic pressure, computed once).
    2.  Explicit momentum tendencies (Coriolis + PGF + horiz. viscosity).
        + Wind stress surface forcing (if supplied).
    3.  Leapfrog momentum advance:  u_new = u_prev + 2*dt * G_u
    4.  Asselin-Robert filter on u(n):
          u_filt = u + alpha*(u_new - 2*u + u_prev)
    5.  Implicit vertical viscosity applied to u_new, v_new.
    6.  Explicit tracer tendencies (advection + horiz. diffusion).
        + Heat flux / freshwater surface forcing (if supplied).
    7.  Adams-Bashforth 3 tracer advance.
    8.  Implicit vertical diffusion applied to T_new, S_new.
    9.  Free-surface update: eta_new = eta + dt * deta_dt
    10. Diagnose w from continuity; set w[0] = deta_dt (kinematic BC).
    11. Apply all masks and assemble new OceanState.

    Args:
        state   : OceanState at time n
        grid    : OceanGrid (static)
        params  : ModelParams
        forcing : SurfaceForcing for this time step, or None

    Returns:
        OceanState at time n+1
    """
    dt = params.dt

    # ------------------------------------------------------------------
    # 1. Shared diagnostics
    # ------------------------------------------------------------------
    rho         = equation_of_state(state.T, state.S, params)
    p_hyd_prime = hydrostatic_pressure(rho, grid, params)

    # ------------------------------------------------------------------
    # 2. Explicit momentum tendencies
    # ------------------------------------------------------------------
    G_u = momentum_tendency_u(state, p_hyd_prime, grid, params)
    G_v = momentum_tendency_v(state, p_hyd_prime, grid, params)

    if forcing is not None:
        G_u = G_u + _wind_tendency_u(forcing.tau_x, grid, params)
        G_v = G_v + _wind_tendency_v(forcing.tau_y, grid, params)

    # ------------------------------------------------------------------
    # 3. Leapfrog momentum advance
    # ------------------------------------------------------------------
    u_new = (state.u_prev + 2.0 * dt * G_u) * grid.mask_u
    v_new = (state.v_prev + 2.0 * dt * G_v) * grid.mask_v

    # ------------------------------------------------------------------
    # 4. Asselin-Robert filter on current-time-level u(n), v(n)
    #    This damps the spurious computational mode of the leapfrog
    #    scheme; the filtered values become u_prev for the next step.
    # ------------------------------------------------------------------
    alpha  = params.asselin_coeff
    u_filt = (state.u + alpha * (u_new - 2.0 * state.u + state.u_prev)) * grid.mask_u
    v_filt = (state.v + alpha * (v_new - 2.0 * state.v + state.v_prev)) * grid.mask_v

    # ------------------------------------------------------------------
    # 5. Implicit vertical viscosity (applied to the leapfrog result)
    # ------------------------------------------------------------------
    u_new = implicit_vertical_visc(u_new, params.nu_v, dt, grid, grid.mask_u)
    v_new = implicit_vertical_visc(v_new, params.nu_v, dt, grid, grid.mask_v)

    # ------------------------------------------------------------------
    # 6. Explicit tracer tendencies
    #    Use Asselin-filtered velocities for advection so that the
    #    tracer and momentum fields see a consistent velocity state.
    # ------------------------------------------------------------------
    # Use current w (not yet updated) for tracer advection
    G_T = tracer_tendency(state.T, u_filt, v_filt, state.w, params.kappa_h, grid)
    G_S = tracer_tendency(state.S, u_filt, v_filt, state.w, params.kappa_h, grid)

    if forcing is not None:
        G_T = G_T + heat_surface_tendency(forcing.heat_flux, grid, params)
        G_S = G_S + salt_surface_tendency(forcing.fw_flux,   grid, params)

    # ------------------------------------------------------------------
    # 7. Adams-Bashforth 3 tracer advance
    #    Coefficients from params.ab3_coeffs = (a0, a1, a2).
    #    For the first two steps T_tend_prev and T_tend_prev2 are zero
    #    (set in create_zero_state / create_rest_state), so the scheme
    #    degrades gracefully: step 0 → AB1, step 1 → AB2, step 2+ → AB3.
    # ------------------------------------------------------------------
    a0, a1, a2 = params.ab3_coeffs
    T_new = (state.T + dt * (a0 * G_T
                             + a1 * state.T_tend_prev
                             + a2 * state.T_tend_prev2)) * grid.mask_c
    S_new = (state.S + dt * (a0 * G_S
                             + a1 * state.S_tend_prev
                             + a2 * state.S_tend_prev2)) * grid.mask_c

    # ------------------------------------------------------------------
    # 8. Implicit vertical diffusion
    # ------------------------------------------------------------------
    T_new = implicit_vertical_mix(T_new, params.kappa_v, dt, grid,
                                  rhs_explicit=jnp.zeros_like(T_new))
    S_new = implicit_vertical_mix(S_new, params.kappa_v, dt, grid,
                                  rhs_explicit=jnp.zeros_like(S_new))

    # ------------------------------------------------------------------
    # 9. Free-surface update
    # ------------------------------------------------------------------
    deta_dt = free_surface_tendency(u_filt, v_filt, grid)
    eta_new = (state.eta + dt * deta_dt) * grid.mask_c[:, :, 0]

    # ------------------------------------------------------------------
    # 10. Diagnose w; impose kinematic BC w[0] = deta/dt
    # ------------------------------------------------------------------
    w_new = compute_w(u_new, v_new, grid)
    # Set the surface face to the kinematic signal; mask_w gates it.
    w_new = w_new.at[:, :, 0].set(deta_dt * grid.mask_w[:, :, 0])

    # ------------------------------------------------------------------
    # 11. Assemble new state
    # ------------------------------------------------------------------
    return OceanState(
        u   = u_new,
        v   = v_new,
        w   = w_new,
        T   = T_new,
        S   = S_new,
        eta = eta_new,
        # Asselin-filtered velocities become the "previous" level
        u_prev = u_filt,
        v_prev = v_filt,
        # Rotate tendency history for AB3
        T_tend_prev  = G_T,
        S_tend_prev  = G_S,
        T_tend_prev2 = state.T_tend_prev,
        S_tend_prev2 = state.S_tend_prev,
        time = state.time + dt,
    )


# ---------------------------------------------------------------------------
# Multi-step run via lax.scan
# ---------------------------------------------------------------------------

def run(
    state:            OceanState,
    grid:             OceanGrid,
    params:           ModelParams,
    n_steps:          int,
    forcing_sequence: Optional[SurfaceForcing] = None,
    save_history:     bool = False,
) -> tuple[OceanState, Optional[OceanState]]:
    """
    Integrate the model for ``n_steps`` time steps using ``jax.lax.scan``.

    The scan loop is JIT-compilable and supports reverse-mode AD through
    the entire trajectory, enabling gradient-based optimisation over model
    parameters or initial conditions.

    Args:
        state            : Initial OceanState
        grid             : OceanGrid (static)
        params           : ModelParams
        n_steps          : Number of time steps to advance
        forcing_sequence : SurfaceForcing whose array fields have a leading
                           time axis of length n_steps, e.g.
                           ``heat_flux.shape == (n_steps, Nx, Ny)``.
                           Pass ``None`` for no surface forcing.
        save_history     : If True, return all intermediate OceanState
                           snapshots (memory-intensive for long runs).
                           If False, only the final state is returned and
                           history is None.

    Returns:
        (final_state, history)
        final_state : OceanState after n_steps
        history     : OceanState with a leading time axis of length n_steps
                      if save_history=True, else None.
    """
    def _step_fn(carry: OceanState, forcing_t: Optional[SurfaceForcing]):
        new_state = step(carry, grid, params, forcing_t)
        output    = new_state if save_history else None
        return new_state, output

    if forcing_sequence is None:
        # No time-varying forcing: scan over a dummy axis
        final_state, history = jax.lax.scan(
            lambda carry, _: _step_fn(carry, None),
            state,
            None,
            length=n_steps,
        )
    else:
        final_state, history = jax.lax.scan(
            _step_fn,
            state,
            forcing_sequence,
        )

    return final_state, history
