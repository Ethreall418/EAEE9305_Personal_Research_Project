"""
OceanJAX Physics – Open Boundary Conditions
=============================================
East/west open-boundary support for the otherwise zonally-periodic model.

Why this module exists
----------------------
All horizontal operators in ``operators.py`` close the x-axis with
``jnp.roll`` (periodic).  Initialising from an ORAS5 full state (which
includes basin-scale meridional overturning) in a closed periodic box
produces a mass-balance error that grows ``eta`` without bound.  This
module provides the small set of reference fields and a single
``apply_obc`` helper that injects ORAS5 boundary information back into
the model state each time step.

Design (north/south are walls already; only x is opened):

  West face (i=0, implicit – no u array index)
    Dynamics (eta, w) : replace the periodic wrap with the depth-integrated
                        reference transport ``U_col_ref`` at i=0.
    Tracers           : west face is a hard wall (zero advective flux);
                        horizontal diffusion sees zero flux as well.
  East face (u[Nx-1])
    Dynamics (eta, w) : outflow (u >= 0) free; inflow (u < 0) overridden
                        with ``u_ref``.
    Tracers           : outflow upwinds from the interior; inflow upwinds
                        from the reference value ``phi_ref[Nx-1]``.
                        Horizontal diffusion uses Neumann at i=Nx-1.
  eta                 : i=0 and i=Nx-1 columns relaxed toward ``eta_ref``
                        each step (fractional blend ``alpha_eta``).
                        ``eta_prev`` is relaxed identically to keep the
                        leapfrog history consistent.

Plugging into the time stepper
------------------------------
All physics functions that touch the zonal boundary take an optional
``obc`` keyword (default ``None``).  When ``obc is None`` the behaviour
is bit-identical to the pre-OBC model.  When supplied, the time stepper
also calls ``apply_obc`` at the end of each step to enforce the eta
relaxation and east-boundary inflow override.
"""

from __future__ import annotations

import equinox as eqx
import jax.numpy as jnp

from OceanJAX.grid import OceanGrid


# ---------------------------------------------------------------------------
# OpenBCs container
# ---------------------------------------------------------------------------

class OpenBCs(eqx.Module):
    """
    Reference fields and parameters for east/west open boundaries.

    Fields
    ------
    u_ref     : (Nx, Ny, Nz) [m s-1]
                Reference zonal velocity at u-points.  Only the i=Nx-1 column
                is consulted (east-boundary inflow override).
    T_ref     : (Nx, Ny, Nz) [degC]
                Reference temperature at cell centres.  Only the i=Nx-1 column
                is consulted by tracer advection (east-boundary inflow upwind).
    S_ref     : (Nx, Ny, Nz) [psu]
                Reference salinity at cell centres; usage mirrors T_ref.
    eta_ref   : (Nx, Ny) [m]
                Reference SSH.  Both i=0 and i=Nx-1 columns are used by
                ``apply_obc`` to nudge eta toward the reference value.
    U_col_ref : (Nx, Ny) [m^2 s-1]
                Reference depth-integrated zonal transport.  Only the i=0
                column is consulted (west-boundary flux replacement in
                ``free_surface_tendency`` and ``compute_w``).
    alpha_eta : float, static
                Fractional eta relaxation per step (0..1).  0 disables the
                relaxation; 1 hard-sets eta to eta_ref at the boundaries.
                A value around 0.5 amounts to fast nudging (~ 1-step damping).
    """
    u_ref:     jnp.ndarray
    T_ref:     jnp.ndarray
    S_ref:     jnp.ndarray
    eta_ref:   jnp.ndarray
    U_col_ref: jnp.ndarray
    alpha_eta: float = eqx.field(static=True)


# ---------------------------------------------------------------------------
# apply_obc – end-of-step relaxation and east-boundary override
# ---------------------------------------------------------------------------

def apply_obc(
    state,
    grid: OceanGrid,
    obc:  OpenBCs,
):
    """
    Apply the boundary relaxation and east-boundary inflow override.

    Operations performed in-place on a returned new ``OceanState``:

      1. Blend ``eta`` and ``eta_prev`` toward ``eta_ref`` at i=0 and i=Nx-1:

           eta_new[i, :] = (1 - alpha) * eta[i, :]  +  alpha * eta_ref[i, :]

         Relaxing both the current and previous level keeps the
         Robert-Asselin filter from amplifying boundary discontinuities.

      2. At i=Nx-1, where ``u[i, :, :] < 0`` (inflow from the east), set
         ``u`` and ``u_prev`` to ``u_ref`` so that subsequent tracer
         advection and free-surface diagnostics see consistent inflow
         velocities.  Outflow (``u >= 0``) is left untouched.

    The function returns a fresh OceanState with the relaxed/overridden
    fields; all other fields are unchanged.

    Args:
        state : OceanState after the explicit + implicit update.
        grid  : OceanGrid.
        obc   : OpenBCs with reference fields and ``alpha_eta``.

    Returns:
        OceanState with the boundary fields updated.
    """
    alpha     = obc.alpha_eta
    surf_mask = grid.mask_c[:, :, 0]

    # --- eta relaxation at i=0 and i=Nx-1 ---------------------------------
    eta      = state.eta
    eta_prev = state.eta_prev

    eta      = eta.at[0,  :].set((1.0 - alpha) * eta[0,  :] + alpha * obc.eta_ref[0,  :])
    eta      = eta.at[-1, :].set((1.0 - alpha) * eta[-1, :] + alpha * obc.eta_ref[-1, :])
    eta_prev = eta_prev.at[0,  :].set((1.0 - alpha) * eta_prev[0,  :] + alpha * obc.eta_ref[0,  :])
    eta_prev = eta_prev.at[-1, :].set((1.0 - alpha) * eta_prev[-1, :] + alpha * obc.eta_ref[-1, :])

    eta      = eta      * surf_mask
    eta_prev = eta_prev * surf_mask

    # --- East-boundary u override on inflow -------------------------------
    u      = state.u
    u_prev = state.u_prev
    mask_u_east = grid.mask_u[-1, :, :]

    inflow_mask  = (u[-1, :, :] < 0.0).astype(u.dtype)
    u_east       = (inflow_mask * obc.u_ref[-1, :, :]
                    + (1.0 - inflow_mask) * u[-1, :, :]) * mask_u_east
    u_prev_east  = (inflow_mask * obc.u_ref[-1, :, :]
                    + (1.0 - inflow_mask) * u_prev[-1, :, :]) * mask_u_east

    u      = u.at[-1, :, :].set(u_east)
    u_prev = u_prev.at[-1, :, :].set(u_prev_east)

    return eqx.tree_at(
        lambda s: (s.eta, s.eta_prev, s.u, s.u_prev),
        state,
        (eta, eta_prev, u, u_prev),
    )
