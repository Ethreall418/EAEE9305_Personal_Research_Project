"""
OceanJAX State Module
=====================
Defines OceanState and ModelParams as equinox Modules.

OceanState holds all prognostic and auxiliary model fields.
ModelParams holds all scalar physical and numerical parameters.
"""

from __future__ import annotations

from typing import Optional

import equinox as eqx
import jax.numpy as jnp

from OceanJAX.grid import OceanGrid


# ---------------------------------------------------------------------------
# ModelParams
# ---------------------------------------------------------------------------
class ModelParams(eqx.Module):
    """
    Scalar physical and numerical parameters.
    All fields are JAX-differentiable floats unless marked static.
    """
    # ---- equation of state ------------------------------------------------
    g:       float   # gravitational acceleration [m s-2]
    rho0:    float   # reference density [kg m-3]
    alpha_T: float   # thermal expansion coefficient [K-1]
    beta_S:  float   # haline contraction coefficient [psu-1]
    T_ref:   float   # reference temperature [degC]
    S_ref:   float   # reference salinity [psu]

    # ---- viscosity / diffusivity -----------------------------------------
    nu_h:    float   # horizontal viscosity [m^2 s-1]
    nu_v:    float   # background vertical viscosity [m^2 s-1]
    kappa_h: float   # horizontal tracer diffusivity [m^2 s-1]
    kappa_v: float   # background vertical tracer diffusivity [m^2 s-1]

    # ---- time stepping ----------------------------------------------------
    dt:            float   # timestep [s]
    asselin_coeff: float   # Asselin–Robert filter coefficient (default 0.1)

    # Adams-Bashforth 3 coefficients (length-3 tuple treated as static)
    # Default: AB2 = (3/2, -1/2); AB3 = (23/12, -16/12, 5/12)
    ab3_coeffs: tuple = eqx.field(static=True)

    def __init__(
        self,
        g: float = 9.81,
        rho0: float = 1025.0,
        alpha_T: float = 2e-4,
        beta_S: float = 7.4e-4,
        T_ref: float = 10.0,
        S_ref: float = 35.0,
        nu_h: float = 200.0,
        nu_v: float = 1e-4,
        kappa_h: float = 100.0,
        kappa_v: float = 1e-5,
        dt: float = 900.0,
        asselin_coeff: float = 0.1,
        ab3_coeffs: tuple = (23.0 / 12.0, -16.0 / 12.0, 5.0 / 12.0),
    ):
        self.g            = g
        self.rho0         = rho0
        self.alpha_T      = alpha_T
        self.beta_S       = beta_S
        self.T_ref        = T_ref
        self.S_ref        = S_ref
        self.nu_h         = nu_h
        self.nu_v         = nu_v
        self.kappa_h      = kappa_h
        self.kappa_v      = kappa_v
        self.dt           = dt
        self.asselin_coeff = asselin_coeff
        self.ab3_coeffs   = ab3_coeffs


# ---------------------------------------------------------------------------
# OceanState
# ---------------------------------------------------------------------------
class OceanState(eqx.Module):
    """
    Prognostic and auxiliary fields for the ocean model.

    Velocity on Arakawa C-grid:
        u (Nx, Ny, Nz)    - zonal velocity at east faces
        v (Nx, Ny, Nz)    - meridional velocity at north faces
        w (Nx, Ny, Nz+1)  - vertical velocity at top faces

    Tracers at cell centres:
        T (Nx, Ny, Nz)    - potential temperature [degC]
        S (Nx, Ny, Nz)    - practical salinity [psu]

    Sea-surface height:
        eta (Nx, Ny)      - SSH [m]

    Time-stepping history (leapfrog / AB3):
        u_prev, v_prev         - velocity at time n-1  (Nx, Ny, Nz)
        eta_prev               - SSH at time n-1 (Nx, Ny); leapfrog history
        T_tend_prev            - T tendency at time n-1 (Nx, Ny, Nz)
        S_tend_prev            - S tendency at time n-1 (Nx, Ny, Nz)
        T_tend_prev2           - T tendency at time n-2 (for full AB3)
        S_tend_prev2           - S tendency at time n-2

    Scalar:
        time      [s]
        step_count  number of steps taken (int32); used to select AB1/AB2/AB3
                    bootstrap coefficients in the first two steps.
    """
    u: jnp.ndarray           # (Nx, Ny, Nz)
    v: jnp.ndarray           # (Nx, Ny, Nz)
    w: jnp.ndarray           # (Nx, Ny, Nz+1)
    T: jnp.ndarray           # (Nx, Ny, Nz)
    S: jnp.ndarray           # (Nx, Ny, Nz)
    eta: jnp.ndarray         # (Nx, Ny)

    # leapfrog history
    u_prev:   jnp.ndarray    # (Nx, Ny, Nz)
    v_prev:   jnp.ndarray    # (Nx, Ny, Nz)
    eta_prev: jnp.ndarray    # (Nx, Ny)

    # Adams-Bashforth history (two previous tendencies for AB3)
    T_tend_prev:  jnp.ndarray   # (Nx, Ny, Nz)
    S_tend_prev:  jnp.ndarray   # (Nx, Ny, Nz)
    T_tend_prev2: jnp.ndarray   # (Nx, Ny, Nz)
    S_tend_prev2: jnp.ndarray   # (Nx, Ny, Nz)

    time:       jnp.ndarray  # scalar float32
    step_count: jnp.ndarray  # scalar int32

    # ------------------------------------------------------------------
    # apply_masks
    # ------------------------------------------------------------------
    def apply_masks(self, grid: OceanGrid) -> "OceanState":
        """Return a new OceanState with all land points zeroed."""
        return OceanState(
            u  = self.u   * grid.mask_u,
            v  = self.v   * grid.mask_v,
            w  = self.w   * grid.mask_w,
            T  = self.T   * grid.mask_c,
            S  = self.S   * grid.mask_c,
            eta      = self.eta      * grid.mask_c[:, :, 0],
            u_prev   = self.u_prev   * grid.mask_u,
            v_prev   = self.v_prev   * grid.mask_v,
            eta_prev = self.eta_prev * grid.mask_c[:, :, 0],
            T_tend_prev  = self.T_tend_prev  * grid.mask_c,
            S_tend_prev  = self.S_tend_prev  * grid.mask_c,
            T_tend_prev2 = self.T_tend_prev2 * grid.mask_c,
            S_tend_prev2 = self.S_tend_prev2 * grid.mask_c,
            time       = self.time,
            step_count = self.step_count,
        )


# ---------------------------------------------------------------------------
# Factory functions
# ---------------------------------------------------------------------------
def create_zero_state(grid: OceanGrid) -> OceanState:
    """Create an OceanState with all fields initialised to zero."""
    Nx, Ny, Nz = grid.Nx, grid.Ny, grid.Nz
    z = jnp.zeros
    return OceanState(
        u  = z((Nx, Ny, Nz),    dtype=jnp.float32),
        v  = z((Nx, Ny, Nz),    dtype=jnp.float32),
        w  = z((Nx, Ny, Nz+1),  dtype=jnp.float32),
        T  = z((Nx, Ny, Nz),    dtype=jnp.float32),
        S  = z((Nx, Ny, Nz),    dtype=jnp.float32),
        eta      = z((Nx, Ny),        dtype=jnp.float32),
        u_prev   = z((Nx, Ny, Nz), dtype=jnp.float32),
        v_prev   = z((Nx, Ny, Nz), dtype=jnp.float32),
        eta_prev = z((Nx, Ny),     dtype=jnp.float32),
        T_tend_prev  = z((Nx, Ny, Nz), dtype=jnp.float32),
        S_tend_prev  = z((Nx, Ny, Nz), dtype=jnp.float32),
        T_tend_prev2 = z((Nx, Ny, Nz), dtype=jnp.float32),
        S_tend_prev2 = z((Nx, Ny, Nz), dtype=jnp.float32),
        time       = jnp.array(0.0, dtype=jnp.float32),
        step_count = jnp.array(0,   dtype=jnp.int32),
    )


def create_from_arrays(
    grid: OceanGrid,
    u: jnp.ndarray,
    v: jnp.ndarray,
    T: jnp.ndarray,
    S: jnp.ndarray,
    eta: jnp.ndarray,
    w: Optional[jnp.ndarray] = None,
    time: float = 0.0,
) -> OceanState:
    """
    Create OceanState from supplied prognostic arrays.
    w is diagnosed from continuity if not provided.
    History fields are initialised to zero.
    """
    from OceanJAX.Physics.dynamics import compute_w  # deferred import to avoid circularity

    Nx, Ny, Nz = grid.Nx, grid.Ny, grid.Nz
    z = jnp.zeros

    u  = jnp.asarray(u,   dtype=jnp.float32)
    v  = jnp.asarray(v,   dtype=jnp.float32)
    T  = jnp.asarray(T,   dtype=jnp.float32)
    S  = jnp.asarray(S,   dtype=jnp.float32)
    eta= jnp.asarray(eta, dtype=jnp.float32)

    if w is None:
        w = compute_w(u, v, grid)
    else:
        w = jnp.asarray(w, dtype=jnp.float32)

    state = OceanState(
        u=u, v=v, w=w, T=T, S=S, eta=eta,
        u_prev   = u.copy(),
        v_prev   = v.copy(),
        eta_prev = eta.copy(),
        T_tend_prev  = z((Nx, Ny, Nz), dtype=jnp.float32),
        S_tend_prev  = z((Nx, Ny, Nz), dtype=jnp.float32),
        T_tend_prev2 = z((Nx, Ny, Nz), dtype=jnp.float32),
        S_tend_prev2 = z((Nx, Ny, Nz), dtype=jnp.float32),
        time       = jnp.array(time, dtype=jnp.float32),
        step_count = jnp.array(0,    dtype=jnp.int32),
    )
    return state.apply_masks(grid)


def create_rest_state(
    grid: OceanGrid,
    T_background: float = 10.0,
    S_background: float = 35.0,
) -> OceanState:
    """
    Create a resting ocean state with uniform T and S.
    Useful as a fallback when ORAS5 data are not available.
    """
    Nx, Ny, Nz = grid.Nx, grid.Ny, grid.Nz
    z = jnp.zeros
    T = jnp.full((Nx, Ny, Nz), T_background, dtype=jnp.float32) * grid.mask_c
    S = jnp.full((Nx, Ny, Nz), S_background, dtype=jnp.float32) * grid.mask_c
    return OceanState(
        u  = z((Nx, Ny, Nz),   dtype=jnp.float32),
        v  = z((Nx, Ny, Nz),   dtype=jnp.float32),
        w  = z((Nx, Ny, Nz+1), dtype=jnp.float32),
        T  = T,
        S  = S,
        eta      = z((Nx, Ny),       dtype=jnp.float32),
        u_prev   = z((Nx, Ny, Nz), dtype=jnp.float32),
        v_prev   = z((Nx, Ny, Nz), dtype=jnp.float32),
        eta_prev = z((Nx, Ny),     dtype=jnp.float32),
        T_tend_prev  = z((Nx, Ny, Nz), dtype=jnp.float32),
        S_tend_prev  = z((Nx, Ny, Nz), dtype=jnp.float32),
        T_tend_prev2 = z((Nx, Ny, Nz), dtype=jnp.float32),
        S_tend_prev2 = z((Nx, Ny, Nz), dtype=jnp.float32),
        time       = jnp.array(0.0, dtype=jnp.float32),
        step_count = jnp.array(0,   dtype=jnp.int32),
    )
