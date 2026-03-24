"""
OceanJAX Physics – Dynamics Module
====================================
Boussinesq hydrostatic primitive equations on an Arakawa C-grid.

Responsibility contract
-----------------------
Every function returns a **tendency** [m s-2 or m s-1] or a **diagnostic
field** only.  No time integration, no state mutation.

Governing equations (Boussinesq, hydrostatic)
---------------------------------------------
Momentum (explicit part only; vertical viscosity is implicit):

  du/dt = +f*v  - (1/rho0)*dp/dx  +  nu_h * lap_u(u)
  dv/dt = -f*u  - (1/rho0)*dp/dy  +  nu_h * lap_v(v)

Nonlinear momentum advection is intentionally excluded here; it requires
a separate, geometry-consistent discretisation on the staggered u/v cells
and can be added as ``momentum_advection_u/v`` without changing this interface.

Hydrostatic balance:
  dp/dz = rho * g   (positive downward)

Continuity (Boussinesq, incompressible):
  du/dx + dv/dy + dw/dz = 0

Free surface (rigid-lid approximation for w-diagnostic; eta evolves via
the barotropic tendency):
  deta/dt = -div_h(integral_0^H u dz, integral_0^H v dz)

Linear equation of state:
  rho = rho0 * (1 - alpha_T*(T - T_ref) + beta_S*(S - S_ref))

Contents
--------
  equation_of_state         – linear EOS -> rho (Nx,Ny,Nz)
  hydrostatic_pressure      – cumulative g-rho integration -> p_c (Nx,Ny,Nz)
  pressure_gradient_u       – barotropic + baroclinic PGF at u-points
  pressure_gradient_v       – barotropic + baroclinic PGF at v-points
  coriolis_u                – +f*v tendency at u-points
  coriolis_v                – -f*u tendency at v-points
  momentum_tendency_u       – full explicit RHS for u
  momentum_tendency_v       – full explicit RHS for v
  free_surface_tendency     – deta/dt from depth-integrated divergence
  compute_w                 – diagnose w from continuity (lax.scan)
"""

from __future__ import annotations

import jax
import jax.numpy as jnp

from OceanJAX.grid import OceanGrid
from OceanJAX.operators import grad_x, grad_y


# ---------------------------------------------------------------------------
# Equation of state
# ---------------------------------------------------------------------------

def equation_of_state(
    T:      jnp.ndarray,
    S:      jnp.ndarray,
    params,
) -> jnp.ndarray:
    """
    Linear Boussinesq equation of state.

      rho = rho0 * (1 - alpha_T*(T - T_ref) + beta_S*(S - S_ref))

    Warming decreases density (alpha_T > 0); salinity increases it (beta_S > 0).

    Args:
        T      : (Nx, Ny, Nz) potential temperature [degC]
        S      : (Nx, Ny, Nz) practical salinity [psu]
        params : ModelParams

    Returns:
        rho : (Nx, Ny, Nz) [kg m-3]
    """
    return params.rho0 * (
        1.0
        - params.alpha_T * (T - params.T_ref)
        + params.beta_S  * (S - params.S_ref)
    )


# ---------------------------------------------------------------------------
# Hydrostatic pressure
# ---------------------------------------------------------------------------

def hydrostatic_pressure(
    rho:    jnp.ndarray,
    grid:   OceanGrid,
    params,
) -> jnp.ndarray:
    """
    Baroclinic hydrostatic pressure at cell centres by downward integration.

    The pressure at the w-face above cell k is:
      p_w[k+1] = p_w[k] + rho[k] * g * dz_c[k],   p_w[0] = 0

    The cell-centre value is the arithmetic mean of the bounding faces:
      p_c[k]   = p_w[k] + 0.5 * rho[k] * g * dz_c[k]

    The scan iterates from the surface (k=0) downward; dry cells are
    zeroed by mask_c before integration so that the pressure below a
    seamount is not contaminated by the wet-cell values above it.

    Args:
        rho    : (Nx, Ny, Nz) [kg m-3]
        grid   : OceanGrid
        params : ModelParams  (uses g)

    Returns:
        p_c : (Nx, Ny, Nz) [Pa], zeroed at dry cells
    """
    # Pressure increment for each layer; zero dry cells
    dp = rho * grid.mask_c * params.g * grid.dz_c   # (Nx, Ny, Nz)

    def _step(p_top, dp_k):
        """
        p_top : pressure at the top w-face of the current layer, (Nx, Ny)
        dp_k  : rho*g*dz for the current layer, (Nx, Ny)
        Returns cell-centre pressure and the pressure at the bottom face.
        """
        p_center = p_top + 0.5 * dp_k
        p_bottom = p_top + dp_k
        return p_bottom, p_center

    # Transpose to (Nz, Nx, Ny) for lax.scan to iterate over k
    dp_T = jnp.moveaxis(dp, -1, 0)                          # (Nz, Nx, Ny)
    _, p_c_T = jax.lax.scan(_step, jnp.zeros((grid.Nx, grid.Ny)), dp_T)
    p_c = jnp.moveaxis(p_c_T, 0, -1)                        # (Nx, Ny, Nz)
    return p_c * grid.mask_c


# ---------------------------------------------------------------------------
# Pressure gradient force
# ---------------------------------------------------------------------------

def pressure_gradient_u(
    p_hyd: jnp.ndarray,
    eta:   jnp.ndarray,
    grid:  OceanGrid,
    params,
) -> jnp.ndarray:
    """
    Zonal pressure gradient force at u-points [m s-2].

      PGF_u = -g * grad_x(eta) - (1/rho0) * grad_x(p_hyd)

    The barotropic term (g*grad_x(eta)) is 2D and broadcast over all depths.
    Both gradients are computed by the existing grad_x operator (zeroed at
    dry u-faces via mask_u).

    Args:
        p_hyd  : (Nx, Ny, Nz) baroclinic hydrostatic pressure [Pa]
        eta    : (Nx, Ny)     sea-surface height [m]
        grid   : OceanGrid
        params : ModelParams  (uses g, rho0)

    Returns:
        (Nx, Ny, Nz) [m s-2], zeroed at dry u-faces
    """
    pg_bt = params.g * grad_x(eta, grid)                    # (Nx, Ny)
    pg_bc = grad_x(p_hyd, grid) / params.rho0               # (Nx, Ny, Nz)
    return -(pg_bt[:, :, jnp.newaxis] + pg_bc) * grid.mask_u


def pressure_gradient_v(
    p_hyd: jnp.ndarray,
    eta:   jnp.ndarray,
    grid:  OceanGrid,
    params,
) -> jnp.ndarray:
    """
    Meridional pressure gradient force at v-points [m s-2].

      PGF_v = -g * grad_y(eta) - (1/rho0) * grad_y(p_hyd)

    Args / Returns: same convention as pressure_gradient_u.
    """
    pg_bt = params.g * grad_y(eta, grid)                    # (Nx, Ny)
    pg_bc = grad_y(p_hyd, grid) / params.rho0               # (Nx, Ny, Nz)
    return -(pg_bt[:, :, jnp.newaxis] + pg_bc) * grid.mask_v


# ---------------------------------------------------------------------------
# Coriolis
# ---------------------------------------------------------------------------

def coriolis_u(
    v:    jnp.ndarray,
    grid: OceanGrid,
) -> jnp.ndarray:
    """
    Coriolis acceleration for the u-momentum equation: +f * v_at_u  [m s-2].

    f at the u-point (i+1/2, j):
      f_u[i,j] = 0.5 * (f_c[i,j] + f_c[i+1,j])
      Since f_c is independent of i (f = 2*Omega*sin(lat_c)), f_u = f_c.

    v at the u-point (i+1/2, j) — 4-point average over the surrounding
    v-face centres (i, j-1/2), (i+1, j-1/2), (i, j+1/2), (i+1, j+1/2):

      v_at_u[i,j,k] = 0.25 * (v[i,j-1] + v[i,j] + v[i+1,j-1] + v[i+1,j])

    The south-wall term v[i, j-1] is naturally zero via mask_v at j=0.
    Periodic in x.

    Args:
        v    : (Nx, Ny, Nz) meridional velocity at v-points
        grid : OceanGrid  (uses f_c, mask_u)

    Returns:
        (Nx, Ny, Nz) [m s-2], zeroed at dry u-faces
    """
    # v[i+1, j, k] — eastern neighbor (periodic in x)
    v_e = jnp.roll(v, -1, axis=0)

    # v[i, j-1, k] and v[i+1, j-1, k] — southern v-faces; zero at j=0 (wall)
    v_s   = jnp.concatenate(
        [jnp.zeros((grid.Nx, 1, grid.Nz), dtype=v.dtype), v[:, :-1, :]], axis=1
    )
    v_e_s = jnp.concatenate(
        [jnp.zeros((grid.Nx, 1, grid.Nz), dtype=v.dtype), v_e[:, :-1, :]], axis=1
    )

    v_at_u = 0.25 * (v + v_s + v_e + v_e_s)   # (Nx, Ny, Nz)

    # f_c is zonally invariant; broadcast from (Nx, Ny) to (Nx, Ny, Nz)
    return grid.f_c[:, :, jnp.newaxis] * v_at_u * grid.mask_u


def coriolis_v(
    u:    jnp.ndarray,
    grid: OceanGrid,
) -> jnp.ndarray:
    """
    Coriolis acceleration for the v-momentum equation: -f * u_at_v  [m s-2].

    f at the v-point (i, j+1/2):
      f_v[i,j] = 0.5 * (f_c[i,j] + f_c[i,j+1])

    u at the v-point (i, j+1/2) — 4-point average:

      u_at_v[i,j,k] = 0.25 * (u[i-1,j] + u[i,j] + u[i-1,j+1] + u[i,j+1])

    The northern term u[i, j+1] is zeroed via mask_u at j=Ny-1.
    Periodic in x.

    Args:
        u    : (Nx, Ny, Nz) zonal velocity at u-points
        grid : OceanGrid  (uses f_c, mask_v)

    Returns:
        (Nx, Ny, Nz) [m s-2], zeroed at dry v-faces
    """
    # u[i-1, j, k] — western neighbor (periodic in x)
    u_w = jnp.roll(u, 1, axis=0)

    # u[i, j+1, k] and u[i-1, j+1, k] — northern u-faces; zero at j=Ny-1
    u_n   = jnp.roll(u, -1, axis=1)
    u_n   = u_n.at[:, -1, :].set(0.0)
    u_w_n = jnp.roll(u_n, 1, axis=0)

    u_at_v = 0.25 * (u + u_w + u_n + u_w_n)   # (Nx, Ny, Nz)

    # f at v-point: average in j; north Neumann (copy last value)
    f_n = jnp.roll(grid.f_c, -1, axis=1)
    f_n = f_n.at[:, -1].set(grid.f_c[:, -1])
    f_v = 0.5 * (grid.f_c + f_n)               # (Nx, Ny)

    return -f_v[:, :, jnp.newaxis] * u_at_v * grid.mask_v


# ---------------------------------------------------------------------------
# Explicit momentum tendencies
# ---------------------------------------------------------------------------

def momentum_tendency_u(
    state,
    grid:   OceanGrid,
    params,
) -> jnp.ndarray:
    """
    Full explicit tendency for zonal momentum u [m s-2].

      du/dt|_explicit = Coriolis + PGF + horizontal viscosity

    Nonlinear momentum advection (u·∇u) is excluded; it requires a
    separate energy-consistent discretisation on the staggered u-cell
    geometry and can be added as ``momentum_advection_u`` without
    changing this interface.

    Vertical viscosity is treated implicitly by the time stepper
    (OceanJAX.Physics.mixing.implicit_vertical_visc); do not add it here
    to avoid double-counting.

    Surface wind stress forcing should be added by the time stepper
    separately, analogous to tracer surface forcing.

    Args:
        state  : OceanState  (uses u, v, T, S, eta)
        grid   : OceanGrid
        params : ModelParams

    Returns:
        du/dt|_explicit : (Nx, Ny, Nz) [m s-2]
    """
    from OceanJAX.Physics.mixing import horizontal_viscosity   # deferred

    rho   = equation_of_state(state.T, state.S, params)
    p_hyd = hydrostatic_pressure(rho, grid, params)

    cor  = coriolis_u(state.v, grid)
    pgf  = pressure_gradient_u(p_hyd, state.eta, grid, params)
    visc_u, _ = horizontal_viscosity(state.u, state.v, params.nu_h, grid)

    return (cor + pgf + visc_u) * grid.mask_u


def momentum_tendency_v(
    state,
    grid:   OceanGrid,
    params,
) -> jnp.ndarray:
    """
    Full explicit tendency for meridional momentum v [m s-2].

      dv/dt|_explicit = Coriolis + PGF + horizontal viscosity

    Same exclusions as ``momentum_tendency_u``.

    Args:
        state  : OceanState  (uses u, v, T, S, eta)
        grid   : OceanGrid
        params : ModelParams

    Returns:
        dv/dt|_explicit : (Nx, Ny, Nz) [m s-2]
    """
    from OceanJAX.Physics.mixing import horizontal_viscosity   # deferred

    rho   = equation_of_state(state.T, state.S, params)
    p_hyd = hydrostatic_pressure(rho, grid, params)

    cor  = coriolis_v(state.u, grid)
    pgf  = pressure_gradient_v(p_hyd, state.eta, grid, params)
    _, visc_v = horizontal_viscosity(state.u, state.v, params.nu_h, grid)

    return (cor + pgf + visc_v) * grid.mask_v


# ---------------------------------------------------------------------------
# Free-surface tendency
# ---------------------------------------------------------------------------

def free_surface_tendency(
    u:    jnp.ndarray,
    v:    jnp.ndarray,
    grid: OceanGrid,
) -> jnp.ndarray:
    """
    Sea-surface height tendency from the depth-integrated continuity equation.

      deta/dt = -div_h(U_col, V_col)

    where U_col and V_col are the depth-integrated transports [m2 s-1]:
      U_col[i,j] = sum_k  u[i,j,k] * dz_c[k]
      V_col[i,j] = sum_k  v[i,j,k] * dz_c[k]

    The 2-D flux-form divergence mirrors div_h in operators.py: east-face
    width dy_c, north-face width dx_v (cos(lat_v) metric), zero south flux
    at j=0, output gated by the surface mask_c[:,:,0].

    Args:
        u    : (Nx, Ny, Nz) zonal velocity at u-points [m s-1]
        v    : (Nx, Ny, Nz) meridional velocity at v-points [m s-1]
        grid : OceanGrid

    Returns:
        deta/dt : (Nx, Ny) [m s-1], zeroed at dry surface cells
    """
    # Depth-integrated transports (mask already embedded in u, v)
    U_col = jnp.sum(u * grid.mask_u * grid.dz_c, axis=-1)   # (Nx, Ny)
    V_col = jnp.sum(v * grid.mask_v * grid.dz_c, axis=-1)   # (Nx, Ny)

    # 2-D flux-form divergence
    Fu   = U_col * grid.dy_c                                  # east-face flux
    dFu  = Fu - jnp.roll(Fu, 1, axis=0)

    Fv   = V_col * grid.dx_v                                  # north-face flux
    Fv_s = jnp.concatenate(
        [jnp.zeros((grid.Nx, 1), dtype=u.dtype), Fv[:, :-1]], axis=1
    )
    dFv  = Fv - Fv_s

    return -((dFu + dFv) / grid.area_c) * grid.mask_c[:, :, 0]


# ---------------------------------------------------------------------------
# Continuity: diagnose w
# ---------------------------------------------------------------------------

def compute_w(
    u:    jnp.ndarray,
    v:    jnp.ndarray,
    grid: OceanGrid,
) -> jnp.ndarray:
    """
    Diagnose vertical velocity w from the incompressibility constraint.

    Integrates the continuity equation downward from the surface:

      w[k+1] = w[k] - div_h(u, v)[k] * dz_c[k]

    with the surface boundary condition w[0] = 0 (rigid-lid kinematic
    BC for the advection solver; see mask_w_adv).  For a free-surface
    model the correct w[0] = deta/dt is set by the time stepper after
    this function returns, but mask_w_adv zeroes the surface face for
    tracer advection regardless.

    The horizontal divergence is computed in flux form using the same
    area metrics as div_h in operators.py, ensuring consistency with
    the tracer advection discretisation.

    Args:
        u    : (Nx, Ny, Nz)  zonal velocity at u-points [m s-1]
        v    : (Nx, Ny, Nz)  meridional velocity at v-points [m s-1]
        grid : OceanGrid

    Returns:
        w : (Nx, Ny, Nz+1) [m s-1], gated by mask_w
    """
    # Horizontal divergence at cell centres: (Nx, Ny, Nz)
    Fu    = u * grid.mask_u * grid.dy_c[:, :, jnp.newaxis]
    dFu   = Fu - jnp.roll(Fu, 1, axis=0)

    Fv    = v * grid.mask_v * grid.dx_v[:, :, jnp.newaxis]
    Fv_s  = jnp.concatenate(
        [jnp.zeros((grid.Nx, 1, grid.Nz), dtype=u.dtype), Fv[:, :-1, :]], axis=1
    )
    dFv   = Fv - Fv_s

    div_uv = (dFu + dFv) / grid.area_c[:, :, jnp.newaxis]   # (Nx, Ny, Nz)

    # Downward increment: dw[k] = -div_uv[k] * dz_c[k]
    dw = -div_uv * grid.dz_c                                  # (Nx, Ny, Nz)

    def _step(w_k, dw_k):
        """
        w_k  : w at the top face of the current layer, (Nx, Ny)
        dw_k : w increment for this layer (= -div_uv*dz), (Nx, Ny)
        Returns w at the bottom face (= top of next layer), and saves w_k.
        """
        w_next = w_k + dw_k
        return w_next, w_k

    # lax.scan over k; initial carry = w[0] = 0 (rigid-lid surface BC)
    dw_T = jnp.moveaxis(dw, -1, 0)                           # (Nz, Nx, Ny)
    w_bottom, w_tops_T = jax.lax.scan(
        _step, jnp.zeros((grid.Nx, grid.Ny), dtype=u.dtype), dw_T
    )
    # w_tops_T : (Nz, Nx, Ny) = [w[0], w[1], ..., w[Nz-1]]
    # w_bottom : (Nx, Ny)     = w[Nz] (should be ~0 for consistent u,v)

    w_tops = jnp.moveaxis(w_tops_T, 0, -1)                   # (Nx, Ny, Nz)
    w = jnp.concatenate([w_tops, w_bottom[:, :, jnp.newaxis]], axis=-1)  # (Nx,Ny,Nz+1)
    return w * grid.mask_w
