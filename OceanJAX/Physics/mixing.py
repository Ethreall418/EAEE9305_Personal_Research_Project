"""
OceanJAX Physics – Mixing Module
==================================
Vertical and horizontal mixing parameterisations.

Responsibility contract
-----------------------
Every function returns either a **tendency** [tracer s⁻¹ or m s⁻²] or a
**diffusivity field** [m² s⁻¹].  No time integration is performed here.

Vertical diffusion is treated **implicitly**: the time stepper calls
``implicit_vertical_mix`` which solves the tridiagonal system

  (I - dt * L_v) phi^{n+1} = phi^n + dt * explicit_tend

where L_v is the vertical diffusion operator, and returns phi^{n+1}
directly.  This avoids the severe stability constraint that would arise
from an explicit vertical  diffusion step.

Horizontal viscosity and diffusion are treated **explicitly** and return
tendencies that the time stepper adds before advancing.

Contents
--------
thomas_algorithm          – differentiable tridiagonal solver via lax.scan
implicit_vertical_mix     – implicit vertical diffusion for tracers
implicit_vertical_visc    – implicit vertical diffusion for velocities (u or v)
                            uses velocity-consistent vertical face masks
_laplacian_u / _v         – scalar Laplacian at u- / v-points with correct metrics
horizontal_viscosity      – Laplacian viscosity tendency for (u, v)
richardson_number         – raw (unclipped) gradient Richardson number
ri_based_diffusivity      – Richardson-number shear/convection diffusivity
"""

from __future__ import annotations

import jax
import jax.numpy as jnp

from OceanJAX.grid import OceanGrid


# ---------------------------------------------------------------------------
# Module-level utility
# ---------------------------------------------------------------------------

def _diff_w(phi: jnp.ndarray) -> jnp.ndarray:
    """
    Centred vertical difference of phi at w-faces, shape (Nx, Ny, Nz+1).
    Interior faces k=1..Nz-1 get phi[k] - phi[k-1]; boundary faces are zero.
    """
    interior = phi[..., 1:] - phi[..., :-1]
    zeros    = jnp.zeros(phi.shape[:2] + (1,), dtype=phi.dtype)
    return jnp.concatenate([zeros, interior, zeros], axis=-1)


# ---------------------------------------------------------------------------
# Tridiagonal solver (Thomas algorithm) – differentiable via lax.scan
# ---------------------------------------------------------------------------

def thomas_algorithm(
    a: jnp.ndarray,
    b: jnp.ndarray,
    c: jnp.ndarray,
    d: jnp.ndarray,
) -> jnp.ndarray:
    """
    Solve the tridiagonal system  A x = d  using the Thomas algorithm.

    The system has the form:

      b[0] x[0] + c[0] x[1]                          = d[0]
      a[k] x[k-1] + b[k] x[k] + c[k] x[k+1]         = d[k]   1 ≤ k ≤ N-2
                    a[N-1] x[N-2] + b[N-1] x[N-1]    = d[N-1]

    Args:
        a : (N,) lower diagonal  (a[0] is unused)
        b : (N,) main  diagonal
        c : (N,) upper diagonal  (c[N-1] is unused)
        d : (N,) right-hand side

    Returns:
        x : (N,) solution

    Implementation uses ``jax.lax.scan`` so the solver is fully
    differentiable via reverse-mode AD and JIT-compilable.
    No Python loops over array values.
    """
    N = b.shape[0]

    # ---- Forward sweep: eliminate lower diagonal ---------------------------
    def fwd_step(carry, k):
        b_prev, d_prev = carry          # modified b and d from previous row
        w   = a[k] / b_prev            # elimination factor
        b_k = b[k] - w * c[k - 1]     # modified main diagonal
        d_k = d[k] - w * d_prev       # modified RHS
        return (b_k, d_k), (b_k, d_k)

    # Initialise with row 0
    init    = (b[0], d[0])
    # Scan over rows 1..N-1
    _, (b_mod, d_mod) = jax.lax.scan(fwd_step, init, jnp.arange(1, N))

    # Concatenate row 0 back
    b_all = jnp.concatenate([b[:1], b_mod])   # (N,)
    d_all = jnp.concatenate([d[:1], d_mod])   # (N,)

    # ---- Back substitution -------------------------------------------------
    def bwd_step(x_next, k):
        x_k = (d_all[k] - c[k] * x_next) / b_all[k]
        return x_k, x_k

    # Initialise with last row
    x_last = d_all[-1] / b_all[-1]
    _, x_interior = jax.lax.scan(
        bwd_step, x_last, jnp.arange(N - 2, -1, -1)
    )
    # x_interior is reversed (N-1 entries); append x_last and flip
    x = jnp.concatenate([x_interior[::-1], jnp.array([x_last])])
    return x


def _build_tridiag_implicit(kappa: jnp.ndarray, dz_c: jnp.ndarray,
                             dz_w: jnp.ndarray, dt: float,
                             mask_w_col: jnp.ndarray) -> tuple:
    """
    Build the tridiagonal coefficients for implicit vertical diffusion
    of a single water column.

    Discretisation of  -d/dz(kappa * dC/dz) at cell centres:

      flux_k   = kappa[k] / dz_w[k]    (flux coefficient at w-face k)

    The implicit system for column update C^{n+1}:

      C^{n+1}[k] - dt * (flux_{k+1} * C^{n+1}[k+1]
                        - (flux_k + flux_{k+1}) * C^{n+1}[k]
                        + flux_k * C^{n+1}[k-1]) / dz_c[k]
      = C^n[k]

    Boundary conditions (via mask_w_col):
      surface face k=0  : flux = 0 (Neumann, surface forcing handled separately)
      bottom  face k=Nz : flux = 0 (Neumann, no-flux seafloor)

    Args:
        kappa      : (Nz+1,) diffusivity at w-faces [m² s⁻¹]
        dz_c       : (Nz,)   cell thicknesses [m]
        dz_w       : (Nz+1,) distances between cell centres [m]
        dt         : timestep [s]
        mask_w_col : (Nz+1,) w-face mask for this column

    Returns:
        (a, b, c) tridiagonal coefficients, each (Nz,)
    """
    Nz = dz_c.shape[0]

    # Flux coefficients at each w-face [s⁻¹ equivalent]
    # Safe division: dz_w > 0 everywhere except possibly boundary
    safe_dz_w = jnp.where(dz_w > 0, dz_w, 1.0)
    flux = kappa * mask_w_col / safe_dz_w          # (Nz+1,)

    flux_top = flux[:Nz]    # face k   (top of cell k)
    flux_bot = flux[1:]     # face k+1 (bottom of cell k)

    # Lower diagonal: a[k] = -dt * flux_top[k] / dz_c[k]   (k=1..Nz-1)
    a = -dt * flux_top / dz_c                      # (Nz,)  a[0] unused
    # Upper diagonal: c[k] = -dt * flux_bot[k] / dz_c[k]   (k=0..Nz-2)
    c = -dt * flux_bot / dz_c                      # (Nz,)  c[Nz-1] unused
    # Main diagonal: b[k] = 1 - a[k] - c[k]
    b = 1.0 - a - c                                # (Nz,)

    return a, b, c


# ---------------------------------------------------------------------------
# Implicit vertical diffusion for tracers
# ---------------------------------------------------------------------------

def implicit_vertical_mix(
    phi:   jnp.ndarray,
    kappa: jnp.ndarray,
    dt:    float,
    grid:  OceanGrid,
    rhs_explicit: jnp.ndarray | None = None,
) -> jnp.ndarray:
    """
    Implicitly mix a tracer field in the vertical direction.

    Solves column-by-column:

      (I - dt * L_v) phi^{n+1} = phi^n + dt * rhs_explicit

    where L_v is the vertical diffusion operator.  A separate call to
    ``thomas_algorithm`` is made for each (i, j) column via ``jax.vmap``.

    Args:
        phi          : (Nx, Ny, Nz)    tracer field at time n
        kappa        : (Nx, Ny, Nz+1)  vertical diffusivity at w-faces [m² s⁻¹]
        dt           : timestep [s]
        grid         : OceanGrid
        rhs_explicit : (Nx, Ny, Nz) or None
                       Explicit tendency already accumulated for this timestep.
                       If None, treated as zero.

    Returns:
        phi^{n+1} : (Nx, Ny, Nz), masked to zero on dry cells
    """
    if rhs_explicit is None:
        rhs = phi
    else:
        rhs = phi + dt * rhs_explicit

    # Accept scalar kappa (e.g. params.kappa_v) and broadcast to (Nx, Ny, Nz+1)
    Nx, Ny, Nz = phi.shape
    if jnp.ndim(kappa) == 0:
        kappa = jnp.full((Nx, Ny, Nz + 1), kappa)

    def solve_column(phi_col, kappa_col, rhs_col, mask_w_col, mask_c_col):
        """Solve one (i,j) column. All inputs are 1-D in z."""
        a, b, c = _build_tridiag_implicit(
            kappa_col, grid.dz_c, grid.dz_w, dt, mask_w_col
        )
        # For dry cells, the system degenerates; keep phi = 0 there.
        # The mask on the diagonal (b=1 for dry cells, a=c=0) achieves this
        # naturally since rhs = 0 for dry cells.
        phi_new = thomas_algorithm(a, b, c, rhs_col * mask_c_col)
        return phi_new * mask_c_col

    # vmap over (i, j) simultaneously by flattening the horizontal dims
    phi_2d    = phi.reshape(Nx * Ny, Nz)
    kappa_2d  = kappa.reshape(Nx * Ny, Nz + 1)
    rhs_2d    = rhs.reshape(Nx * Ny, Nz)
    # Use mask_w_adv (surface face k=0 always closed) so that the implicit
    # diffusion operator has no flux through the sea surface.  The surface
    # tracer exchange is handled exclusively by the explicit forcing tendencies
    # in tracers.py, exactly as for tracer advection.  Using mask_w here would
    # open a spurious diffusive flux at k=0 (a[0] ≠ 0 in the tridiagonal
    # system), corrupting the top-layer temperature even in a resting ocean.
    mask_w_2d = grid.mask_w_adv.reshape(Nx * Ny, Nz + 1)
    mask_c_2d = grid.mask_c.reshape(Nx * Ny, Nz)

    phi_new_2d = jax.vmap(solve_column)(
        phi_2d, kappa_2d, rhs_2d, mask_w_2d, mask_c_2d
    )
    return phi_new_2d.reshape(Nx, Ny, Nz)


# ---------------------------------------------------------------------------
# Implicit vertical diffusion for velocity (u or v)
# ---------------------------------------------------------------------------

def implicit_vertical_visc(
    vel:   jnp.ndarray,
    nu_v:  jnp.ndarray,
    dt:    float,
    grid:  OceanGrid,
    mask:  jnp.ndarray,
) -> jnp.ndarray:
    """
    Implicitly mix a horizontal velocity component in the vertical direction.

    Uses a velocity-consistent vertical face mask: internal face k is active
    only when both ``mask[..., k-1]`` and ``mask[..., k]`` are 1, matching the
    actual grid locations of u or v rather than the tracer mask_w.  Using the
    tracer mask_w would incorrectly couple layers at seamount edges where a
    u- or v-column is entirely dry despite adjacent tracer columns being wet.

    Args:
        vel   : (Nx, Ny, Nz)    velocity component at time n
        nu_v  : (Nx, Ny, Nz+1) vertical viscosity at w-faces [m² s⁻¹]
        dt    : timestep [s]
        grid  : OceanGrid
        mask  : (Nx, Ny, Nz)   wet mask for this velocity component
                                (mask_u for u, mask_v for v)

    Returns:
        vel^{n+1} : (Nx, Ny, Nz)
    """
    Nx, Ny, Nz = vel.shape

    # Accept scalar nu_v (e.g. params.nu_v) and broadcast to (Nx, Ny, Nz+1)
    if jnp.ndim(nu_v) == 0:
        nu_v = jnp.full((Nx, Ny, Nz + 1), nu_v)

    # Build the vertical face mask consistent with this velocity component.
    # Face k is open only when both adjacent velocity layers are wet.
    # Surface (k=0) and bottom (k=Nz) are always closed (no-flux BC).
    mask_w_vel = jnp.zeros((Nx, Ny, Nz + 1), dtype=mask.dtype)
    mask_w_vel = mask_w_vel.at[:, :, 1:Nz].set(mask[:, :, :-1] * mask[:, :, 1:])

    def solve_column(vel_col, nu_col, mask_w_col, mask_col):
        a, b, c = _build_tridiag_implicit(
            nu_col, grid.dz_c, grid.dz_w, dt, mask_w_col
        )
        return thomas_algorithm(a, b, c, vel_col * mask_col) * mask_col

    vel_2d    = vel.reshape(Nx * Ny, Nz)
    nu_2d     = nu_v.reshape(Nx * Ny, Nz + 1)
    mask_w_2d = mask_w_vel.reshape(Nx * Ny, Nz + 1)
    mask_2d   = mask.reshape(Nx * Ny, Nz)

    vel_new_2d = jax.vmap(solve_column)(vel_2d, nu_2d, mask_w_2d, mask_2d)
    return vel_new_2d.reshape(Nx, Ny, Nz)


# ---------------------------------------------------------------------------
# Horizontal viscosity (explicit Laplacian at velocity points)
# ---------------------------------------------------------------------------

def _laplacian_u(
    u:    jnp.ndarray,
    nu_h: float | jnp.ndarray,
    grid: OceanGrid,
) -> jnp.ndarray:
    """
    Scalar horizontal Laplacian of u at u-points (east faces).

    Uses the geometry of the u-point grid cell:

      x-direction
        Neighbours : u[i-1] and u[i+1]  (adjacent east faces)
        Spacing    : dx_c[i]   (west gap) and dx_c[i+1] (east gap)
        Face height: dy_c[j]
        Cell area  : 0.5 * (dx_c[i] + dx_c[i+1]) * dy_c[j]

      y-direction
        Neighbours : u[i,j-1] and u[i,j+1]  (same east face, adjacent rows)
        Spacing    : dy_v[j]  (= dy_c[j], distance between tracer-centre rows)
        Face width : dx_v[j]  (zonal width at north-face latitude lat_v[j])

    Face masks gate fluxes through faces that adjoin a dry u-point.
    Output is zeroed at dry u-points via mask_u.

    This is a scalar approximation; the full vector Laplacian on a spherical
    C-grid also includes off-diagonal stress terms that are neglected here.
    """
    Nx, Ny, Nz = u.shape

    # ---- x-direction -------------------------------------------------------
    u_e      = jnp.roll(u, -1, axis=0)                            # u[i+1]
    # Distance from u[i] to u[i+1] = width of tracer cell i+1
    dist_e   = jnp.roll(grid.dx_c, -1, axis=0)[:, :, jnp.newaxis]
    # Gate: both u[i] and u[i+1] must be active (periodic in x)
    mu_ee    = grid.mask_u * jnp.roll(grid.mask_u, -1, axis=0)
    fx_e     = nu_h * grid.dy_c[:, :, jnp.newaxis] * (u_e - u) / dist_e * mu_ee
    fx_w     = jnp.roll(fx_e, 1, axis=0)

    # ---- y-direction -------------------------------------------------------
    u_n      = jnp.roll(u, -1, axis=1)                            # u[i,j+1]
    # Gate: both u-rows active; north wall explicitly closed
    mu_nn    = (grid.mask_u * jnp.roll(grid.mask_u, -1, axis=1)
                ).at[:, -1, :].set(0.0)
    fy_n     = nu_h * grid.dx_v[:, :, jnp.newaxis] * (u_n - u) / grid.dy_v[:, :, jnp.newaxis] * mu_nn
    fy_s     = jnp.concatenate(
        [jnp.zeros((Nx, 1, Nz), dtype=u.dtype), fy_n[:, :-1, :]], axis=1
    )

    # ---- u-cell area -------------------------------------------------------
    area_u = 0.5 * (grid.dx_c + jnp.roll(grid.dx_c, -1, axis=0)) * grid.dy_c

    return ((fx_e - fx_w + fy_n - fy_s) / area_u[:, :, jnp.newaxis]) * grid.mask_u


def _laplacian_v(
    v:    jnp.ndarray,
    nu_h: float | jnp.ndarray,
    grid: OceanGrid,
) -> jnp.ndarray:
    """
    Scalar horizontal Laplacian of v at v-points (north faces).

    Uses the geometry of the v-point grid cell:

      x-direction
        Neighbours : v[i-1,j] and v[i+1,j]  (adjacent north faces)
        Spacing    : 0.5*(dx_v[i] + dx_v[i+1])  (distance between v-columns)
        Face height: dy_v[j]
        Cell area  : dx_v[j] * dy_v[j]

      y-direction
        Neighbours : v[i,j-1] and v[i,j+1]  (same north face, adjacent rows)
        v[j] at lat_v[j]; v[j+1] at lat_v[j+1]
        Spacing    : dy_c[j+1]  (height of tracer cell j+1)
        Face width : dx_c[j+1]  (zonal width at lat_c[j+1], midpoint of v-gap)

    Face masks gate fluxes through faces that adjoin a dry v-point.
    Output is zeroed at dry v-points via mask_v.
    """
    Nx, Ny, Nz = v.shape

    # ---- x-direction -------------------------------------------------------
    v_e      = jnp.roll(v, -1, axis=0)
    # Distance between v[i] and v[i+1] ≈ 0.5*(dx_v[i] + dx_v[i+1])
    dist_e   = 0.5 * (grid.dx_v + jnp.roll(grid.dx_v, -1, axis=0))[:, :, jnp.newaxis]
    mv_ee    = grid.mask_v * jnp.roll(grid.mask_v, -1, axis=0)
    fx_e     = nu_h * grid.dy_v[:, :, jnp.newaxis] * (v_e - v) / dist_e * mv_ee
    fx_w     = jnp.roll(fx_e, 1, axis=0)

    # ---- y-direction -------------------------------------------------------
    v_n      = jnp.roll(v, -1, axis=1)                            # v[i,j+1]
    # Distance from v[j] (lat_v[j]) to v[j+1] (lat_v[j+1]) = dy_c[j+1]
    dist_n   = jnp.roll(grid.dy_c, -1, axis=1)[:, :, jnp.newaxis]
    # Face width at midpoint between v-rows ≈ dx_c[j+1]
    width_n  = jnp.roll(grid.dx_c, -1, axis=1)[:, :, jnp.newaxis]
    mv_nn    = (grid.mask_v * jnp.roll(grid.mask_v, -1, axis=1)
                ).at[:, -1, :].set(0.0)
    fy_n     = nu_h * width_n * (v_n - v) / dist_n * mv_nn
    fy_s     = jnp.concatenate(
        [jnp.zeros((Nx, 1, Nz), dtype=v.dtype), fy_n[:, :-1, :]], axis=1
    )

    # ---- v-cell area -------------------------------------------------------
    area_v = grid.dx_v * grid.dy_v   # dx_v[j] * dy_c[j]

    return ((fx_e - fx_w + fy_n - fy_s) / area_v[:, :, jnp.newaxis]) * grid.mask_v


def horizontal_viscosity(
    u:    jnp.ndarray,
    v:    jnp.ndarray,
    nu_h: float | jnp.ndarray,
    grid: OceanGrid,
) -> tuple[jnp.ndarray, jnp.ndarray]:
    """
    Explicit horizontal Laplacian viscosity tendencies for (u, v).

    Delegates to ``_laplacian_u`` and ``_laplacian_v``, which use the correct
    grid-cell geometry and masks for each velocity point rather than the
    tracer-cell metrics used by ``kappa_laplacian_h``.

    This is a scalar Laplacian approximation.  The full vector Laplacian on
    a spherical C-grid includes off-diagonal stress terms that are omitted
    here; they are second-order corrections relevant mainly at very high
    resolution or when modelling viscous boundary layers.

    Args:
        u    : (Nx, Ny, Nz) zonal velocity at east faces
        v    : (Nx, Ny, Nz) meridional velocity at north faces
        nu_h : scalar or (Nx, Ny, Nz) horizontal viscosity [m² s⁻¹]
        grid : OceanGrid

    Returns:
        (du_dt_visc, dv_dt_visc) : each (Nx, Ny, Nz)
    """
    return _laplacian_u(u, nu_h, grid), _laplacian_v(v, nu_h, grid)


# ---------------------------------------------------------------------------
# Richardson-number-based background diffusivity
# ---------------------------------------------------------------------------

def richardson_number(
    T:    jnp.ndarray,
    S:    jnp.ndarray,
    u:    jnp.ndarray,
    v:    jnp.ndarray,
    grid: OceanGrid,
    params,
) -> jnp.ndarray:
    """
    Raw gradient Richardson number at w-faces (Nx, Ny, Nz+1).

      Ri = N² / S²

    where N² is the squared buoyancy frequency:

      N² = -(g / rho0) * drho/dz   (positive = stably stratified, negative = unstable)

    and S² is the velocity shear squared:

      S² = (du/dz)² + (dv/dz)²

    Both are evaluated at w-faces using centred differences.
    A small floor on S² prevents division by zero, but the returned Ri is
    **not clipped**: negative values indicate static instability and must
    be preserved for diagnostic use.  Closures that need a bounded Ri
    (e.g. ``ri_based_diffusivity``) apply their own clamping internally.

    Args:
        T, S   : (Nx, Ny, Nz) temperature and salinity
        u, v   : (Nx, Ny, Nz) horizontal velocities
        grid   : OceanGrid
        params : ModelParams

    Returns:
        Ri : (Nx, Ny, Nz+1), raw (possibly negative), zeroed at dry w-faces
    """
    from OceanJAX.Physics.dynamics import equation_of_state  # deferred

    rho       = equation_of_state(T, S, params)   # (Nx, Ny, Nz)
    safe_dz_w = jnp.where(grid.dz_w > 0, grid.dz_w, 1.0)

    drho_dz = _diff_w(rho) / safe_dz_w
    du_dz   = _diff_w(u)   / safe_dz_w
    dv_dz   = _diff_w(v)   / safe_dz_w

    n2 = -(params.g / params.rho0) * drho_dz
    s2 = du_dz ** 2 + dv_dz ** 2

    # Return raw Ri; no clipping — negative Ri signals static instability
    ri = n2 / jnp.where(s2 > 1e-10, s2, 1e-10)
    return ri * grid.mask_w


# ---------------------------------------------------------------------------
# Simplified KPP vertical diffusivity
# ---------------------------------------------------------------------------

def ri_based_diffusivity(
    T:     jnp.ndarray,
    S:     jnp.ndarray,
    u:     jnp.ndarray,
    v:     jnp.ndarray,
    grid:  OceanGrid,
    params,
    kappa_0:    float = 1e-5,
    kappa_conv: float = 1e-1,
    ri_crit:    float = 0.7,
) -> jnp.ndarray:
    """
    Richardson-number-based vertical diffusivity for tracers.

    Enhances a background diffusivity with shear-driven mixing when the
    gradient Richardson number falls below a critical value, and applies
    a convective-adjustment diffusivity in statically unstable layers:

      kappa(k) = kappa_0
               + kappa_conv * (1 - Ri / Ri_crit)²   if 0 <= Ri < Ri_crit
               + kappa_conv                           if N² < 0  (convective)

    This is **not** a full KPP scheme (Large et al. 1994): it omits the
    boundary-layer depth, counter-gradient fluxes, and nonlocal transport.
    It provides a physically motivated background mixing suitable for
    multi-year integrations and can be swapped for an ML closure
    (``OceanJAX.ml.closure``) without changing the calling interface.

    The raw Richardson number is computed internally.  Clamping to
    [0, Ri_crit] is applied here, not in ``richardson_number``, so that
    the diagnostic function preserves the full signed Ri signal.

    Args:
        T, S        : (Nx, Ny, Nz)
        u, v        : (Nx, Ny, Nz)
        grid        : OceanGrid
        params      : ModelParams  (uses g, rho0)
        kappa_0     : background diffusivity [m² s⁻¹]  (default 1e-5)
        kappa_conv  : shear / convective diffusivity [m² s⁻¹]  (default 0.1)
        ri_crit     : critical Richardson number for shear mixing (default 0.7)

    Returns:
        kappa : (Nx, Ny, Nz+1) [m² s⁻¹], zeroed at dry w-faces
    """
    # Reuse the diagnostic Ri, then clamp internally for the closure
    ri = richardson_number(T, S, u, v, grid, params)   # raw, possibly negative

    # N² is needed separately to detect convective instability
    from OceanJAX.Physics.dynamics import equation_of_state  # deferred
    rho = equation_of_state(T, S, params)

    safe_dz_w = jnp.where(grid.dz_w > 0, grid.dz_w, 1.0)
    n2 = -(params.g / params.rho0) * _diff_w(rho) / safe_dz_w

    # Shear enhancement: (1 - Ri/Ri_crit)² for 0 <= Ri < Ri_crit
    ri_clamped    = jnp.clip(ri, 0.0, ri_crit)
    shear_factor  = jnp.where(
        (ri >= 0.0) & (ri < ri_crit),
        (1.0 - ri_clamped / ri_crit) ** 2,
        0.0,
    )

    # Convective adjustment: N² < 0
    conv_factor = jnp.where(n2 < 0.0, 1.0, 0.0)

    kappa = kappa_0 + kappa_conv * (shear_factor + conv_factor)
    return kappa * grid.mask_w
