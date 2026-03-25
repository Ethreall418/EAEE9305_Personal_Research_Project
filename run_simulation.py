"""
run_simulation.py  v2
=====================
Entry-point script for OceanJAX simulations.

Usage
-----
    python run_simulation.py [options]

The script:
  1. Builds an OceanGrid over a user-specified domain.
  2. Initialises the ocean state:
       ``--init rest``   — uniform T/S, zero velocity  (default).
       ``--init oras5``  — interpolated from a Copernicus Marine ORAS5
                           NetCDF file (requires ``--oras5_path``).
  3. Applies optional constant surface forcing.
  4. Integrates forward using chunked jax.lax.scan calls.
  5. Writes T, S, eta snapshots to a NetCDF file whenever the step counter
     crosses a save-interval boundary (independent of chunk_size).
  6. Prints per-chunk diagnostics (step, time, T/S/eta range, non-finite flag, wall-clock).

Chunking vs. saving
-------------------
``chunk_size`` controls how many steps are compiled into one ``jax.lax.scan``
trace.  ``save_interval`` controls how often output is written.  Neither needs
to be a multiple of the other.  A partial final chunk runs correctly when
``n_steps`` is not a multiple of ``chunk_size``.

Output layout
-------------
NetCDF variables follow the internal array order: (time, x, y, z) for 3-D
fields and (time, x, y) for eta.  No transposition is performed.
"""

from __future__ import annotations

import argparse
import sys
import time as _time

import jax
import jax.numpy as jnp
import netCDF4 as nc
import numpy as np

from OceanJAX.grid import OceanGrid
from OceanJAX.state import ModelParams, OceanState, create_rest_state
from OceanJAX.timeStepping import SurfaceForcing
from OceanJAX.timeStepping import run as ocean_run
# load_oras5 is imported lazily inside _build_initial_state to keep startup fast
# when --init rest is used.


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _parse_args(argv=None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Run an idealised OceanJAX simulation.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # Grid geometry
    g = p.add_argument_group("grid")
    g.add_argument("--nx",        type=int,   default=10,    help="Cells in x (longitude)")
    g.add_argument("--ny",        type=int,   default=10,    help="Cells in y (latitude)")
    g.add_argument("--nz",        type=int,   default=5,     help="Cells in z (depth)")
    g.add_argument("--lon_min",   type=float, default=0.0,   help="West boundary [deg east]")
    g.add_argument("--lon_max",   type=float, default=10.0,  help="East boundary [deg east]")
    g.add_argument("--lat_min",   type=float, default=10.0,  help="South boundary [deg north]")
    g.add_argument("--lat_max",   type=float, default=20.0,  help="North boundary [deg north]")
    g.add_argument("--depth_max", type=float, default=500.0, help="Ocean depth [m]")

    # Initialisation
    ini = p.add_argument_group("initialisation")
    ini.add_argument("--init", choices=["rest", "oras5"], default="rest",
                     help="Initial condition source")
    ini.add_argument("--oras5_path", type=str, default=None,
                     help="Path to ORAS5 NetCDF file (required when --init oras5)")
    ini.add_argument("--oras5_time_index", type=int, default=0,
                     help="Time index within the ORAS5 file")
    ini.add_argument("--oras5_T_fill", type=float, default=10.0,
                     help="T fallback for uncovered points [degC]")
    ini.add_argument("--oras5_S_fill", type=float, default=35.0,
                     help="S fallback for uncovered points [psu]")

    # Physics
    ph = p.add_argument_group("physics")
    ph.add_argument("--dt",    type=float, default=900.0, help="Time step [s]")
    ph.add_argument("--T_bg",  type=float, default=10.0,
                    help="Initial uniform temperature [degC]  (--init rest only)")
    ph.add_argument("--S_bg",  type=float, default=35.0,
                    help="Initial uniform salinity [psu]  (--init rest only)")

    # Surface forcing (constant in time and space)
    f = p.add_argument_group("surface forcing")
    f.add_argument("--heat_flux", type=float, default=0.0,
                   help="Constant net downward heat flux [W m-2]")
    f.add_argument("--fw_flux",   type=float, default=0.0,
                   help="Constant E-P freshwater flux [m s-1]")
    f.add_argument("--tau_x",     type=float, default=0.0,
                   help="Constant zonal wind stress [N m-2]")
    f.add_argument("--tau_y",     type=float, default=0.0,
                   help="Constant meridional wind stress [N m-2]")

    # Run control
    r = p.add_argument_group("run control")
    r.add_argument("--n_steps",       type=int, default=100,
                   help="Total number of time steps")
    r.add_argument("--chunk_size",    type=int, default=10,
                   help="Steps per JIT-compiled scan call")
    r.add_argument("--save_interval", type=int, default=10,
                   help="Steps between NetCDF writes (independent of chunk_size)")

    # Output
    p.add_argument("--output", type=str, default="output.nc",
                   help="Output NetCDF file path")

    return p.parse_args(argv)


def _validate_args(args: argparse.Namespace) -> None:
    errors = []
    # run control
    if args.chunk_size < 1:
        errors.append("chunk_size must be >= 1")
    if args.n_steps < 1:
        errors.append("n_steps must be >= 1")
    if args.save_interval < 1:
        errors.append("save_interval must be >= 1")
    # grid geometry
    if args.dt <= 0:
        errors.append("dt must be > 0")
    if args.depth_max <= 0:
        errors.append("depth_max must be > 0")
    if args.lon_max <= args.lon_min:
        errors.append("lon_max must be > lon_min")
    if args.lat_max <= args.lat_min:
        errors.append("lat_max must be > lat_min")
    # initialisation
    if args.init == "oras5" and args.oras5_path is None:
        errors.append("--oras5_path is required when --init oras5")
    if errors:
        for e in errors:
            print(f"ERROR: {e}", file=sys.stderr)
        sys.exit(1)


# ---------------------------------------------------------------------------
# Initial state
# ---------------------------------------------------------------------------

def _build_initial_state(args: argparse.Namespace, grid: OceanGrid) -> OceanState:
    """Build the initial OceanState according to --init mode."""
    if args.init == "rest":
        return create_rest_state(grid, T_background=args.T_bg,
                                 S_background=args.S_bg)
    # oras5
    from OceanJAX.data.oras5 import load_oras5
    print(
        f"Loading ORAS5 from {args.oras5_path!r}  "
        f"(time_index={args.oras5_time_index})",
        file=sys.stderr,
    )
    return load_oras5(
        args.oras5_path,
        grid,
        time_index=args.oras5_time_index,
        T_fill=args.oras5_T_fill,
        S_fill=args.oras5_S_fill,
    )


# ---------------------------------------------------------------------------
# Forcing helper
# ---------------------------------------------------------------------------

def _make_forcing(n: int, args: argparse.Namespace, grid: OceanGrid) -> SurfaceForcing | None:
    """Build a constant SurfaceForcing with a leading time axis of length n."""
    if not any([args.heat_flux, args.fw_flux, args.tau_x, args.tau_y]):
        return None
    ones = jnp.ones((n, grid.Nx, grid.Ny), dtype=jnp.float32)
    return SurfaceForcing(
        heat_flux=ones * args.heat_flux,
        fw_flux=ones   * args.fw_flux,
        tau_x=ones     * args.tau_x,
        tau_y=ones     * args.tau_y,
    )


# ---------------------------------------------------------------------------
# NetCDF helpers
# ---------------------------------------------------------------------------

def _create_output_file(path: str, grid: OceanGrid, args: argparse.Namespace) -> nc.Dataset:
    """
    Create and initialise the NetCDF output file.

    Dimension order for data variables matches the internal JAX array layout:
      T, S : (time, x, y, z)  [Nx, Ny, Nz in memory]
      eta  : (time, x, y)     [Nx, Ny]
    No transposition is needed when writing.
    """
    ds = nc.Dataset(path, mode="w", format="NETCDF4")

    # Global attributes
    ds.description   = "OceanJAX idealised simulation output"
    ds.Conventions   = "CF-1.8"
    ds.history       = " ".join(sys.argv)
    ds.dt            = args.dt
    ds.nx            = args.nx
    ds.ny            = args.ny
    ds.nz            = args.nz
    ds.lon_min       = args.lon_min
    ds.lon_max       = args.lon_max
    ds.lat_min       = args.lat_min
    ds.lat_max       = args.lat_max
    ds.depth_max     = args.depth_max
    ds.n_steps       = args.n_steps
    ds.chunk_size    = args.chunk_size
    ds.save_interval = args.save_interval
    ds.init_mode     = args.init
    if args.init == "oras5":
        ds.oras5_path       = str(args.oras5_path)
        ds.oras5_time_index = args.oras5_time_index

    # Dimensions
    ds.createDimension("time", None)   # unlimited
    ds.createDimension("x",    grid.Nx)
    ds.createDimension("y",    grid.Ny)
    ds.createDimension("z",    grid.Nz)

    # Coordinate variables
    v = ds.createVariable("time", "f4", ("time",))
    v.units     = "s"
    v.long_name = "elapsed simulation time"

    v = ds.createVariable("x", "f4", ("x",))
    v.units     = "degrees_east"
    v.long_name = "tracer-cell longitude"
    v[:]        = np.array(grid.lon_c)

    v = ds.createVariable("y", "f4", ("y",))
    v.units     = "degrees_north"
    v.long_name = "tracer-cell latitude"
    v[:]        = np.array(grid.lat_c)

    v = ds.createVariable("z", "f4", ("z",))
    v.units     = "m"
    v.long_name = "tracer-cell depth (positive downward)"
    v.positive  = "down"
    v[:]        = np.array(grid.z_c)

    # Data variables
    v = ds.createVariable("T", "f4", ("time", "x", "y", "z"),
                          fill_value=np.float32(np.nan))
    v.units     = "degC"
    v.long_name = "potential temperature"

    v = ds.createVariable("S", "f4", ("time", "x", "y", "z"),
                          fill_value=np.float32(np.nan))
    v.units     = "psu"
    v.long_name = "practical salinity"

    v = ds.createVariable("eta", "f4", ("time", "x", "y"),
                          fill_value=np.float32(np.nan))
    v.units     = "m"
    v.long_name = "sea-surface height"

    return ds


def _append_snapshot(ds: nc.Dataset, state: OceanState) -> None:
    """Append the current state as one time record."""
    i = len(ds.variables["time"])
    ds.variables["time"][i]       = float(state.time)
    ds.variables["T"][i, :, :, :] = np.array(state.T)
    ds.variables["S"][i, :, :, :] = np.array(state.S)
    ds.variables["eta"][i, :, :]  = np.array(state.eta)
    ds.sync()


# ---------------------------------------------------------------------------
# Diagnostics
# ---------------------------------------------------------------------------

def _print_diag(step: int, n_steps: int, state: OceanState, wall_dt: float) -> bool:
    """Print one diagnostic line to stderr.  Returns True if any non-finite value is detected."""
    T   = np.array(state.T)
    S   = np.array(state.S)
    u   = np.array(state.u)
    v   = np.array(state.v)
    eta = np.array(state.eta)
    # Use isfinite rather than isnan to also catch Inf values (e.g. blow-up
    # before the field reaches NaN).
    not_finite = bool(
        np.any(~np.isfinite(T))   or
        np.any(~np.isfinite(S))   or
        np.any(~np.isfinite(u))   or
        np.any(~np.isfinite(v))   or
        np.any(~np.isfinite(eta))
    )

    print(
        f"[step {step:05d}/{n_steps:05d}]  "
        f"t={float(state.time):.1f} s  "
        f"T=[{T.min():.4f}, {T.max():.4f}]  "
        f"S=[{S.min():.4f}, {S.max():.4f}]  "
        f"eta=[{eta.min():.5f}, {eta.max():.5f}]  "
        f"non-finite={not_finite}  "
        f"wall={wall_dt:.2f}s",
        file=sys.stderr,
        flush=True,
    )
    return not_finite


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(argv=None) -> None:
    args = _parse_args(argv)
    _validate_args(args)

    # --- build grid ---
    dz = args.depth_max / args.nz
    depth_levels = (np.arange(args.nz) + 0.5) * dz
    grid = OceanGrid.create(
        lon_bounds=(args.lon_min, args.lon_max),
        lat_bounds=(args.lat_min, args.lat_max),
        depth_levels=depth_levels,
        Nx=args.nx,
        Ny=args.ny,
    )

    # --- model parameters and initial state ---
    params = ModelParams(dt=args.dt)
    state  = _build_initial_state(args, grid)

    # --- JIT-compiled scan loop ---
    # n_steps is static: JAX caches a separate compiled trace per unique value.
    # With exact-save chunking the effective chunk size is
    # min(chunk_size, save_interval), so at most two distinct sizes are compiled.
    run_jit = jax.jit(ocean_run, static_argnames=("n_steps", "save_history"))

    # --- open output file and save t=0 ---
    ds = _create_output_file(args.output, grid, args)
    _append_snapshot(ds, state)
    print(f"Output: {args.output}  (t=0 saved)", file=sys.stderr)

    steps_done     = 0
    next_save_step = args.save_interval

    try:
        while steps_done < args.n_steps:
            steps_remaining  = args.n_steps - steps_done
            steps_until_save = next_save_step - steps_done
            # Cap chunk at the next save boundary so saves are exact.
            chunk_steps = min(args.chunk_size, steps_remaining, steps_until_save)

            t0 = _time.perf_counter()
            forcing_seq = _make_forcing(chunk_steps, args, grid)
            state, _ = run_jit(
                state, grid, params,
                n_steps=chunk_steps,
                forcing_sequence=forcing_seq,
                save_history=False,
            )
            jax.block_until_ready(state.T)

            wall_dt     = _time.perf_counter() - t0
            steps_done += chunk_steps

            # NaN check before writing — avoids saving a corrupted snapshot.
            has_nan = _print_diag(steps_done, args.n_steps, state, wall_dt)
            if has_nan:
                print("ERROR: non-finite value detected — aborting.", file=sys.stderr)
                sys.exit(1)

            if steps_done == next_save_step:
                _append_snapshot(ds, state)
                next_save_step += args.save_interval

    finally:
        ds.close()

    print(f"Done. {steps_done} steps written to {args.output}", file=sys.stderr)


if __name__ == "__main__":
    main()
