"""
experiment.py
=============
OceanJAX experiment template.  Edit the CONFIG block below, then run:

    python experiment.py

Nothing else needs to change.
"""

from __future__ import annotations

import sys
import time as _time
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import netCDF4 as nc_lib

# ==============================================================================
# EXPERIMENT CONFIGURATION — only edit this section
# ==============================================================================

# --- Domain & resolution ------------------------------------------------------
LON       = (-40.0, -5.0)    # (lon_min, lon_max) degrees east
LAT       = (-15.0, 15.0)    # (lat_min, lat_max) degrees north
DEPTH_MAX = 500.0             # m
NX, NY, NZ = 20, 15, 10      # grid cells in x, y, z
DT        = 300.0             # time step [s]

# --- Initial conditions -------------------------------------------------------
#   "rest"       — uniform T_BG / S_BG, zero velocity
#   "oras5_cold" — T/S from ORAS5, u = v = eta = 0  (recommended: most stable)
#   "oras5_full" — full ORAS5 state (T, S, u, v, eta)
INIT_MODE  = "oras5_cold"
ORAS5_PATH = "OceanJAX/data/data_oras5/oras5_2026_01_native_merged.nc"
ORAS5_TIME_INDEX = 0          # time slice index in the ORAS5 file
T_BG, S_BG = 10.0, 35.0      # background T [°C] and S [psu] for "rest" mode

# --- Integration length -------------------------------------------------------
N_DAYS = 30                   # total simulation length in days

# --- Surface forcing ----------------------------------------------------------
#
# FORCING_SOURCE controls where forcing data comes from:
#
#   None         — use only the constant values below (HEAT_FLUX etc.)
#   "<path>"     — read from a NetCDF file (ORAS5 flux file, ERA5, etc.)
#                  Fields listed in FORCING_FIELDS are read from the file;
#                  the rest fall back to the constant values below.
#
# FORCING_FIELDS: which fields to read from the file.  Any subset of:
#   {"heat_flux", "fw_flux", "tau_x", "tau_y"}
#
# Constant fallback values (used when FORCING_SOURCE is None, or for fields
# not listed in FORCING_FIELDS, or absent from the file):
#   HEAT_FLUX  [W m-2]  net downward heat flux   (positive = warming ocean)
#   FW_FLUX    [m s-1]  net E-P freshwater flux  (positive = net evaporation)
#   TAU_X      [N m-2]  zonal wind stress        (positive = eastward)
#   TAU_Y      [N m-2]  meridional wind stress   (positive = northward)

FORCING_SOURCE = [                                     # list of per-field ORAS5 files
    "OceanJAX/data/data_oras5/sohefldo_control_monthly_highres_2D_202601_OPER_v0.1.nc",
    "OceanJAX/data/data_oras5/sowaflup_control_monthly_highres_2D_202601_OPER_v0.1.nc",
    "OceanJAX/data/data_oras5/sozotaux_control_monthly_highres_2D_202601_OPER_v0.1.nc",
    "OceanJAX/data/data_oras5/sometauy_control_monthly_highres_2D_202601_OPER_v0.1.nc",
]
FORCING_FIELDS = {"heat_flux", "fw_flux", "tau_x", "tau_y"}
HEAT_FLUX = 0.0
FW_FLUX   = 0.0
TAU_X     = 0.0
TAU_Y     = 0.0

# --- Ensemble / multi-GPU -----------------------------------------------------
#
# N_ENSEMBLE = 1  → single run (default, identical to previous behaviour).
# N_ENSEMBLE > 1  → batch / ensemble run using OceanJAX.parallel.
#
#   Each member starts from the same base initial condition, optionally with
#   independent Gaussian T perturbations (std = ENSEMBLE_PERTURB_T).
#   Members share the same grid, params, and surface forcing.
#
#   Execution:
#     - Single GPU  : vmap over the batch dimension (batch_run).
#     - Multiple GPU: batch dimension sharded across devices via
#                     NamedSharding (sharded_ensemble_run).
#
#   Output: NetCDF gains a "member" dimension;
#           diagnostics print ensemble mean ± spread.
#
# Note: N_ENSEMBLE must be divisible by the number of available GPUs.

N_ENSEMBLE        = 1      # number of ensemble members (1 = single run)
ENSEMBLE_PERTURB_T = 0.0   # Gaussian T perturbation std [°C] for each member

# --- Output -------------------------------------------------------------------
OUTPUT_NC     = "output_cold_full_forcing.nc"
SAVE_INTERVAL = 288   # steps between NetCDF snapshots  (288 × 300 s = 1 day)
CHUNK_SIZE    = 288   # steps per JIT-compiled scan call

# ==============================================================================
# END OF CONFIGURATION
# ==============================================================================


_SCRIPT_DIR = Path(__file__).parent
_ORAS5_FILE = _SCRIPT_DIR / ORAS5_PATH


# ---------------------------------------------------------------------------
# Grid & state builders
# ---------------------------------------------------------------------------

def _build_grid():
    from OceanJAX.grid import OceanGrid
    dz           = DEPTH_MAX / NZ
    depth_levels = (np.arange(NZ) + 0.5) * dz
    return OceanGrid.create(
        lon_bounds   = LON,
        lat_bounds   = LAT,
        depth_levels = depth_levels,
        Nx           = NX,
        Ny           = NY,
    )


def _build_state(grid):
    import warnings
    from OceanJAX.state import create_rest_state, create_from_arrays
    from OceanJAX.data.oras5 import read_oras5, regrid_to_model

    if INIT_MODE == "rest":
        print(f"Init: rest  T={T_BG} °C  S={S_BG} psu")
        return create_rest_state(grid, T_background=T_BG, S_background=S_BG)

    if not _ORAS5_FILE.exists():
        print(f"ERROR: ORAS5 file not found: {_ORAS5_FILE}", file=sys.stderr)
        sys.exit(1)

    print(f"Reading ORAS5 from {_ORAS5_FILE} (time_index={ORAS5_TIME_INDEX}) ...")
    t0  = _time.perf_counter()
    raw = read_oras5(_ORAS5_FILE, time_index=ORAS5_TIME_INDEX)
    print(f"  done in {_time.perf_counter() - t0:.1f} s")

    with warnings.catch_warnings(record=True):
        warnings.simplefilter("always")
        full_state = regrid_to_model(raw, grid)

    if INIT_MODE == "oras5_full":
        state = full_state
        u_a, eta_a = np.array(state.u), np.array(state.eta)
        print(f"Init: oras5_full  "
              f"u=[{u_a.min():.3f}, {u_a.max():.3f}] m/s  "
              f"eta=[{eta_a.min():.3f}, {eta_a.max():.3f}] m")
    else:  # oras5_cold
        zeros3 = jnp.zeros((NX, NY, NZ), dtype=jnp.float32)
        zeros2 = jnp.zeros((NX, NY),     dtype=jnp.float32)
        state  = create_from_arrays(
            grid, u=zeros3, v=zeros3,
            T=full_state.T, S=full_state.S, eta=zeros2,
        )
        print("Init: oras5_cold  u=v=eta=0")

    T_a, S_a = np.array(state.T), np.array(state.S)
    wet = np.asarray(grid.mask_c) > 0
    print(f"  T_wet=[{T_a[wet].min():.2f}, {T_a[wet].max():.2f}] °C  "
          f"S_wet=[{S_a[wet].min():.2f}, {S_a[wet].max():.2f}] psu")
    return state


def _build_ensemble_states(base_state, grid):
    """
    Stack N_ENSEMBLE copies of base_state into a batched OceanState.
    Optionally adds independent Gaussian T perturbations (std = ENSEMBLE_PERTURB_T).

    Returns an OceanState whose array fields carry a leading axis of size N_ENSEMBLE,
    e.g. T.shape == (N_ENSEMBLE, NX, NY, NZ).
    """
    import equinox as eqx

    # Add leading batch axis by stacking N_ENSEMBLE copies
    batched = jax.tree_util.tree_map(
        lambda x: jnp.stack([x] * N_ENSEMBLE), base_state
    )

    if ENSEMBLE_PERTURB_T > 0.0:
        key   = jax.random.PRNGKey(0)
        keys  = jax.random.split(key, N_ENSEMBLE)
        noise = jax.vmap(
            lambda k: jax.random.normal(k, base_state.T.shape, dtype=jnp.float32)
        )(keys) * ENSEMBLE_PERTURB_T                           # (N_ENSEMBLE, NX, NY, NZ)
        T_new = (batched.T + noise) * grid.mask_c[None]
        batched = eqx.tree_at(lambda s: s.T, batched, T_new)

    return batched


# ---------------------------------------------------------------------------
# Forcing builder
# ---------------------------------------------------------------------------

def _build_forcing(n_steps: int, grid):
    """
    Build a SurfaceForcing for one chunk of n_steps.

    Priority:
      1. If FORCING_SOURCE is set, read the listed FORCING_FIELDS from that file.
      2. Any field not in FORCING_FIELDS (or absent from the file) falls back to
         the corresponding constant (HEAT_FLUX, FW_FLUX, TAU_X, TAU_Y).
      3. If all four constants are zero and FORCING_SOURCE is None, return None.
    """
    from OceanJAX.timeStepping import SurfaceForcing

    const = {
        "heat_flux": float(HEAT_FLUX),
        "fw_flux":   float(FW_FLUX),
        "tau_x":     float(TAU_X),
        "tau_y":     float(TAU_Y),
    }

    if FORCING_SOURCE is None and not any(const.values()):
        return None

    base: dict[str, np.ndarray] = {
        k: np.full((NX, NY), v, dtype=np.float32) for k, v in const.items()
    }

    if FORCING_SOURCE is not None:
        from OceanJAX.data.oras5 import read_oras5_forcing, regrid_forcing
        raw_f   = read_oras5_forcing(FORCING_SOURCE, time_index=0)
        file_sf = regrid_forcing(raw_f, grid, use_fields=FORCING_FIELDS)
        for key in ("heat_flux", "fw_flux", "tau_x", "tau_y"):
            if key in FORCING_FIELDS:
                base[key] = np.asarray(getattr(file_sf, key))

    # Shape: (n_steps, NX, NY)
    ones = np.ones((n_steps, 1, 1), dtype=np.float32)
    return SurfaceForcing(
        heat_flux = jnp.asarray(base["heat_flux"] * ones),
        fw_flux   = jnp.asarray(base["fw_flux"]   * ones),
        tau_x     = jnp.asarray(base["tau_x"]     * ones),
        tau_y     = jnp.asarray(base["tau_y"]     * ones),
    )


def _broadcast_forcing_to_ensemble(forcing, n_members: int):
    """
    Add a leading ensemble axis to a SurfaceForcing chunk.
    Shape: (n_steps, NX, NY) → (n_members, n_steps, NX, NY).
    Returns None if forcing is None.
    """
    if forcing is None:
        return None
    return jax.tree_util.tree_map(
        lambda x: jnp.broadcast_to(x[None], (n_members,) + x.shape),
        forcing,
    )


# ---------------------------------------------------------------------------
# NetCDF helpers
# ---------------------------------------------------------------------------

def _create_nc(path: str, grid) -> nc_lib.Dataset:
    ds = nc_lib.Dataset(path, mode="w", format="NETCDF4")
    ds.description = f"OceanJAX experiment  init={INIT_MODE}  N_ensemble={N_ENSEMBLE}"
    ds.domain      = f"lon={LON} lat={LAT}"
    ds.dt          = DT
    ds.n_days      = N_DAYS
    ds.heat_flux   = HEAT_FLUX
    ds.fw_flux     = FW_FLUX
    ds.tau_x       = TAU_X
    ds.tau_y       = TAU_Y
    ds.n_ensemble  = N_ENSEMBLE

    ds.createDimension("time",   None)
    ds.createDimension("x",      grid.Nx)
    ds.createDimension("y",      grid.Ny)
    ds.createDimension("z",      grid.Nz)
    ds.createDimension("zw",     grid.Nz + 1)

    v = ds.createVariable("time", "f4", ("time",));  v.units = "s"
    v = ds.createVariable("x",    "f4", ("x",));     v.units = "degrees_east";  v[:] = np.array(grid.lon_c)
    v = ds.createVariable("y",    "f4", ("y",));     v.units = "degrees_north"; v[:] = np.array(grid.lat_c)
    v = ds.createVariable("z",    "f4", ("z",));     v.units = "m";             v[:] = np.array(grid.z_c)
    v = ds.createVariable("zw",   "f4", ("zw",));    v.units = "m";             v[:] = np.array(grid.z_w)

    if N_ENSEMBLE > 1:
        ds.createDimension("member", N_ENSEMBLE)
        ds.createVariable("T",   "f4", ("time", "member", "x", "y", "z"),  fill_value=np.float32(np.nan))
        ds.createVariable("S",   "f4", ("time", "member", "x", "y", "z"),  fill_value=np.float32(np.nan))
        ds.createVariable("eta", "f4", ("time", "member", "x", "y"),        fill_value=np.float32(np.nan))
        ds.createVariable("u",   "f4", ("time", "member", "x", "y", "z"),  fill_value=np.float32(np.nan))
        ds.createVariable("v",   "f4", ("time", "member", "x", "y", "z"),  fill_value=np.float32(np.nan))
        ds.createVariable("w",   "f4", ("time", "member", "x", "y", "zw"), fill_value=np.float32(np.nan))
    else:
        ds.createVariable("T",   "f4", ("time", "x", "y", "z"),  fill_value=np.float32(np.nan))
        ds.createVariable("S",   "f4", ("time", "x", "y", "z"),  fill_value=np.float32(np.nan))
        ds.createVariable("eta", "f4", ("time", "x", "y"),        fill_value=np.float32(np.nan))
        ds.createVariable("u",   "f4", ("time", "x", "y", "z"),  fill_value=np.float32(np.nan))
        ds.createVariable("v",   "f4", ("time", "x", "y", "z"),  fill_value=np.float32(np.nan))
        ds.createVariable("w",   "f4", ("time", "x", "y", "zw"), fill_value=np.float32(np.nan))
    # attach units
    ds.variables["T"].units   = "degC"
    ds.variables["S"].units   = "psu"
    ds.variables["eta"].units = "m"
    ds.variables["u"].units   = "m s-1"
    ds.variables["v"].units   = "m s-1"
    ds.variables["w"].units   = "m s-1"
    return ds


def _write_snapshot(ds: nc_lib.Dataset, state) -> None:
    """Write one time record.  Handles both single and ensemble states."""
    i = len(ds.variables["time"])
    # For ensemble state, state.time has shape (N_ENSEMBLE,); use member 0's time.
    t = float(state.time) if state.time.ndim == 0 else float(state.time[0])
    ds.variables["time"][i] = t
    if N_ENSEMBLE > 1:
        # state arrays: (N_ENSEMBLE, NX, NY, NZ) or (N_ENSEMBLE, NX, NY)
        ds.variables["T"][i,   :, :, :, :] = np.array(state.T)
        ds.variables["S"][i,   :, :, :, :] = np.array(state.S)
        ds.variables["eta"][i, :, :, :]    = np.array(state.eta)
        ds.variables["u"][i,   :, :, :, :] = np.array(state.u)
        ds.variables["v"][i,   :, :, :, :] = np.array(state.v)
        ds.variables["w"][i,   :, :, :, :] = np.array(state.w)
    else:
        ds.variables["T"][i,   :, :, :] = np.array(state.T)
        ds.variables["S"][i,   :, :, :] = np.array(state.S)
        ds.variables["eta"][i, :, :]    = np.array(state.eta)
        ds.variables["u"][i,   :, :, :] = np.array(state.u)
        ds.variables["v"][i,   :, :, :] = np.array(state.v)
        ds.variables["w"][i,   :, :, :] = np.array(state.w)
    ds.sync()


def _diag_line(state, sim_day: float, steps_done: int, wall: float) -> bool:
    """
    Print one diagnostic line.  Handles both single and ensemble states.
    Returns True if any non-finite value is detected.
    """
    if N_ENSEMBLE > 1:
        # Ensemble: report mean ± std across members
        T_all   = np.array(state.T)    # (B, NX, NY, NZ)
        S_all   = np.array(state.S)
        eta_all = np.array(state.eta)
        bad = (not np.all(np.isfinite(T_all)) or
               not np.all(np.isfinite(S_all)) or
               not np.all(np.isfinite(eta_all)))
        T_mean   = T_all.mean(axis=0);    T_std  = T_all.std(axis=0)
        eta_mean = eta_all.mean(axis=0);  eta_std = eta_all.std(axis=0)
        print(f"{sim_day:5.1f}  {steps_done:6d}  "
              f"T_mean=[{T_mean.min():.3f},{T_mean.max():.3f}] "
              f"±{T_std.max():.4f}  "
              f"eta_mean=[{eta_mean.min():.4f},{eta_mean.max():.4f}] "
              f"±{eta_std.max():.5f}  "
              f"{'NON-FINITE!' if bad else 'ok':>10}  {wall:5.1f}s",
              flush=True)
    else:
        T_a   = np.array(state.T)
        S_a   = np.array(state.S)
        eta_a = np.array(state.eta)
        bad   = (not np.all(np.isfinite(T_a)) or
                 not np.all(np.isfinite(S_a)) or
                 not np.all(np.isfinite(eta_a)))
        print(f"{sim_day:5.1f}  {steps_done:6d}  "
              f"{T_a.min():7.3f} {T_a.max():7.3f}  "
              f"{S_a.min():6.3f} {S_a.max():6.3f}  "
              f"{eta_a.min():8.4f} {eta_a.max():8.4f}  "
              f"{'NON-FINITE!' if bad else 'ok':>10}  {wall:5.1f}s",
              flush=True)
    return bad


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    from OceanJAX.state import ModelParams
    from OceanJAX.timeStepping import run as ocean_run

    n_steps   = round(N_DAYS * 86400 / DT)
    ensemble  = N_ENSEMBLE > 1
    n_devices = len(jax.devices())

    print("=" * 62)
    print(f"OceanJAX experiment")
    print(f"  domain    : lon={LON}  lat={LAT}  depth={DEPTH_MAX} m")
    print(f"  grid      : {NX}x{NY}x{NZ}  dt={DT} s")
    print(f"  run       : {N_DAYS} days  ({n_steps} steps)")
    if ensemble:
        print(f"  ensemble  : {N_ENSEMBLE} members  "
              f"perturb_T={ENSEMBLE_PERTURB_T} °C  "
              f"devices={n_devices}")
    print(f"  output    : {OUTPUT_NC}  save_every={SAVE_INTERVAL} steps")
    print("=" * 62)

    grid   = _build_grid()
    params = ModelParams(dt=DT)

    # Build initial state(s)
    base_state = _build_state(grid)
    if ensemble:
        state = _build_ensemble_states(base_state, grid)
        print(f"  Ensemble of {N_ENSEMBLE} members created.")
    else:
        state = base_state

    # Compile run function
    if ensemble:
        from OceanJAX.parallel.ensemble import sharded_ensemble_run
        # sharded_ensemble_run handles jit internally
        def _run_chunk(s, chunk, forcing):
            f_batch = _broadcast_forcing_to_ensemble(forcing, N_ENSEMBLE)
            return sharded_ensemble_run(s, grid, params, chunk,
                                        forcing_sequence=f_batch,
                                        save_history=False)
    else:
        run_jit = jax.jit(ocean_run, static_argnames=("n_steps", "save_history"))
        def _run_chunk(s, chunk, forcing):
            return run_jit(s, grid, params, n_steps=chunk,
                           forcing_sequence=forcing, save_history=False)

    # Open output file and save t=0
    ds = _create_nc(OUTPUT_NC, grid)
    _write_snapshot(ds, state)
    print(f"\nOutput: {OUTPUT_NC}  (t=0 saved)\n")

    if ensemble:
        print(f"{'Day':>5}  {'Step':>6}  "
              f"{'T_mean range':^25}  {'eta_mean range':^22}  "
              f"{'status':>10}  {'wall':>6}")
    else:
        print(f"{'Day':>5}  {'Step':>6}  {'T_min':>7} {'T_max':>7}  "
              f"{'S_min':>6} {'S_max':>6}  {'eta_min':>8} {'eta_max':>8}  "
              f"{'status':>10}  {'wall':>6}")
    print("-" * 90)

    steps_done     = 0
    next_save_step = SAVE_INTERVAL
    all_ok         = True

    try:
        while steps_done < n_steps:
            steps_remaining  = n_steps - steps_done
            steps_until_save = next_save_step - steps_done
            chunk = min(CHUNK_SIZE, steps_remaining, steps_until_save)

            forcing = _build_forcing(chunk, grid)

            t0 = _time.perf_counter()
            state, _ = _run_chunk(state, chunk, forcing)
            jax.block_until_ready(state.T)
            wall = _time.perf_counter() - t0

            steps_done += chunk
            # For ensemble, each member has its own time scalar; use member 0.
            sim_day = float(state.time) / 86400.0 if state.time.ndim == 0 \
                      else float(state.time[0])   / 86400.0

            bad = _diag_line(state, sim_day, steps_done, wall)

            if steps_done == next_save_step:
                _write_snapshot(ds, state)
                next_save_step += SAVE_INTERVAL

            if bad:
                print("\nABORTED: non-finite values detected.", file=sys.stderr)
                all_ok = False
                break

    finally:
        ds.close()

    print("-" * 90)
    label = f"stable for {N_DAYS} days" if all_ok else "model blew up"
    if ensemble:
        label += f"  [{N_ENSEMBLE} members]"
    print(f"\n{'PASS' if all_ok else 'FAIL'} — {label}")
    print(f"Output: {OUTPUT_NC}")
    sys.exit(0 if all_ok else 1)


if __name__ == "__main__":
    main()
