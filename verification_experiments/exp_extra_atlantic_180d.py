"""
Extra Experiment — Full Atlantic Basin, 180-day Integration
============================================================
Domain
------
  lon = (-80, 20)   lat = (-50, 60)   depth = 1000 m
  Grid: 50 x 40 x 10,  dt = 300 s

  Covers the full Atlantic basin from the Southern Ocean boundary
  to the subpolar North Atlantic, including:
    - Equatorial current system
    - Subtropical gyres (North and South Atlantic)
    - Gulf Stream and its extension region
    - Antarctic Circumpolar Current margin

Initial conditions
------------------
  oras5_cold: T/S from ORAS5 Jan-2026, u = v = eta = 0

Forcing
-------
  Full ORAS5 Jan-2026 surface fluxes (monthly mean, repeated for 180 days):
    heat_flux, fw_flux, tau_x, tau_y

Integration
-----------
  180 days (51840 steps at dt=300s)
  Snapshot saved every day (288 steps)
  Output: output_atlantic_180d.nc

Stability estimate
------------------
  c_bt = sqrt(9.81 * 1000) ~ 99 m/s
  dx_min ~ 200 km (50 cells over 100 deg lon ~ 222 km)
  CFL = 99 * 300 / 200000 ~ 0.15  (well within stability boundary)
"""

from __future__ import annotations

import sys
import time as _time
import warnings
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import jax
import jax.numpy as jnp
import netCDF4 as nc_lib
from OceanJAX.grid import OceanGrid
from OceanJAX.state import ModelParams, create_from_arrays
from OceanJAX.data.oras5 import (read_oras5, regrid_to_model,
                                  read_oras5_forcing, regrid_forcing)
from OceanJAX.timeStepping import SurfaceForcing, run

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
ORAS5_IC      = "OceanJAX/data/data_oras5/oras5_2026_01_native_merged.nc"
FORCING_FILES = [
    "OceanJAX/data/data_oras5/sohefldo_control_monthly_highres_2D_202601_OPER_v0.1.nc",
    "OceanJAX/data/data_oras5/sowaflup_control_monthly_highres_2D_202601_OPER_v0.1.nc",
    "OceanJAX/data/data_oras5/sozotaux_control_monthly_highres_2D_202601_OPER_v0.1.nc",
    "OceanJAX/data/data_oras5/sometauy_control_monthly_highres_2D_202601_OPER_v0.1.nc",
]
OUTPUT_NC = "output_atlantic_180d.nc"

LON, LAT   = (-80.0, 20.0), (-50.0, 60.0)
DEPTH_MAX  = 1000.0
NX, NY, NZ = 50, 40, 10
DT         = 300.0
N_DAYS     = 180

STEPS_PER_DAY = int(86400 / DT)    # 288
TOTAL_STEPS   = N_DAYS * STEPS_PER_DAY
SAVE_INTERVAL = STEPS_PER_DAY      # save every day

# ---------------------------------------------------------------------------
# Build grid
# ---------------------------------------------------------------------------
dz           = DEPTH_MAX / NZ
depth_levels = (np.arange(NZ) + 0.5) * dz    # [50, 150, ..., 950] m

print("=" * 66)
print("Extra Experiment: Full Atlantic Basin, 180-day Integration")
print(f"  Domain : lon={LON}  lat={LAT}  depth={DEPTH_MAX} m")
print(f"  Grid   : {NX}x{NY}x{NZ}  dt={DT} s")
print(f"  Run    : {N_DAYS} days  ({TOTAL_STEPS} steps)")
print(f"  Output : {OUTPUT_NC}")
print("=" * 66)

grid = OceanGrid.create(
    lon_bounds=LON, lat_bounds=LAT,
    depth_levels=depth_levels,
    Nx=NX, Ny=NY,
)
mask_np = np.array(grid.mask_c)
wet     = mask_np > 0
n_wet   = int(wet.sum())
n_total = NX * NY * NZ
print(f"  Wet cells: {n_wet}/{n_total}  ({100*n_wet/n_total:.1f}%)")

# ---------------------------------------------------------------------------
# Initial state: oras5_cold
# ---------------------------------------------------------------------------
print("\nLoading ORAS5 initial conditions ...", flush=True)
t0 = _time.time()
with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    raw        = read_oras5(ORAS5_IC, time_index=0)
    full_state = regrid_to_model(raw, grid)
print(f"  done in {_time.time()-t0:.1f} s")

zeros3 = jnp.zeros((NX, NY, NZ), dtype=jnp.float32)
zeros2 = jnp.zeros((NX, NY),     dtype=jnp.float32)
state  = create_from_arrays(grid, u=zeros3, v=zeros3,
                             T=full_state.T, S=full_state.S, eta=zeros2)

T0_np = np.array(state.T)
S0_np = np.array(state.S)
SST0  = float(T0_np[:, :, 0][mask_np[:, :, 0] > 0].mean())
print(f"  T_wet=[{T0_np[wet].min():.2f}, {T0_np[wet].max():.2f}] C")
print(f"  S_wet=[{S0_np[wet].min():.2f}, {S0_np[wet].max():.2f}] psu")
print(f"  SST_mean_0 = {SST0:.3f} C")

# ---------------------------------------------------------------------------
# Forcing: full ORAS5 surface fluxes (Jan-2026 monthly mean, repeated)
# ---------------------------------------------------------------------------
print("\nLoading ORAS5 surface forcing ...", flush=True)
with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    raw_f = read_oras5_forcing(FORCING_FILES, time_index=0)
    sf    = regrid_forcing(raw_f, grid,
                           use_fields={"heat_flux", "fw_flux", "tau_x", "tau_y"})

hf_np = np.array(sf.heat_flux)
fw_np = np.array(sf.fw_flux)
tx_np = np.array(sf.tau_x)
ty_np = np.array(sf.tau_y)

wet2 = mask_np[:, :, 0] > 0
print(f"  heat_flux: [{hf_np[wet2].min():.1f}, {hf_np[wet2].max():.1f}] W/m2  "
      f"mean={hf_np[wet2].mean():.2f}")
print(f"  tau_x    : [{tx_np[wet2].min():.3f}, {tx_np[wet2].max():.3f}] N/m2")
print(f"  tau_y    : [{ty_np[wet2].min():.3f}, {ty_np[wet2].max():.3f}] N/m2")


def _make_forcing(n_steps: int) -> SurfaceForcing:
    ones = np.ones((n_steps, 1, 1), dtype=np.float32)
    return SurfaceForcing(
        heat_flux = jnp.asarray(hf_np * ones),
        fw_flux   = jnp.asarray(fw_np * ones),
        tau_x     = jnp.asarray(tx_np * ones),
        tau_y     = jnp.asarray(ty_np * ones),
    )


params  = ModelParams(dt=DT)
run_jit = jax.jit(run, static_argnames=("n_steps", "save_history"))

# ---------------------------------------------------------------------------
# NetCDF output setup
# ---------------------------------------------------------------------------
def _create_nc(path: str) -> nc_lib.Dataset:
    ds = nc_lib.Dataset(path, mode="w", format="NETCDF4")
    ds.description = "OceanJAX Full Atlantic 180-day integration"
    ds.domain      = f"lon={LON} lat={LAT} depth={DEPTH_MAX}m"
    ds.grid        = f"{NX}x{NY}x{NZ}"
    ds.dt          = DT
    ds.n_days      = N_DAYS
    ds.createDimension("time", None)
    ds.createDimension("x",    NX)
    ds.createDimension("y",    NY)
    ds.createDimension("z",    NZ)
    ds.createDimension("zw",   NZ + 1)
    v = ds.createVariable("time", "f4", ("time",));  v.units = "days"
    v = ds.createVariable("x",    "f4", ("x",));     v.units = "deg_E";  v[:] = np.array(grid.lon_c)
    v = ds.createVariable("y",    "f4", ("y",));     v.units = "deg_N";  v[:] = np.array(grid.lat_c)
    v = ds.createVariable("z",    "f4", ("z",));     v.units = "m";      v[:] = np.array(grid.z_c)
    v = ds.createVariable("zw",   "f4", ("zw",));    v.units = "m";      v[:] = np.array(grid.z_w)
    ds.createVariable("T",   "f4", ("time","x","y","z"), fill_value=np.float32(np.nan)); ds["T"].units   = "degC"
    ds.createVariable("S",   "f4", ("time","x","y","z"), fill_value=np.float32(np.nan)); ds["S"].units   = "psu"
    ds.createVariable("eta", "f4", ("time","x","y"),     fill_value=np.float32(np.nan)); ds["eta"].units = "m"
    ds.createVariable("u",   "f4", ("time","x","y","z"), fill_value=np.float32(np.nan)); ds["u"].units   = "m/s"
    ds.createVariable("v",   "f4", ("time","x","y","z"), fill_value=np.float32(np.nan)); ds["v"].units   = "m/s"
    return ds


def _write_snapshot(ds: nc_lib.Dataset, state, day: float) -> None:
    i = len(ds.variables["time"])
    ds["time"][i]      = day
    ds["T"][i, :,:,:]  = np.array(state.T)
    ds["S"][i, :,:,:]  = np.array(state.S)
    ds["eta"][i, :,:]  = np.array(state.eta)
    ds["u"][i, :,:,:]  = np.array(state.u)
    ds["v"][i, :,:,:]  = np.array(state.v)
    ds.sync()


# ---------------------------------------------------------------------------
# Integration loop
# ---------------------------------------------------------------------------
print(f"\n{'Day':>5}  {'T_min':>7} {'T_max':>7}  {'S_min':>6} {'S_max':>6}  "
      f"{'eta_min':>8} {'eta_max':>8}  {'SST_mean':>9}  {'wall':>6}")
print("-" * 85)

ds = _create_nc(OUTPUT_NC)
_write_snapshot(ds, state, 0.0)

t0_wall  = _time.time()
all_ok   = True

for day in range(1, N_DAYS + 1):
    t0_chunk = _time.time()
    forcing  = _make_forcing(STEPS_PER_DAY)
    state, _ = run_jit(state, grid, params, n_steps=STEPS_PER_DAY,
                       forcing_sequence=forcing, save_history=False)
    jax.block_until_ready(state.T)
    chunk_wall = _time.time() - t0_chunk

    T_a   = np.array(state.T)
    S_a   = np.array(state.S)
    eta_a = np.array(state.eta)

    bad = not (np.all(np.isfinite(T_a)) and np.all(np.isfinite(S_a))
               and np.all(np.isfinite(eta_a)))

    T_min    = float(T_a[wet].min())
    T_max    = float(T_a[wet].max())
    S_min    = float(S_a[wet].min())
    S_max    = float(S_a[wet].max())
    eta_min  = float(eta_a.min())
    eta_max  = float(eta_a.max())
    sst_mean = float(T_a[:, :, 0][wet2].mean())

    print(f"{day:5d}  {T_min:7.3f} {T_max:7.3f}  {S_min:6.3f} {S_max:6.3f}  "
          f"{eta_min:8.4f} {eta_max:8.4f}  {sst_mean:9.3f}  {chunk_wall:5.1f}s",
          flush=True)

    _write_snapshot(ds, state, float(day))

    if bad:
        print("  [ABORT] Non-finite values.")
        all_ok = False
        break

ds.close()

total_wall = _time.time() - t0_wall

# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------
print()
print("=" * 66)
print(f"Total wall time : {total_wall:.1f} s  ({total_wall/60:.1f} min)")
print(f"Output file     : {OUTPUT_NC}")
print(f"Overall         : {'PASS' if all_ok else 'FAIL'}")
