"""
Experiment 5.2.2 — ORAS5 Cold Start, No Forcing (30 days)
==========================================================
Setup
-----
  - Same domain and grid as 5.2.1: 20×15×10, tropical Atlantic, dt=300 s
  - Init (oras5_cold): T/S from ORAS5 Jan-2026, u=v=eta=0
  - Forcing: NONE

Rationale
---------
Control run paired with 5.2.1 (same IC, no forcing).
Comparing 5.2.2 vs 5.2.1 isolates the forcing signal:
  - Any SST/SSS difference between the two runs is due to surface fluxes.
  - Without forcing, the ocean should evolve only through internal dynamics
    (pressure-driven adjustment, mixing), resulting in minimal SST change.

Pass criteria
-------------
  No non-finite values throughout
  T_wet ∈ [5, 35] °C
  S_wet ∈ [30, 40] psu
  |eta|_max < 2 m
  |ΔSST| < 1 °C over 30 days  (no forcing → minimal surface heat exchange)
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
from OceanJAX.grid import OceanGrid
from OceanJAX.state import ModelParams, create_from_arrays
from OceanJAX.data.oras5 import read_oras5, regrid_to_model
from OceanJAX.timeStepping import run

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
ORAS5_IC   = "OceanJAX/data/data_oras5/oras5_2026_01_native_merged.nc"

LON, LAT   = (-40.0, -5.0), (-15.0, 15.0)
DEPTH_MAX  = 500.0
NX, NY, NZ = 20, 15, 10
DT         = 300.0
N_DAYS     = 30

STEPS_PER_DAY = int(86400 / DT)

# ---------------------------------------------------------------------------
# Build grid
# ---------------------------------------------------------------------------
dz           = DEPTH_MAX / NZ
depth_levels = (np.arange(NZ) + 0.5) * dz

grid = OceanGrid.create(
    lon_bounds=LON, lat_bounds=LAT,
    depth_levels=depth_levels,
    Nx=NX, Ny=NY,
)

# ---------------------------------------------------------------------------
# Initial state: oras5_cold
# ---------------------------------------------------------------------------
print("Loading ORAS5 initial conditions ...", flush=True)
with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    raw        = read_oras5(ORAS5_IC, time_index=0)
    full_state = regrid_to_model(raw, grid)

zeros3 = jnp.zeros((NX, NY, NZ), dtype=jnp.float32)
zeros2 = jnp.zeros((NX, NY),     dtype=jnp.float32)
state  = create_from_arrays(grid, u=zeros3, v=zeros3,
                             T=full_state.T, S=full_state.S, eta=zeros2)

mask_np = np.array(grid.mask_c)
wet     = mask_np > 0
T0_np   = np.array(state.T)
SST0    = float(T0_np[:, :, 0][mask_np[:, :, 0] > 0].mean())

print(f"  T_wet=[{T0_np[wet].min():.2f}, {T0_np[wet].max():.2f}] °C")
S0_np = np.array(state.S)
print(f"  S_wet=[{S0_np[wet].min():.2f}, {S0_np[wet].max():.2f}] psu")
print(f"  SST_mean_0 = {SST0:.3f} °C")
print("  Forcing: NONE")

params  = ModelParams(dt=DT)
run_jit = jax.jit(run, static_argnames=("n_steps", "save_history"))

# ---------------------------------------------------------------------------
# Integration loop
# ---------------------------------------------------------------------------
print()
print(f"{'Day':>5}  {'T_min':>7} {'T_max':>7}  {'S_min':>6} {'S_max':>6}  "
      f"{'eta_min':>8} {'eta_max':>8}  {'SST_mean':>9}  {'status':>10}")
print("-" * 90)

records = []
t0_wall = _time.time()

for day in range(1, N_DAYS + 1):
    state, _ = run_jit(state, grid, params, n_steps=STEPS_PER_DAY,
                       forcing_sequence=None, save_history=False)

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
    sst_mean = float(T_a[:, :, 0][mask_np[:, :, 0] > 0].mean())

    records.append(dict(day=day, T_min=T_min, T_max=T_max,
                        S_min=S_min, S_max=S_max,
                        eta_min=eta_min, eta_max=eta_max,
                        sst_mean=sst_mean, bad=bad))

    print(f"{day:5d}  {T_min:7.3f} {T_max:7.3f}  {S_min:6.3f} {S_max:6.3f}  "
          f"{eta_min:8.4f} {eta_max:8.4f}  {sst_mean:9.3f}  "
          f"{'NON-FINITE' if bad else 'ok':>10}")

    if bad:
        print("  [ABORT]")
        break

wall = _time.time() - t0_wall

# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------
print()
print("=" * 90)
print(f"Wall time: {wall:.1f} s")
print()

SST_final   = records[-1]["sst_mean"]
dSST        = SST_final - SST0
T_min_all   = min(r["T_min"]  for r in records)
T_max_all   = max(r["T_max"]  for r in records)
S_min_all   = min(r["S_min"]  for r in records)
S_max_all   = max(r["S_max"]  for r in records)
eta_abs_max = max(max(abs(r["eta_min"]), abs(r["eta_max"])) for r in records)

print(f"SST:  initial={SST0:.3f} C  ->  final={SST_final:.3f} C  (dSST={dSST:+.3f} C)")
print(f"T_wet range over run: [{T_min_all:.3f}, {T_max_all:.3f}] °C")
print(f"S_wet range over run: [{S_min_all:.3f}, {S_max_all:.3f}] psu")
print(f"|eta|_max over run  : {eta_abs_max:.4f} m")
print()

checks = {
    "no non-finite values"            : not any(r["bad"] for r in records),
    "T_wet > 5 °C throughout"         : T_min_all > 5.0,
    "T_wet < 35 °C throughout"        : T_max_all < 35.0,
    "S_wet > 30 psu throughout"       : S_min_all > 30.0,
    "S_wet < 40 psu throughout"       : S_max_all < 40.0,
    "|eta|_max < 2 m"                 : eta_abs_max < 2.0,
    "|dSST| < 5 C (physical bound)"    : abs(dSST) < 5.0,
}

for name, passed in checks.items():
    print(f"  {'PASS' if passed else 'FAIL'}  {name}")

all_pass = all(checks.values())
print()
print("Overall:", "PASS" if all_pass else "FAIL")
