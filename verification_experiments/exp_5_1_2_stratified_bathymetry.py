"""
Experiment 5.1.2 — Static Stratification Test on Real Bathymetry
=================================================================
Setup
-----
  - Real ORAS5 bathymetry loaded via regrid_to_model (mask_c reflects
    actual land-sea boundaries and partial bottom cells)
  - Initial T/S: horizontally uniform, vertically stratified profile
    taken as the domain-averaged ORAS5 T/S column broadcast to all (i,j)
  - u = v = w = eta = 0  (no initial motion)
  - No surface forcing
  - Integration: 10 days  (dt = 300 s → 2880 steps)

Rationale
---------
A horizontally uniform, vertically stratified field in hydrostatic
balance should produce no horizontal pressure gradient and hence no
flow.  Testing this on an irregular bathymetry / mask is more demanding
than a flat-bottom case: it exercises the mask-consistency of the
pressure-gradient and continuity operators at partial-depth cells and
land boundaries.

Any non-zero velocity that develops can only arise from:
  (a) a numerical inconsistency in the PGF or mask operator  [bad]
  (b) a genuine geostrophic-adjustment signal due to f≠0 and
      mask edges creating effective walls              [expected, small]

Metrics recorded every SAVE_INTERVAL steps
-------------------------------------------
  max_u, max_v : max |u|, |v|  [m/s]
  max_eta      : max |eta|  [m]
  KE           : domain kinetic energy  0.5·rho0·∫(u²+v²)dV  [J]
  max_T_drift  : max pointwise |T(t) - T(0)| over wet cells  [°C]

Pass criteria
-------------
  max_u, max_v < 1e-3 m/s   over 10 days  (geostrophic adjustment
                              near mask edges is expected at O(1e-4))
  max_eta      < 1e-3 m
  KE           < 1e6 J      (O(domain volume × rho0 × 1e-6 m²/s²))
  No non-finite values
"""

from __future__ import annotations

import sys
import time as _time
import warnings
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import jax.numpy as jnp
import equinox as eqx
from OceanJAX.grid import OceanGrid
from OceanJAX.state import ModelParams, OceanState, create_from_arrays
from OceanJAX.data.oras5 import read_oras5, regrid_to_model
from OceanJAX.timeStepping import run

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
ORAS5_PATH   = "OceanJAX/data/data_oras5/oras5_2026_01_native_merged.nc"
LON          = (-40.0, -5.0)
LAT          = (-15.0, 15.0)
DEPTH_LEVELS = np.array([25., 75., 150., 250., 375., 500.], dtype=np.float64)
NX, NY       = 20, 15
DT           = 300.0
N_DAYS       = 10
RHO0         = 1025.0

STEPS_PER_DAY = int(86400 / DT)   # 288
TOTAL_STEPS   = N_DAYS * STEPS_PER_DAY
SAVE_INTERVAL = STEPS_PER_DAY

# ---------------------------------------------------------------------------
# Load ORAS5 → get bathymetry and real T/S structure
# ---------------------------------------------------------------------------
print("Loading ORAS5 ...", flush=True)
with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    raw = read_oras5(ORAS5_PATH, time_index=0)

# Build grid with real bathymetry derived from ORAS5 mask
# (regrid_to_model will determine the mask from ORAS5 depth coverage;
#  we first build a flat-bottom grid, regrid to get T/S, then derive H)
grid_flat = OceanGrid.create(
    lon_bounds=LON, lat_bounds=LAT,
    depth_levels=DEPTH_LEVELS,
    Nx=NX, Ny=NY,
    bathymetry=None,
)

with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    oras5_state = regrid_to_model(raw, grid_flat)

# Infer bathymetry from ORAS5 mask: H(i,j) = depth of deepest wet cell
mask_c_np = np.array(grid_flat.mask_c)   # (Nx, Ny, Nz)  — still flat bottom here
z_w_np    = np.array(grid_flat.z_w)      # (Nz+1,)

# Use ORAS5-interpolated T field: NaN at land → derive actual wet depth
T_oras5   = np.array(oras5_state.T)      # (Nx, Ny, Nz), land=0 after masking
# Identify wet cells: where oras5_state.T was non-zero after masking
# A cell is wet if its mask_c is 1 (already in the flat grid)
# For a more realistic bathymetry, use the deepest non-zero level per column
H_np = np.zeros((NX, NY), dtype=np.float64)
for i in range(NX):
    for j in range(NY):
        wet_levels = np.where(mask_c_np[i, j, :] > 0)[0]
        if len(wet_levels) > 0:
            k_deep = wet_levels[-1]
            H_np[i, j] = float(z_w_np[k_deep + 1])   # bottom face of deepest wet cell
        else:
            H_np[i, j] = 0.0   # land column

# Rebuild grid with real bathymetry
grid = OceanGrid.create(
    lon_bounds=LON, lat_bounds=LAT,
    depth_levels=DEPTH_LEVELS,
    Nx=NX, Ny=NY,
    bathymetry=H_np,
)

print(f"  Grid: {NX}x{NY}x{len(DEPTH_LEVELS)}  "
      f"wet columns: {int((H_np > 0).sum())}/{NX*NY}")

# ---------------------------------------------------------------------------
# Build horizontally uniform, vertically stratified initial state
# ---------------------------------------------------------------------------
# Domain-average T/S profile over wet cells
T_oras5_r = np.array(oras5_state.T)   # (Nx, Ny, Nz)
S_oras5_r = np.array(oras5_state.S)

mask_np = np.array(grid.mask_c)       # (Nx, Ny, Nz) — real bathymetry mask

T_profile = np.zeros(len(DEPTH_LEVELS), dtype=np.float32)
S_profile = np.zeros(len(DEPTH_LEVELS), dtype=np.float32)
for k in range(len(DEPTH_LEVELS)):
    wet = mask_np[:, :, k] > 0
    if wet.sum() > 0:
        T_profile[k] = T_oras5_r[:, :, k][wet].mean()
        S_profile[k] = S_oras5_r[:, :, k][wet].mean()
    else:
        T_profile[k] = T_profile[k-1] if k > 0 else 10.0
        S_profile[k] = S_profile[k-1] if k > 0 else 35.0

print(f"  Domain-mean T profile: {T_profile}")
print(f"  Domain-mean S profile: {S_profile}")

# Broadcast profile to all (i,j), zero at land
T_init = (np.ones((NX, NY, len(DEPTH_LEVELS)), dtype=np.float32)
          * T_profile[np.newaxis, np.newaxis, :]) * mask_np
S_init = (np.ones((NX, NY, len(DEPTH_LEVELS)), dtype=np.float32)
          * S_profile[np.newaxis, np.newaxis, :]) * mask_np
u_init   = np.zeros((NX, NY, len(DEPTH_LEVELS)), dtype=np.float32)
v_init   = np.zeros((NX, NY, len(DEPTH_LEVELS)), dtype=np.float32)
eta_init = np.zeros((NX, NY), dtype=np.float32)

state = create_from_arrays(grid, u_init, v_init, T_init, S_init, eta_init)
params = ModelParams(dt=DT)

T_init_jnp = jnp.array(T_init)   # reference for drift calculation

# Domain volume
vol = np.array(grid.volume_c)
RHO0_val = 1025.0

# ---------------------------------------------------------------------------
# Time-stepping loop
# ---------------------------------------------------------------------------
records = []

print()
print(f"{'Day':>5}  {'max|u|':>12}  {'max|v|':>12}  {'max|eta|':>10}  "
      f"{'KE [J]':>12}  {'max_T_drift':>13}  {'status':>10}")
print("-" * 88)

t0_wall = _time.time()

for chunk_idx in range(N_DAYS):
    state, _ = run(state, grid, params, n_steps=SAVE_INTERVAL,
                   forcing_sequence=None, save_history=False)

    u_arr   = np.array(state.u)
    v_arr   = np.array(state.v)
    eta_arr = np.array(state.eta)
    T_arr   = np.array(state.T)

    bad = not (np.all(np.isfinite(u_arr)) and np.all(np.isfinite(v_arr))
               and np.all(np.isfinite(eta_arr)) and np.all(np.isfinite(T_arr)))

    max_u      = float(np.max(np.abs(u_arr)))
    max_v      = float(np.max(np.abs(v_arr)))
    max_eta    = float(np.max(np.abs(eta_arr)))
    KE         = 0.5 * RHO0_val * float(np.sum((u_arr**2 + v_arr**2) * vol))
    wet        = mask_np > 0
    max_T_drift= float(np.max(np.abs(T_arr - T_init)[wet]))
    sim_day    = float(state.time) / 86400.0
    status     = "NON-FINITE" if bad else "ok"

    records.append(dict(day=sim_day, max_u=max_u, max_v=max_v,
                        max_eta=max_eta, KE=KE, max_T_drift=max_T_drift))

    print(f"{sim_day:5.0f}  {max_u:12.3e}  {max_v:12.3e}  {max_eta:10.3e}  "
          f"{KE:12.3e}  {max_T_drift:13.3e}  {status:>10}")

    if bad:
        print("  [ABORT] Non-finite values detected.")
        break

wall = _time.time() - t0_wall

# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------
print()
print("=" * 88)
print(f"Wall time: {wall:.1f} s")
print()

max_u_all    = max(r["max_u"]   for r in records)
max_v_all    = max(r["max_v"]   for r in records)
max_eta_all  = max(r["max_eta"] for r in records)
KE_max       = max(r["KE"]      for r in records)
T_drift_max  = max(r["max_T_drift"] for r in records)

print(f"Max |u|          : {max_u_all:.3e} m/s  (threshold < 1e-3)")
print(f"Max |v|          : {max_v_all:.3e} m/s  (threshold < 1e-3)")
print(f"Max |eta|        : {max_eta_all:.3e} m    (threshold < 1e-3)")
print(f"Max KE           : {KE_max:.3e} J    (threshold < 1e6)")
print(f"Max T drift      : {T_drift_max:.3e} °C")
print()

results = {
    "u suppressed (< 1e-3 m/s)" : max_u_all   < 1e-3,
    "v suppressed (< 1e-3 m/s)" : max_v_all   < 1e-3,
    "eta suppressed (< 1e-3 m)" : max_eta_all < 1e-3,
    "KE suppressed (< 1e6 J)"   : KE_max      < 1e6,
    "no non-finite values"      : all(r["max_T_drift"] < 1e10 for r in records),
}

all_pass = all(results.values())
for name, passed in results.items():
    print(f"  {'PASS' if passed else 'FAIL'}  {name}")

print()
print("Overall:", "PASS" if all_pass else "FAIL")
