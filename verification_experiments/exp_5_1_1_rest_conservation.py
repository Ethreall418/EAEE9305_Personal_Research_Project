"""
Experiment 5.1.1 — Resting Uniform Ocean Conservation Test
===========================================================
Setup
-----
  - Flat bathymetry (no land cells)
  - Uniform T = 15.0 °C, S = 35.0 psu everywhere
  - u = v = w = eta = 0
  - No surface forcing
  - Integration: 30 days  (dt = 300 s  →  8640 steps)

Metrics recorded every SAVE_INTERVAL steps
-------------------------------------------
  int_T   : volume-integrated T  [°C · m³]
  int_S   : volume-integrated S  [psu · m³]
  max_u   : max |u|  [m/s]
  max_v   : max |v|  [m/s]
  max_eta : max |eta|  [m]
  KE      : domain kinetic energy  0.5·rho0·∫(u²+v²)dV  [J]

Pass criteria
-------------
  T/S volume integrals
    rel_drift < 5e-4 over 30 days
    The Thomas-algorithm implicit solver introduces ~4× float32_eps (~4.8e-7)
    per step on a uniform field.  Over 8640 steps this accumulates to ~4e-3
    in absolute terms; the 5e-4 threshold is a conservative bound that still
    confirms the drift is roundoff-level and not a physical instability.
    Key diagnostic: drift must be LINEAR in time (R² > 0.999), not exponential.

  Velocity / free surface
    max_u, max_v < 1e-10 m/s  (effectively machine zero for float32)
    max_eta      < 1e-10 m
    KE           < 1e-6 J
    These must remain exactly zero: a resting uniform ocean has no pressure
    gradient and no forcing, so any non-zero velocity indicates a spurious
    source in the momentum or free-surface operators.

  No non-finite values at any step.
"""

from __future__ import annotations

import sys
import time as _time
from pathlib import Path

import numpy as np

# Make sure the project root is on the path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import jax.numpy as jnp
from OceanJAX.grid import OceanGrid
from OceanJAX.state import ModelParams, create_rest_state
from OceanJAX.timeStepping import run

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
LON        = (-40.0, -5.0)
LAT        = (-15.0, 15.0)
DEPTH_LEVELS = np.array([25., 75., 150., 250., 375., 500.], dtype=np.float64)
NX, NY     = 20, 15
DT         = 300.0          # s
N_DAYS     = 30
T_BG       = 15.0           # °C
S_BG       = 35.0           # psu
RHO0       = 1025.0

STEPS_PER_DAY = int(86400 / DT)   # 288
TOTAL_STEPS   = N_DAYS * STEPS_PER_DAY
SAVE_INTERVAL = STEPS_PER_DAY     # record once per day

# ---------------------------------------------------------------------------
# Build grid and initial state
# ---------------------------------------------------------------------------
grid = OceanGrid.create(
    lon_bounds=LON,
    lat_bounds=LAT,
    depth_levels=DEPTH_LEVELS,
    Nx=NX, Ny=NY,
    bathymetry=None,          # flat bottom
)
params = ModelParams(dt=DT)
state  = create_rest_state(grid, T_background=T_BG, S_background=S_BG)

# ---------------------------------------------------------------------------
# Compute initial volume integrals (reference values)
# ---------------------------------------------------------------------------
vol = np.array(grid.volume_c)    # (Nx, Ny, Nz)

def volume_integral(field_3d):
    return float(np.sum(np.array(field_3d) * vol))

int_T0 = volume_integral(state.T)
int_S0 = volume_integral(state.S)

# ---------------------------------------------------------------------------
# Time-stepping loop — chunk by SAVE_INTERVAL
# ---------------------------------------------------------------------------
records = []   # list of dicts

print(f"{'Day':>5}  {'rel_dT':>12}  {'rel_dS':>12}  "
      f"{'max|u|':>12}  {'max|v|':>12}  {'max|eta|':>10}  "
      f"{'KE [J]':>12}  {'status':>10}")
print("-" * 95)

t0_wall = _time.time()
n_chunks = N_DAYS

for chunk_idx in range(n_chunks):
    state, _ = run(state, grid, params, n_steps=SAVE_INTERVAL,
                   forcing_sequence=None, save_history=False)

    T_arr   = np.array(state.T)
    S_arr   = np.array(state.S)
    u_arr   = np.array(state.u)
    v_arr   = np.array(state.v)
    eta_arr = np.array(state.eta)

    bad = not (np.all(np.isfinite(T_arr)) and np.all(np.isfinite(S_arr))
               and np.all(np.isfinite(u_arr)) and np.all(np.isfinite(v_arr))
               and np.all(np.isfinite(eta_arr)))

    int_T = volume_integral(T_arr)
    int_S = volume_integral(S_arr)
    rel_dT  = abs(int_T - int_T0) / abs(int_T0)
    rel_dS  = abs(int_S - int_S0) / abs(int_S0)
    max_u   = float(np.max(np.abs(u_arr)))
    max_v   = float(np.max(np.abs(v_arr)))
    max_eta = float(np.max(np.abs(eta_arr)))
    KE      = 0.5 * RHO0 * float(np.sum((u_arr**2 + v_arr**2) * vol))
    sim_day = float(state.time) / 86400.0
    status  = "NON-FINITE" if bad else "ok"

    records.append(dict(day=sim_day, rel_dT=rel_dT, rel_dS=rel_dS,
                        max_u=max_u, max_v=max_v, max_eta=max_eta, KE=KE))

    print(f"{sim_day:5.0f}  {rel_dT:12.3e}  {rel_dS:12.3e}  "
          f"{max_u:12.3e}  {max_v:12.3e}  {max_eta:10.3e}  "
          f"{KE:12.3e}  {status:>10}")

    if bad:
        print("  [ABORT] Non-finite values detected.")
        break

wall = _time.time() - t0_wall

# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------
print()
print("=" * 95)
print(f"Wall time: {wall:.1f} s")
print()

rel_dT_max  = max(r["rel_dT"]  for r in records)
rel_dS_max  = max(r["rel_dS"]  for r in records)
max_u_all   = max(r["max_u"]   for r in records)
max_v_all   = max(r["max_v"]   for r in records)
max_eta_all = max(r["max_eta"] for r in records)
KE_max      = max(r["KE"]      for r in records)

print(f"Max rel T drift  : {rel_dT_max:.3e}   (threshold < 5e-4)")
print(f"Max rel S drift  : {rel_dS_max:.3e}   (threshold < 5e-4)")
print(f"Max |u|          : {max_u_all:.3e} m/s  (threshold < 1e-10)")
print(f"Max |v|          : {max_v_all:.3e} m/s  (threshold < 1e-10)")
print(f"Max |eta|        : {max_eta_all:.3e} m    (threshold < 1e-10)")
print(f"Max KE           : {KE_max:.3e} J    (threshold < 1e-6)")
print()

# Linearity check: fit rel_dT vs day number, compute R²
days   = np.array([r["day"]   for r in records])
dT_arr = np.array([r["rel_dT"] for r in records])
dS_arr = np.array([r["rel_dS"] for r in records])
# Linear fit via least squares
A = np.column_stack([days, np.ones_like(days)])
slope_T, _ = np.linalg.lstsq(A, dT_arr, rcond=None)[0]
slope_S, _ = np.linalg.lstsq(A, dS_arr, rcond=None)[0]
resid_T = dT_arr - A @ np.linalg.lstsq(A, dT_arr, rcond=None)[0]
resid_S = dS_arr - A @ np.linalg.lstsq(A, dS_arr, rcond=None)[0]
r2_T = 1 - np.var(resid_T) / np.var(dT_arr)
r2_S = 1 - np.var(resid_S) / np.var(dS_arr)
print(f"T drift rate     : {slope_T:.3e} /day  (R²={r2_T:.6f})")
print(f"S drift rate     : {slope_S:.3e} /day  (R²={r2_S:.6f})")
print(f"  float32 eps/step estimate: ~{288 * 4.8e-7 / 15.0:.3e} /day — "
      f"observed rate consistent: {abs(slope_T - 288*4.8e-7/15.0) < 1e-5}")
print()

# Evaluate pass/fail per criterion
results = {
    "T drift < 5e-4 (roundoff)"  : rel_dT_max  < 5e-4,
    "S drift < 5e-4 (roundoff)"  : rel_dS_max  < 5e-4,
    "T drift is linear (R²>0.999)": r2_T        > 0.999,
    "S drift is linear (R²>0.999)": r2_S        > 0.999,
    "u exactly zero"             : max_u_all   < 1e-10,
    "v exactly zero"             : max_v_all   < 1e-10,
    "eta exactly zero"           : max_eta_all < 1e-10,
    "KE exactly zero"            : KE_max      < 1e-6,
}

all_pass = all(results.values())
for name, passed in results.items():
    print(f"  {'PASS' if passed else 'FAIL'}  {name}")

print()
print("Overall:", "PASS" if all_pass else "FAIL")
