"""
Experiment 5.1.3 — NullClosure Zero Side-Effect Verification
=============================================================
Setup
-----
  - oras5_cold initialisation (T/S from ORAS5, u=v=eta=0)
  - Full surface forcing (all 4 ORAS5 forcing files)
  - Two runs, bit-for-bit identical except for the closure argument:
      Run A : closure=None      (pure physics, no ML hook)
      Run B : closure=NullClosure()  (ML hook active but no-op)
  - Integration: 1 day  (dt = 300 s → 288 steps)

Rationale
---------
NullClosure returns zero dT_tend, zero dS_tend, and kappa_v_scale=1.
If the ML hook is correctly implemented, Run A and Run B must produce
bit-identical results.  Any difference indicates either a floating-point
branch inconsistency or an unintended side-effect in the hook code path.

Metrics
-------
  max |T_A - T_B|   over all wet cells
  max |S_A - S_B|
  max |eta_A - eta_B|
  max |u_A - u_B|
  max |v_A - v_B|

Pass criteria
-------------
  All differences == 0.0  (strict bit-identical, not just small)
"""

from __future__ import annotations

import sys
import warnings
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from OceanJAX.grid import OceanGrid
from OceanJAX.state import ModelParams
from OceanJAX.data.oras5 import read_oras5, regrid_to_model, read_oras5_forcing, regrid_forcing
from OceanJAX.timeStepping import run, SurfaceForcing
from OceanJAX.ml.closure import NullClosure

import jax.numpy as jnp

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
ORAS5_PATH   = "OceanJAX/data/data_oras5/oras5_2026_01_native_merged.nc"
FORCING_PATHS = [
    "OceanJAX/data/data_oras5/sohefldo_control_monthly_highres_2D_202601_OPER_v0.1.nc",
    "OceanJAX/data/data_oras5/sowaflup_control_monthly_highres_2D_202601_OPER_v0.1.nc",
    "OceanJAX/data/data_oras5/sozotaux_control_monthly_highres_2D_202601_OPER_v0.1.nc",
    "OceanJAX/data/data_oras5/sometauy_control_monthly_highres_2D_202601_OPER_v0.1.nc",
]
LON          = (-40.0, -5.0)
LAT          = (-15.0, 15.0)
DEPTH_LEVELS = np.array([25., 75., 150., 250., 375., 500.], dtype=np.float64)
NX, NY       = 20, 15
DT           = 300.0
N_STEPS      = 288   # 1 day

# ---------------------------------------------------------------------------
# Build grid and initial state
# ---------------------------------------------------------------------------
print("Loading ORAS5 state ...", flush=True)
grid = OceanGrid.create(
    lon_bounds=LON, lat_bounds=LAT,
    depth_levels=DEPTH_LEVELS,
    Nx=NX, Ny=NY,
)
params = ModelParams(dt=DT)

with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    raw   = read_oras5(ORAS5_PATH, time_index=0)
    state = regrid_to_model(raw, grid)

# Zero out velocities and eta (oras5_cold mode)
import equinox as eqx
zeros3 = jnp.zeros((NX, NY, len(DEPTH_LEVELS)), dtype=jnp.float32)
zeros2 = jnp.zeros((NX, NY), dtype=jnp.float32)
state = eqx.tree_at(lambda s: s.u,        state, zeros3)
state = eqx.tree_at(lambda s: s.v,        state, zeros3)
state = eqx.tree_at(lambda s: s.u_prev,   state, zeros3)
state = eqx.tree_at(lambda s: s.v_prev,   state, zeros3)
state = eqx.tree_at(lambda s: s.eta,      state, zeros2)
state = eqx.tree_at(lambda s: s.eta_prev, state, zeros2)

# ---------------------------------------------------------------------------
# Build forcing sequence (288 steps, constant in time)
# ---------------------------------------------------------------------------
print("Loading surface forcing ...", flush=True)
with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    raw_forcing = read_oras5_forcing(FORCING_PATHS)
    forcing_2d  = regrid_forcing(raw_forcing, grid)

# Broadcast single snapshot to N_STEPS time axis
def _tile(arr):
    return jnp.stack([jnp.array(arr, dtype=jnp.float32)] * N_STEPS)

forcing_seq = SurfaceForcing(
    heat_flux = _tile(forcing_2d.heat_flux),
    fw_flux   = _tile(forcing_2d.fw_flux),
    tau_x     = _tile(forcing_2d.tau_x),
    tau_y     = _tile(forcing_2d.tau_y),
)

# ---------------------------------------------------------------------------
# Run A: closure=None
# ---------------------------------------------------------------------------
print("Run A: closure=None ...", flush=True)
state_A, _ = run(state, grid, params, n_steps=N_STEPS,
                 forcing_sequence=forcing_seq, closure=None)

# ---------------------------------------------------------------------------
# Run B: closure=NullClosure()
# ---------------------------------------------------------------------------
print("Run B: closure=NullClosure() ...", flush=True)
state_B, _ = run(state, grid, params, n_steps=N_STEPS,
                 forcing_sequence=forcing_seq, closure=NullClosure())

# ---------------------------------------------------------------------------
# Compare
# ---------------------------------------------------------------------------
print()
fields = [
    ("T",   state_A.T,   state_B.T),
    ("S",   state_A.S,   state_B.S),
    ("eta", state_A.eta, state_B.eta),
    ("u",   state_A.u,   state_B.u),
    ("v",   state_A.v,   state_B.v),
]

results = {}
print(f"{'Field':>6}  {'max |A-B|':>14}  {'bit-identical':>14}")
print("-" * 42)
for name, fa, fb in fields:
    diff = float(np.max(np.abs(np.array(fa) - np.array(fb))))
    identical = (diff == 0.0)
    results[name] = identical
    print(f"{name:>6}  {diff:14.3e}  {'YES' if identical else 'NO':>14}")

print()
all_pass = all(results.values())
for name, passed in results.items():
    print(f"  {'PASS' if passed else 'FAIL'}  {name} bit-identical")

print()
print("Overall:", "PASS" if all_pass else "FAIL")
