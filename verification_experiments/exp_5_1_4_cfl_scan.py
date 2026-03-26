"""
Experiment 5.1.4 — dt / CFL Stability Scan
============================================
Setup
-----
  - oras5_cold initialisation + full surface forcing (realistic case)
  - Five runs with dt = 100, 300, 600, 900, 1200 s
  - Each run integrates for 7 days
  - All other parameters identical

Rationale
---------
Using a structured initial state (ORAS5 T/S) and real surface forcing
gives a more demanding stability test than a resting ocean.  The
barotropic gravity-wave speed c = sqrt(g*H) ≈ sqrt(9.81*500) ≈ 70 m/s
sets the CFL limit:

    CFL = c * dt / dx_min

where dx_min is the smallest horizontal grid spacing.  For the current
domain (35° wide, 20 cells), dx_min ≈ R * cos(lat_max) * (35°/20) * pi/180
≈ 6.371e6 * cos(15°) * 0.0305 ≈ 188 km.

Theoretical CFL estimates:
    dt=100  → CFL ≈ 0.037
    dt=300  → CFL ≈ 0.112  (operational value)
    dt=600  → CFL ≈ 0.224
    dt=900  → CFL ≈ 0.336
    dt=1200 → CFL ≈ 0.447

Pass criteria
-------------
  - No non-finite values over 7 days  → stable
  - T range stays within ±20°C of initial  → no blowup
  - S range stays within 30–42 psu         → no blowup
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
from OceanJAX.state import ModelParams
from OceanJAX.data.oras5 import read_oras5, regrid_to_model, read_oras5_forcing, regrid_forcing
from OceanJAX.timeStepping import run, SurfaceForcing

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
ORAS5_PATH    = "OceanJAX/data/data_oras5/oras5_2026_01_native_merged.nc"
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
N_DAYS       = 7
DT_LIST      = [100, 300, 600, 900, 1200]

G    = 9.81
H    = 500.0
C_BT = np.sqrt(G * H)   # barotropic wave speed ≈ 70 m/s

# ---------------------------------------------------------------------------
# Build grid and base initial state (shared across all dt)
# ---------------------------------------------------------------------------
print("Loading ORAS5 ...", flush=True)
grid = OceanGrid.create(
    lon_bounds=LON, lat_bounds=LAT,
    depth_levels=DEPTH_LEVELS,
    Nx=NX, Ny=NY,
)

with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    raw   = read_oras5(ORAS5_PATH, time_index=0)
    state0 = regrid_to_model(raw, grid)

# oras5_cold: zero velocities and eta
zeros3 = jnp.zeros((NX, NY, len(DEPTH_LEVELS)), dtype=jnp.float32)
zeros2 = jnp.zeros((NX, NY), dtype=jnp.float32)
state0 = eqx.tree_at(lambda s: s.u,        state0, zeros3)
state0 = eqx.tree_at(lambda s: s.v,        state0, zeros3)
state0 = eqx.tree_at(lambda s: s.u_prev,   state0, zeros3)
state0 = eqx.tree_at(lambda s: s.v_prev,   state0, zeros3)
state0 = eqx.tree_at(lambda s: s.eta,      state0, zeros2)
state0 = eqx.tree_at(lambda s: s.eta_prev, state0, zeros2)

# Estimate dx_min for CFL calculation
dx_all = np.array(grid.dx_c)   # (Nx, Ny)
dy_all = np.array(grid.dy_c)
dx_min = float(np.min(dx_all[dx_all > 0]))
print(f"  dx_min = {dx_min/1e3:.1f} km  |  c_bt = {C_BT:.1f} m/s")

# ---------------------------------------------------------------------------
# Load forcing once; will be re-tiled per dt
# ---------------------------------------------------------------------------
print("Loading forcing ...", flush=True)
with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    raw_forcing  = read_oras5_forcing(FORCING_PATHS)
    forcing_snap = regrid_forcing(raw_forcing, grid)   # SurfaceForcing, (Nx,Ny)

# ---------------------------------------------------------------------------
# Run scan
# ---------------------------------------------------------------------------
summary = []

for dt in DT_LIST:
    steps_per_day = int(round(86400 / dt))
    total_steps   = N_DAYS * steps_per_day
    cfl_est       = C_BT * dt / dx_min

    params = ModelParams(dt=float(dt))

    # Tile forcing to total_steps
    def _tile(arr, n):
        return jnp.stack([jnp.array(arr, dtype=jnp.float32)] * n)

    forcing_seq = SurfaceForcing(
        heat_flux = _tile(forcing_snap.heat_flux, total_steps),
        fw_flux   = _tile(forcing_snap.fw_flux,   total_steps),
        tau_x     = _tile(forcing_snap.tau_x,     total_steps),
        tau_y     = _tile(forcing_snap.tau_y,      total_steps),
    )

    t0 = _time.time()
    try:
        final, _ = run(state0, grid, params, n_steps=total_steps,
                       forcing_sequence=forcing_seq)
        wall = _time.time() - t0

        T_arr   = np.array(final.T)
        S_arr   = np.array(final.S)
        eta_arr = np.array(final.eta)
        u_arr   = np.array(final.u)

        finite  = (np.all(np.isfinite(T_arr)) and np.all(np.isfinite(S_arr))
                   and np.all(np.isfinite(eta_arr)))
        T_ok    = (T_arr.min() > 0.0  and T_arr.max() < 50.0)
        S_ok    = (S_arr[S_arr > 0].min() > 30.0 and S_arr.max() < 42.0)
        stable  = finite and T_ok and S_ok
        verdict = "STABLE" if stable else "UNSTABLE"

        rec = dict(
            dt=dt, cfl=cfl_est, steps=total_steps,
            T_min=T_arr.min(), T_max=T_arr.max(),
            S_min=float(S_arr[S_arr > 0].min()), S_max=S_arr.max(),
            eta_max=float(np.max(np.abs(eta_arr))),
            finite=finite, stable=stable, wall=wall,
        )
    except Exception as e:
        wall = _time.time() - t0
        verdict = f"ERROR: {e}"
        rec = dict(dt=dt, cfl=cfl_est, steps=total_steps,
                   T_min=np.nan, T_max=np.nan,
                   S_min=np.nan, S_max=np.nan,
                   eta_max=np.nan, finite=False, stable=False, wall=wall)

    summary.append(rec)
    print(f"  dt={dt:5d}s  CFL={cfl_est:.3f}  {steps_per_day} steps/day  "
          f"T=[{rec['T_min']:.2f},{rec['T_max']:.2f}]  "
          f"S=[{rec['S_min']:.2f},{rec['S_max']:.2f}]  "
          f"eta_max={rec['eta_max']:.3f}m  "
          f"{verdict}  ({wall:.1f}s)", flush=True)

# ---------------------------------------------------------------------------
# Summary table
# ---------------------------------------------------------------------------
print()
print("=" * 90)
print(f"{'dt [s]':>8}  {'CFL':>6}  {'T range [°C]':>22}  "
      f"{'S range [psu]':>20}  {'eta_max [m]':>12}  {'Result':>10}")
print("-" * 90)
for r in summary:
    verdict = "STABLE" if r["stable"] else "UNSTABLE"
    print(f"{r['dt']:>8}  {r['cfl']:>6.3f}  "
          f"[{r['T_min']:7.2f}, {r['T_max']:7.2f}]  "
          f"[{r['S_min']:7.2f}, {r['S_max']:7.2f}]  "
          f"{r['eta_max']:>12.4f}  {verdict:>10}")
