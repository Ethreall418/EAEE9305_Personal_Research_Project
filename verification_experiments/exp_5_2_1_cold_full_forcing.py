"""
Experiment 5.2.1 — ORAS5 Cold Start, Full Forcing (30 days)
============================================================
Setup
-----
  - Domain: tropical Atlantic  lon=(-40,-5)  lat=(-15,15)  depth=500 m
  - Grid: 20×15×10,  dt=300 s
  - Init (oras5_cold): T/S from ORAS5 Jan-2026, u=v=eta=0
  - Forcing: full ORAS5 surface fluxes
      heat_flux  from sohefldo_*  [W m-2]
      fw_flux    from sowaflup_*  [kg m-2 s-1 → m s-1]
      tau_x      from sozotaux_*  [N m-2]
      tau_y      from sometauy_*  [N m-2]

Rationale
---------
Baseline integration to verify model produces physically consistent
evolution under realistic ORAS5 forcing.  Expected signal:
  - SST cooling of ~1–3 °C in 30 days (net negative heat flux in Jan)
  - Salinity broadly conserved (E-P forcing is weak in tropics)
  - eta amplitude O(0.1–1 m) from wind-driven circulation

Metrics recorded every day
--------------------------
  T_min, T_max  : wet-cell temperature extremes [°C]
  S_min, S_max  : wet-cell salinity extremes [psu]
  eta_min, eta_max : free-surface extremes [m]
  SST_mean      : domain-mean surface-layer temperature [°C]

Pass criteria
-------------
  No non-finite values throughout
  T_wet ∈ [5, 35] °C
  S_wet ∈ [30, 40] psu
  |eta|_max < 2 m
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
from OceanJAX.data.oras5 import read_oras5, regrid_to_model, read_oras5_forcing, regrid_forcing
from OceanJAX.timeStepping import SurfaceForcing, run

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
ORAS5_IC   = "OceanJAX/data/data_oras5/oras5_2026_01_native_merged.nc"
FORCING_FILES = [
    "OceanJAX/data/data_oras5/sohefldo_control_monthly_highres_2D_202601_OPER_v0.1.nc",
    "OceanJAX/data/data_oras5/sowaflup_control_monthly_highres_2D_202601_OPER_v0.1.nc",
    "OceanJAX/data/data_oras5/sozotaux_control_monthly_highres_2D_202601_OPER_v0.1.nc",
    "OceanJAX/data/data_oras5/sometauy_control_monthly_highres_2D_202601_OPER_v0.1.nc",
]

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
# Initial state: oras5_cold (T/S from ORAS5, u=v=eta=0)
# ---------------------------------------------------------------------------
print("Loading ORAS5 initial conditions ...", flush=True)
with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    raw = read_oras5(ORAS5_IC, time_index=0)
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

# ---------------------------------------------------------------------------
# Forcing: full ORAS5 surface fluxes (constant in time for 30 days)
# ---------------------------------------------------------------------------
print("Loading ORAS5 surface forcing ...", flush=True)
with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    raw_f = read_oras5_forcing(FORCING_FILES, time_index=0)
    sf    = regrid_forcing(raw_f, grid,
                           use_fields={"heat_flux", "fw_flux", "tau_x", "tau_y"})

hf_np = np.array(sf.heat_flux)
fw_np = np.array(sf.fw_flux)
tx_np = np.array(sf.tau_x)
ty_np = np.array(sf.tau_y)

print(f"  heat_flux=[{hf_np.min():.1f}, {hf_np.max():.1f}] W/m²  "
      f"mean={hf_np[mask_np[:,:,0]>0].mean():.2f}")
print(f"  tau_x    =[{tx_np.min():.3f}, {tx_np.max():.3f}] N/m²")
print(f"  tau_y    =[{ty_np.min():.3f}, {ty_np.max():.3f}] N/m²")


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
# Integration loop
# ---------------------------------------------------------------------------
print()
print(f"{'Day':>5}  {'T_min':>7} {'T_max':>7}  {'S_min':>6} {'S_max':>6}  "
      f"{'eta_min':>8} {'eta_max':>8}  {'SST_mean':>9}  {'status':>10}")
print("-" * 90)

records = []
t0_wall = _time.time()

for day in range(1, N_DAYS + 1):
    forcing = _make_forcing(STEPS_PER_DAY)
    state, _ = run_jit(state, grid, params, n_steps=STEPS_PER_DAY,
                       forcing_sequence=forcing, save_history=False)

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

SST_final  = records[-1]["sst_mean"]
T_min_all  = min(r["T_min"]  for r in records)
T_max_all  = max(r["T_max"]  for r in records)
S_min_all  = min(r["S_min"]  for r in records)
S_max_all  = max(r["S_max"]  for r in records)
eta_abs_max = max(max(abs(r["eta_min"]), abs(r["eta_max"])) for r in records)

print(f"SST:  initial={SST0:.3f} C  ->  final={SST_final:.3f} C  "
      f"(dSST={SST_final-SST0:+.3f} C)")
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
}

for name, passed in checks.items():
    print(f"  {'PASS' if passed else 'FAIL'}  {name}")

all_pass = all(checks.values())
print()
print("Overall:", "PASS" if all_pass else "FAIL")
