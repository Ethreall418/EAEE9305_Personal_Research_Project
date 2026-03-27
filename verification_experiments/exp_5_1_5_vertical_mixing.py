"""
Experiment 5.1.5 — Vertical Mixing Parameterisation
=====================================================
Setup
-----
  - Ideal flat-bottom domain: 5×5×6 cells, tropical Atlantic
  - Depth levels: [25, 75, 150, 250, 375, 500] m
  - Initial T: 22 °C for z ≤ 150 m (upper 3 layers),
               10 °C for z > 150 m (lower 3 layers)  → step thermocline
  - S = 35 psu uniform, u = v = eta = 0
  - No surface forcing

Three groups
------------
  A — kappa_v = 1e-4 m²/s  (baseline, closure=None)
  B — kappa_v = 1e-2 m²/s  (KappaScaleClosure(scale=100), baseline×100)
  C — kappa_v = 0   m²/s   (KappaScaleClosure(scale=0),   no diffusion)

Physics
-------
  With a step thermocline of half-width H ~ 250 m:
    τ_B = H² / (π²·κ_B) ≈ 250²/(π²·0.01) ≈ 6.3×10⁵ s ≈  7 days
    τ_A = H² / (π²·κ_A) ≈ 250²/(π²·1e-4) ≈ 6.3×10⁷ s ≈ 730 days
  After 30 days:
    Group B should show significant thermocline smoothing.
    Group A should show negligible change.
    Group C should show no change (only float32 Thomas roundoff).

Metrics recorded every day
--------------------------
  std_T   : domain-mean std(T over z)  [°C]   — thermocline sharpness
  T_surf  : domain-mean surface layer temperature [°C]
  T_bot   : domain-mean bottom layer temperature  [°C]
  T_mid   : domain-mean mid-layer (k=2, 150 m) temperature [°C]

Pass criteria
-------------
  Group B: std_T ratio (final/initial) < 0.70   (>30% reduction)
  Group A: std_T ratio (final/initial) > 0.95   (<5% reduction)
  Group C: max |T_change| over all cells < 5e-3 °C
  All:     no non-finite values
"""

from __future__ import annotations

import sys
import time as _time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import jax.numpy as jnp
import equinox as eqx
from OceanJAX.grid import OceanGrid
from OceanJAX.state import ModelParams, OceanState, create_from_arrays
from OceanJAX.timeStepping import run
from OceanJAX.ml.closure import AbstractClosure, ClosureOutput

# ---------------------------------------------------------------------------
# KappaScaleClosure — scales the background kappa_v by a constant factor
# ---------------------------------------------------------------------------

class KappaScaleClosure(AbstractClosure):
    """Multiplies params.kappa_v by `scale`; adds zero tracer tendency."""
    scale: float

    def __call__(self, state, grid, params) -> ClosureOutput:
        zeros = jnp.zeros((grid.Nx, grid.Ny, grid.Nz), dtype=jnp.float32)
        return ClosureOutput(
            dT_tend       = zeros,
            dS_tend       = zeros,
            kappa_v_scale = jnp.array(self.scale, dtype=jnp.float32),
        )


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
LON          = (-30.0, -25.0)
LAT          = (0.0, 5.0)
DEPTH_LEVELS = np.array([25., 75., 150., 250., 375., 500.], dtype=np.float64)
NX, NY, NZ   = 5, 5, len(DEPTH_LEVELS)
DT           = 300.0
N_DAYS       = 30
KAPPA_BASE   = 1e-4       # m²/s — background diffusivity for all groups

STEPS_PER_DAY = int(86400 / DT)   # 288
TOTAL_STEPS   = N_DAYS * STEPS_PER_DAY
SAVE_INTERVAL = STEPS_PER_DAY

# ---------------------------------------------------------------------------
# Build grid (flat bottom, fully ocean)
# ---------------------------------------------------------------------------
grid = OceanGrid.create(
    lon_bounds=LON, lat_bounds=LAT,
    depth_levels=DEPTH_LEVELS,
    Nx=NX, Ny=NY,
    bathymetry=None,   # flat bottom at 500 m
)

# ---------------------------------------------------------------------------
# Initial condition: step thermocline at 150 m
# ---------------------------------------------------------------------------
T_UPPER = 22.0   # °C  (z <= 150 m, levels k=0,1,2)
T_LOWER = 10.0   # °C  (z >  150 m, levels k=3,4,5)
S_CONST = 35.0   # psu (uniform)

T_init_np = np.zeros((NX, NY, NZ), dtype=np.float32)
for k, z_c in enumerate(DEPTH_LEVELS):
    T_init_np[:, :, k] = T_UPPER if z_c <= 150.0 else T_LOWER

S_init_np   = np.full((NX, NY, NZ), S_CONST, dtype=np.float32)
u_init_np   = np.zeros((NX, NY, NZ), dtype=np.float32)
v_init_np   = np.zeros((NX, NY, NZ), dtype=np.float32)
eta_init_np = np.zeros((NX, NY),    dtype=np.float32)

mask_np = np.array(grid.mask_c)   # all ones for flat-bottom ocean

# Initial vertical std (domain mean)
def domain_mean_T_profile(T_arr: np.ndarray) -> np.ndarray:
    """Returns mean T at each depth level (Nz,)."""
    return T_arr.mean(axis=(0, 1))   # (Nz,)

T_init_profile = domain_mean_T_profile(T_init_np)
std_T_init     = float(np.std(T_init_profile))

# ---------------------------------------------------------------------------
# Experiment runner
# ---------------------------------------------------------------------------

def run_group(label: str, closure) -> dict:
    """
    Run a 30-day integration, return a dict with the daily diagnostic records
    and pass/fail for each criterion.
    """
    state  = create_from_arrays(grid, u_init_np, v_init_np,
                                T_init_np, S_init_np, eta_init_np)
    params = ModelParams(dt=DT, kappa_v=KAPPA_BASE)

    records = []
    bad     = False

    print()
    print(f"  Group {label}  (closure={closure!r})")
    print(f"  {'Day':>5}  {'std_T':>10}  {'T_surf':>8}  "
          f"{'T_mid':>8}  {'T_bot':>8}  {'status':>10}")
    print("  " + "-" * 60)

    for day in range(1, N_DAYS + 1):
        state, _ = run(state, grid, params, n_steps=SAVE_INTERVAL,
                       forcing_sequence=None, save_history=False,
                       closure=closure)

        T_arr = np.array(state.T)

        if not np.all(np.isfinite(T_arr)):
            bad = True
            print(f"  {day:5d}  [NON-FINITE — ABORT]")
            break

        profile = domain_mean_T_profile(T_arr)
        std_T   = float(np.std(profile))
        T_surf  = float(profile[0])
        T_mid   = float(profile[2])   # 150 m
        T_bot   = float(profile[-1])

        records.append(dict(day=day, std_T=std_T,
                            T_surf=T_surf, T_mid=T_mid, T_bot=T_bot))
        print(f"  {day:5d}  {std_T:10.4f}  {T_surf:8.4f}  "
              f"{T_mid:8.4f}  {T_bot:8.4f}  {'ok':>10}")

    T_final   = np.array(state.T)
    max_change = float(np.max(np.abs(T_final - T_init_np)[mask_np > 0]))

    std_T_final = records[-1]["std_T"] if records else np.nan
    std_ratio   = std_T_final / std_T_init if std_T_init > 0 else np.nan

    return dict(label=label, records=records, bad=bad,
                std_T_init=std_T_init, std_T_final=std_T_final,
                std_ratio=std_ratio, max_change=max_change)


# ---------------------------------------------------------------------------
# Run all three groups
# ---------------------------------------------------------------------------
print("=" * 68)
print("Experiment 5.1.5 — Vertical Mixing Parameterisation")
print(f"Grid: {NX}×{NY}×{NZ},  dt={DT}s,  {N_DAYS} days")
print(f"Initial T profile: {T_init_profile.tolist()}")
print(f"Initial std(T over z): {std_T_init:.4f} °C")
print("=" * 68)

t0 = _time.time()

results = {}
results["A"] = run_group("A", closure=None)
results["B"] = run_group("B", closure=KappaScaleClosure(scale=100.0))
results["C"] = run_group("C", closure=KappaScaleClosure(scale=0.0))

wall = _time.time() - t0

# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------
print()
print("=" * 68)
print(f"Wall time: {wall:.1f} s")
print()
print(f"  {'Group':<8}  {'kappa_v [m²/s]':<16}  "
      f"{'std_ratio':>10}  {'max_change [°C]':>16}")
print("  " + "-" * 56)

kappa_labels = {"A": "1e-4 (base)", "B": "1e-2 (×100)", "C": "0 (×0)"}
for k, res in results.items():
    print(f"  {k:<8}  {kappa_labels[k]:<16}  "
          f"{res['std_ratio']:10.4f}  {res['max_change']:16.3e}")

print()

# Pass / fail evaluation
checks = {}

checks["A: std_ratio > 0.95 (minimal mixing at kappa=1e-4)"] = (
    not results["A"]["bad"] and results["A"]["std_ratio"] > 0.95
)
checks["B: std_ratio < 0.70 (>30% mixing at kappa=1e-2)"] = (
    not results["B"]["bad"] and results["B"]["std_ratio"] < 0.70
)
checks["C: max_change < 5e-3 °C (no diffusion at kappa=0)"] = (
    not results["C"]["bad"] and results["C"]["max_change"] < 5e-3
)
checks["B mixes more than A (std_ratio_B < std_ratio_A)"] = (
    results["B"]["std_ratio"] < results["A"]["std_ratio"]
)
checks["A mixes more than C (std_ratio_A < std_ratio_C + 0.01)"] = (
    results["A"]["std_ratio"] < results["C"]["std_ratio"] + 0.01
)
checks["no non-finite values in any group"] = (
    not any(r["bad"] for r in results.values())
)

for name, passed in checks.items():
    print(f"  {'PASS' if passed else 'FAIL'}  {name}")

all_pass = all(checks.values())
print()
print("Overall:", "PASS" if all_pass else "FAIL")
