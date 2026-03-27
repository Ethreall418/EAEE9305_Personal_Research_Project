# OceanJAX

A fully differentiable 3-D ocean model built in JAX, implementing Boussinesq hydrostatic primitive equations on an Arakawa C-grid in spherical coordinates. Designed for ML-based parameterization and adjoint-based optimization.

---

## Table of Contents

1. [Requirements](#1-requirements)
2. [Installation](#2-installation)
3. [Project Structure](#3-project-structure)
4. [Quick Start](#4-quick-start)
5. [Step-by-Step Usage Guide](#5-step-by-step-usage-guide)
   - 5.1 [Build a Grid](#51-build-a-grid)
   - 5.2 [Initialize Model State](#52-initialize-model-state)
   - 5.3 [Set Model Parameters](#53-set-model-parameters)
   - 5.4 [Prepare Surface Forcing](#54-prepare-surface-forcing)
   - 5.5 [Run the Model](#55-run-the-model)
   - 5.6 [Save Output](#56-save-output)
6. [Using ORAS5 Reanalysis Data](#6-using-oras5-reanalysis-data)
7. [Time-Varying Forcing](#7-time-varying-forcing)
8. [ML Closure Interface](#8-ml-closure-interface)
9. [Ensemble / Multi-GPU Runs](#9-ensemble--multi-gpu-runs)
10. [Running Verification Tests](#10-running-verification-tests)
11. [Stability Guidelines](#11-stability-guidelines)
12. [ModelParams Reference](#12-modelparams-reference)

---

## 1. Requirements

| Package | Minimum version |
|---|---|
| Python | 3.10 |
| jax | 0.4.25 |
| jaxlib | 0.4.25 |
| equinox | 0.11.0 |
| numpy | 1.26 |
| xarray | 2024.1 |
| netCDF4 | 1.6 |
| scipy | 1.12 |
| pytest | 8.0 |

**GPU (optional):** CUDA 12 + compatible `jax[cuda12]` build. The model runs correctly on CPU; GPU is recommended for grids larger than ~30×30×10 or runs longer than a few days.

---

## 2. Installation

```bash
# 1. Clone the repository
git clone <repo-url>
cd EAEE9280_Personal_Research_Project

# 2. Create and activate a virtual environment
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. (GPU users only) Install CUDA-enabled JAX
pip install --upgrade "jax[cuda12]" \
    -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html

# 5. Verify installation
python -m pytest OceanJAX/tests/ -q
# Expected: 110+ passed, 1 skipped
```

---

## 3. Project Structure

```
OceanJAX/
├── grid.py                 # OceanGrid — Arakawa C-grid, masks, metrics
├── state.py                # OceanState, ModelParams, factory functions
├── operators.py            # grad_x, grad_y, div_h, laplacian
├── timeStepping.py         # step(), run(), SurfaceForcing
├── Physics/
│   ├── dynamics.py         # EOS, hydrostatic pressure, PGF, Coriolis, free surface
│   ├── tracers.py          # tracer advection/diffusion, surface T/S forcing
│   └── mixing.py           # implicit vertical mixing/viscosity, Ri-based diffusivity
├── data/
│   ├── oras5.py            # ORAS5 loader: read_oras5, regrid_to_model,
│   │                       #   read_oras5_forcing, regrid_forcing
│   └── forcing.py          # make_synthetic_forcing, make_forcing_sequence
├── ml/
│   └── closure.py          # AbstractClosure, NullClosure, ClosureOutput
├── parallel/
│   └── ensemble.py         # batch_run, sharded_ensemble_run
└── tests/                  # pytest test suite

experiment.py               # Full experiment driver (NetCDF output, diagnostics)
plot_output.py              # Standard figures from NetCDF output
validate_oras5.py           # ORAS5 file validation utility
verification_experiments/   # Thesis Chapter 5 verification scripts
```

---

## 4. Quick Start

The following runs a 30-day integration on a small idealised domain in under one minute.

```python
import numpy as np
import jax
import jax.numpy as jnp

from OceanJAX.grid import OceanGrid
from OceanJAX.state import ModelParams, create_rest_state
from OceanJAX.timeStepping import run
from OceanJAX.data.forcing import make_synthetic_forcing

# 1. Grid
grid = OceanGrid.create(
    lon_bounds   = (-20.0, 20.0),
    lat_bounds   = (-10.0, 10.0),
    depth_levels = np.linspace(25, 475, 10),
    Nx=8, Ny=6,
)

# 2. Initial state (uniform T=20 C, S=35 psu, no motion)
state  = create_rest_state(grid, T_background=20.0, S_background=35.0)
params = ModelParams(dt=300.0)

# 3. Surface forcing: seasonal heat flux + constant wind stress
N_STEPS = 30 * 288   # 30 days at dt=300 s
forcing = make_synthetic_forcing(
    grid, N_STEPS, dt=300.0,
    heat_flux = {"mean": -50.0, "amplitude": 100.0, "period": 365 * 86400},
    tau_x     = -0.05,
)

# 4. Run (first call compiles ~20-60 s; subsequent calls are fast)
run_jit  = jax.jit(run, static_argnames=("n_steps", "save_history"))
final, _ = run_jit(state, grid, params, n_steps=N_STEPS,
                   forcing_sequence=forcing, save_history=False)

# 5. Inspect results
T_surf = np.array(final.T[:, :, 0])
print(f"Final SST range: [{T_surf.min():.2f}, {T_surf.max():.2f}] C")
```

---

## 5. Step-by-Step Usage Guide

### 5.1 Build a Grid

```python
from OceanJAX.grid import OceanGrid
import numpy as np

grid = OceanGrid.create(
    lon_bounds   = (-40.0, -5.0),    # (lon_min, lon_max) degrees east
    lat_bounds   = (-15.0, 15.0),    # (lat_min, lat_max) degrees north
    depth_levels = np.array([25., 75., 150., 250., 375., 500.]),  # cell centres [m]
    Nx           = 20,               # cells in x (longitude)
    Ny           = 15,               # cells in y (latitude)
    bathymetry   = None,             # optional (Nx, Ny) float64 array of water depths [m]
                                     # None -> flat bottom at max(depth_levels) + dz/2
)
```

**Key grid attributes:**

| Attribute | Shape | Description |
|---|---|---|
| `grid.Nx, Ny, Nz` | scalar | Grid dimensions |
| `grid.lon_c, lat_c, z_c` | 1-D | Cell-centre coordinates |
| `grid.mask_c` | `(Nx, Ny, Nz)` | 1 = ocean cell, 0 = land or below bathymetry |
| `grid.volume_c` | `(Nx, Ny, Nz)` | Cell volumes [m^3] |
| `grid.H` | `(Nx, Ny)` | Water depth per column [m] |

**Recommended grid sizes:**

| Application | Nx x Ny x Nz | Recommended dt |
|---|---|---|
| Testing / verification | 5-10 x 5-8 x 6 | 300 s |
| Regional experiment | 20 x 15 x 10 | 300 s |
| Basin-scale | 50 x 40 x 10 | 300 s |

---

### 5.2 Initialize Model State

```python
from OceanJAX.state import create_rest_state, create_from_arrays
import jax.numpy as jnp

# Option A — Idealised uniform state (no data needed)
state = create_rest_state(grid, T_background=15.0, S_background=35.0)

# Option B — Custom arrays
T_init   = ...   # (Nx, Ny, Nz) float32  [C]
S_init   = ...   # (Nx, Ny, Nz) float32  [psu]
u_init   = jnp.zeros((grid.Nx, grid.Ny, grid.Nz), dtype=jnp.float32)
v_init   = jnp.zeros_like(u_init)
eta_init = jnp.zeros((grid.Nx, grid.Ny), dtype=jnp.float32)

state = create_from_arrays(grid, u=u_init, v=v_init,
                            T=T_init, S=S_init, eta=eta_init)

# Option C — ORAS5 cold start (see Section 6)
```

**OceanState fields:**

| Field | Shape | Description |
|---|---|---|
| `state.T` | `(Nx, Ny, Nz)` | Temperature [C] |
| `state.S` | `(Nx, Ny, Nz)` | Salinity [psu] |
| `state.u`, `state.v` | `(Nx, Ny, Nz)` | Horizontal velocities [m/s] |
| `state.w` | `(Nx, Ny, Nz+1)` | Vertical velocity [m/s] |
| `state.eta` | `(Nx, Ny)` | Free surface height [m] |
| `state.time` | scalar | Elapsed model time [s] |
| `state.step_count` | scalar | Number of steps taken |

---

### 5.3 Set Model Parameters

```python
from OceanJAX.state import ModelParams

params = ModelParams(
    dt      = 300.0,    # time step [s]  — keep <= 900 s (see Section 11)
    kappa_h = 100.0,    # horizontal tracer diffusivity [m^2/s]
    kappa_v = 1e-5,     # vertical tracer diffusivity [m^2/s]
    nu_h    = 200.0,    # horizontal viscosity [m^2/s]
    nu_v    = 1e-4,     # vertical viscosity [m^2/s]
    # All other parameters have physically validated defaults (see Section 12).
)
```

---

### 5.4 Prepare Surface Forcing

For a multi-step run, each forcing field must have shape `(n_steps, Nx, Ny)` so that every time step can receive a different value.

**Option A — No forcing**
```python
forcing = None
```

**Option B — Constant forcing**
```python
from OceanJAX.timeStepping import SurfaceForcing
import jax.numpy as jnp, numpy as np

hf   = np.full((grid.Nx, grid.Ny), -80.0, dtype=np.float32)  # W/m^2
ones = np.ones((N_STEPS, 1, 1), dtype=np.float32)

forcing = SurfaceForcing(
    heat_flux = jnp.asarray(hf * ones),
    fw_flux   = jnp.zeros((N_STEPS, grid.Nx, grid.Ny), dtype=jnp.float32),
    tau_x     = jnp.asarray(np.full((N_STEPS, grid.Nx, grid.Ny), -0.05, dtype=np.float32)),
    tau_y     = jnp.zeros((N_STEPS, grid.Nx, grid.Ny), dtype=jnp.float32),
)
```

**Option C — Synthetic time-varying forcing**
```python
from OceanJAX.data.forcing import make_synthetic_forcing

forcing = make_synthetic_forcing(
    grid, N_STEPS, dt=300.0,
    heat_flux = {"mean": -50.0, "amplitude": 150.0, "period": 365*86400},
    tau_x     = -0.05,         # constant scalar
    tau_y     = wind_pattern,  # constant (Nx, Ny) array
    fw_flux   = 0.0,
)
```

**Option D — Interpolated from real snapshots**
```python
from OceanJAX.data.forcing import make_forcing_sequence

snapshots = [
    (0.0,        sf_january),
    (30*86400,   sf_february),
    # ...
]
forcing = make_forcing_sequence(snapshots, N_STEPS, dt=300.0,
                                interp="cyclic")
```

**Sign conventions for SurfaceForcing:**

| Field | Unit | Positive = |
|---|---|---|
| `heat_flux` | W m^-2 | Ocean gains heat |
| `fw_flux` | m s^-1 | Net evaporation (salinity increases) |
| `tau_x` | N m^-2 | Eastward wind stress |
| `tau_y` | N m^-2 | Northward wind stress |

---

### 5.5 Run the Model

```python
import jax
from OceanJAX.timeStepping import run

# JIT-compile once; reuse for all calls with the same n_steps
run_jit = jax.jit(run, static_argnames=("n_steps", "save_history"))

final_state, history = run_jit(
    state,
    grid,
    params,
    n_steps          = N_STEPS,
    forcing_sequence = forcing,    # SurfaceForcing or None
    save_history     = False,      # True -> history has a leading time axis
)
```

**Chunked runs (recommended for long integrations and per-day diagnostics):**

```python
CHUNK = 288   # 1 day at dt=300 s

for day in range(N_DAYS):
    state, _ = run_jit(state, grid, params,
                       n_steps=CHUNK,
                       forcing_sequence=_build_chunk(day, CHUNK),
                       save_history=False)
    T_arr = np.array(state.T)
    print(f"Day {day+1:3d}: SST = {T_arr[:,:,0].mean():.3f} C  "
          f"eta_max = {np.abs(np.array(state.eta)).max():.4f} m")
```

---

### 5.6 Save Output

```python
import netCDF4 as nc
import numpy as np

ds = nc.Dataset("output.nc", "w", format="NETCDF4")
ds.createDimension("time", None)
ds.createDimension("x", grid.Nx)
ds.createDimension("y", grid.Ny)
ds.createDimension("z", grid.Nz)

ds.createVariable("time", "f4", ("time",));       ds["time"].units = "days"
ds.createVariable("T",    "f4", ("time","x","y","z")); ds["T"].units = "degC"
ds.createVariable("S",    "f4", ("time","x","y","z")); ds["S"].units = "psu"
ds.createVariable("eta",  "f4", ("time","x","y"));     ds["eta"].units = "m"

i = len(ds["time"])
ds["time"][i] = float(state.time) / 86400.0
ds["T"][i]    = np.array(state.T)
ds["S"][i]    = np.array(state.S)
ds["eta"][i]  = np.array(state.eta)
ds.sync()
ds.close()
```

A complete experiment driver with per-day diagnostics and full NetCDF output is available in `experiment.py`. Standard figures can be generated with:

```bash
python plot_output.py   # reads output_*.nc from the working directory
```

---

## 6. Using ORAS5 Reanalysis Data

### 6.1 Download Data

Download from [Copernicus Marine Service](https://marine.copernicus.eu) (product `GLOBAL_MULTIYEAR_PHY_001_030`).

| Variable | File name pattern | Used for |
|---|---|---|
| Temperature | `votemper_*.nc` | Initial T |
| Salinity | `vosaline_*.nc` | Initial S |
| Zonal velocity | `vozocrtx_*.nc` | Full-state init only |
| Meridional velocity | `vomecrty_*.nc` | Full-state init only |
| Sea surface height | `sossheig_*.nc` | Full-state init only |
| Net heat flux | `sohefldo_*.nc` | Surface forcing |
| Freshwater flux | `sowaflup_*.nc` | Surface forcing |
| Zonal wind stress | `sozotaux_*.nc` | Surface forcing |
| Meridional wind stress | `sometauy_*.nc` | Surface forcing |

Place all files in `OceanJAX/data/data_oras5/`.

### 6.2 Merge 3-D Files

```bash
python OceanJAX/data/data_oras5/data_merger.py
# Produces: oras5_<year>_<month>_native_merged.nc
```

### 6.3 Load Initial Conditions

```python
import warnings
from OceanJAX.data.oras5 import read_oras5, regrid_to_model
from OceanJAX.state import create_from_arrays
import jax.numpy as jnp

ORAS5_FILE = "OceanJAX/data/data_oras5/oras5_2026_01_native_merged.nc"

with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    raw        = read_oras5(ORAS5_FILE, time_index=0)
    full_state = regrid_to_model(raw, grid)

# Cold start — T/S from ORAS5, zero velocity and sea level (recommended)
zeros3 = jnp.zeros((grid.Nx, grid.Ny, grid.Nz), dtype=jnp.float32)
zeros2 = jnp.zeros((grid.Nx, grid.Ny),           dtype=jnp.float32)
state  = create_from_arrays(grid, u=zeros3, v=zeros3,
                             T=full_state.T, S=full_state.S, eta=zeros2)

# Full-state start — T/S/u/v/eta all from ORAS5
state = full_state
```

### 6.4 Load Surface Forcing

```python
from OceanJAX.data.oras5 import read_oras5_forcing, regrid_forcing
from OceanJAX.timeStepping import SurfaceForcing
import numpy as np

FORCING_FILES = [
    "OceanJAX/data/data_oras5/sohefldo_*.nc",
    "OceanJAX/data/data_oras5/sowaflup_*.nc",
    "OceanJAX/data/data_oras5/sozotaux_*.nc",
    "OceanJAX/data/data_oras5/sometauy_*.nc",
]

with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    raw_f = read_oras5_forcing(FORCING_FILES, time_index=0)
    sf    = regrid_forcing(raw_f, grid,
                           use_fields={"heat_flux", "fw_flux", "tau_x", "tau_y"})

# Broadcast the single snapshot to all N_STEPS
ones = np.ones((N_STEPS, 1, 1), dtype=np.float32)
forcing = SurfaceForcing(
    heat_flux = jnp.asarray(np.array(sf.heat_flux) * ones),
    fw_flux   = jnp.asarray(np.array(sf.fw_flux)   * ones),
    tau_x     = jnp.asarray(np.array(sf.tau_x)     * ones),
    tau_y     = jnp.asarray(np.array(sf.tau_y)     * ones),
)
```

---

## 7. Time-Varying Forcing

`OceanJAX.data.forcing` provides two functions for building genuinely time-varying forcing sequences.

### `make_synthetic_forcing` — analytical specification

```python
from OceanJAX.data.forcing import make_synthetic_forcing

forcing = make_synthetic_forcing(
    grid, n_steps, dt,
    heat_flux = {"mean": -50.0, "amplitude": 150.0,
                 "period": 365*86400, "phase": 0.0},
    tau_x     = -0.05,           # constant scalar
    tau_y     = wind_pattern,    # constant (Nx, Ny) spatial array
    fw_flux   = 0.0,
)
```

Each field parameter accepts:

| Type | Behaviour |
|---|---|
| `float` | Spatially and temporally constant |
| `np.ndarray (Nx, Ny)` | Spatially varying, constant in time |
| `dict` | `mean + amplitude * sin(2*pi*t/period + phase)`, each can be scalar or `(Nx, Ny)` |

### `make_forcing_sequence` — interpolation from real snapshots

```python
from OceanJAX.data.forcing import make_forcing_sequence

snapshots = []
for k, t_s in enumerate(month_start_times_seconds):
    raw = read_oras5_forcing(files, time_index=k)
    sf  = regrid_forcing(raw, grid)
    snapshots.append((t_s, sf))

forcing = make_forcing_sequence(
    snapshots, n_steps=N_STEPS, dt=300.0,
    interp  = "cyclic",   # repeat the annual cycle indefinitely
    t_start = 0.0,
)
```

| `interp` mode | Behaviour |
|---|---|
| `"linear"` | Piecewise-linear between snapshots; clamps at edges |
| `"nearest"` | Step function; uses the nearest snapshot value |
| `"cyclic"` | Linear + periodic wrapping; ideal for repeating annual cycles |

---

## 8. ML Closure Interface

The model exposes a single hook per time step for injecting learnable parameterizations. The hook fires after the explicit tracer tendencies are computed and before the implicit vertical diffusion.

### Define a closure

```python
import equinox as eqx
import jax.numpy as jnp
from OceanJAX.ml.closure import AbstractClosure, ClosureOutput

class MyClosure(AbstractClosure):
    weights: jnp.ndarray   # any equinox Module fields

    def __call__(self, state, grid, params) -> ClosureOutput:
        dT_tend = self.weights * state.T
        return ClosureOutput(
            dT_tend       = dT_tend,
            dS_tend       = jnp.zeros_like(dT_tend),
            kappa_v_scale = jnp.array(1.0),
        )
```

### Use a closure

```python
closure = MyClosure(weights=jnp.zeros((grid.Nx, grid.Ny, grid.Nz)))

final, _ = run_jit(state, grid, params, n_steps=N_STEPS,
                   forcing_sequence=forcing, closure=closure)
```

### ClosureOutput fields

| Field | Shape | Description |
|---|---|---|
| `dT_tend` | `(Nx, Ny, Nz)` [K/s] | Added to explicit T tendency before AB3 advance |
| `dS_tend` | `(Nx, Ny, Nz)` [psu/s] | Added to explicit S tendency before AB3 advance |
| `kappa_v_scale` | scalar or `(Nx, Ny, Nz+1)` | Multiplies `params.kappa_v` before implicit diffusion |

### No-op closure (for testing)

```python
from OceanJAX.ml.closure import NullClosure

closure = NullClosure()
# Produces bit-identical results to closure=None.
```

### Training with gradients

```python
import optax, equinox as eqx

@eqx.filter_jit
@eqx.filter_grad
def loss_fn(closure, state, target_T):
    final, _ = run(state, grid, params, n_steps=N_STEPS,
                   forcing_sequence=forcing, closure=closure)
    return jnp.mean((final.T - target_T) ** 2)

grads     = loss_fn(closure, state, target_T)
optim     = optax.adam(1e-4)
opt_state = optim.init(eqx.filter(closure, eqx.is_array))
updates, opt_state = optim.update(grads, opt_state)
closure   = eqx.apply_updates(closure, updates)
```

---

## 9. Ensemble / Multi-GPU Runs

```python
from OceanJAX.parallel.ensemble import batch_run, sharded_ensemble_run
import jax

N_MEMBERS = 8

# Stack N_MEMBERS copies of the initial state along a leading batch axis
batched_state = jax.tree_util.tree_map(
    lambda x: jnp.stack([x] * N_MEMBERS), state
)

# Option A: single GPU (vmap over the batch axis)
final_batch, _ = batch_run(
    batched_state, grid, params,
    n_steps=N_STEPS, forcing_sequence=None, save_history=False,
)

# Option B: multi-GPU (NamedSharding; N_MEMBERS must be divisible by n_devices)
final_batch, _ = sharded_ensemble_run(
    batched_state, grid, params,
    n_steps=N_STEPS, forcing_sequence=None, save_history=False,
)

# final_batch.T has shape (N_MEMBERS, Nx, Ny, Nz)
T_mean = jnp.mean(final_batch.T, axis=0)
T_std  = jnp.std( final_batch.T, axis=0)
```

---

## 10. Running Verification Tests

```bash
# Full test suite
python -m pytest OceanJAX/tests/ -v

# Individual modules
python -m pytest OceanJAX/tests/test_dynamics.py     -v
python -m pytest OceanJAX/tests/test_tracers.py      -v
python -m pytest OceanJAX/tests/test_timeStepping.py -v
python -m pytest OceanJAX/tests/test_closure.py      -v
python -m pytest OceanJAX/tests/test_forcing.py      -v
python -m pytest OceanJAX/tests/test_oras5.py        -v
python -m pytest OceanJAX/tests/test_parallel.py     -v

# Verification experiments (Chapter 5)
python verification_experiments/exp_5_1_1_rest_conservation.py
python verification_experiments/exp_5_1_2_stratified_bathymetry.py
python verification_experiments/exp_5_1_3_null_closure.py
python verification_experiments/exp_5_1_4_cfl_scan.py
python verification_experiments/exp_5_1_5_vertical_mixing.py
python verification_experiments/exp_5_2_1_cold_full_forcing.py
python verification_experiments/exp_5_2_2_cold_no_forcing.py
python verification_experiments/exp_5_2_3_full_no_forcing.py
```

---

## 11. Stability Guidelines

The model uses leapfrog + Asselin-Robert for momentum and free surface, and Adams-Bashforth 3 for tracers. The critical constraint is the barotropic CFL number:

```
CFL = c_bt * dt / dx_min

where  c_bt    = sqrt(g * H_max)    barotropic wave speed [m/s]
       dx_min  = minimum horizontal grid spacing [m]
```

| CFL range | Outcome |
|---|---|
| <= 0.33 | Stable |
| 0.33 – 0.44 | Use with caution |
| > 0.44 | Unstable — NaN within first day |

**Validated configurations:**

| DEPTH_MAX | Approx. dx | Max safe dt | Notes |
|---|---|---|---|
| 500 m | 200 km | 900 s | |
| 1000 m | 200 km | 600 s | |
| 500 m | 200 km | **300 s** | Recommended — 3x safety margin |

**Rules of thumb:**
- Always start with `dt = 300 s`.
- Cold-start initial conditions (`u = v = eta = 0`) are significantly more stable than full-state ORAS5 initialization.
- If increasing `DEPTH_MAX` beyond 1000 m, run `verification_experiments/exp_5_1_4_cfl_scan.py` first to identify the stability boundary for your configuration.

---

## 12. ModelParams Reference

| Parameter | Default | Unit | Description |
|---|---|---|---|
| `dt` | 900.0 | s | Time step (**override to 300 s**) |
| `rho0` | 1025.0 | kg/m^3 | Reference density (Boussinesq) |
| `g` | 9.81 | m/s^2 | Gravitational acceleration |
| `alpha_T` | 2e-4 | 1/K | Thermal expansion coefficient |
| `beta_S` | 7.4e-4 | 1/psu | Haline contraction coefficient |
| `T_ref` | 10.0 | C | EOS reference temperature |
| `S_ref` | 35.0 | psu | EOS reference salinity |
| `nu_h` | 200.0 | m^2/s | Horizontal viscosity |
| `nu_v` | 1e-4 | m^2/s | Vertical viscosity |
| `kappa_h` | 100.0 | m^2/s | Horizontal tracer diffusivity |
| `kappa_v` | 1e-5 | m^2/s | Vertical tracer diffusivity |
| `asselin_coeff` | 0.1 | — | Asselin-Robert filter coefficient |
| `ab3_coeffs` | (23/12, -16/12, 5/12) | — | Adams-Bashforth 3 coefficients |
