"""
diagnose_init.py
================
Lightweight ORAS5 → OceanState diagnostic.

Loads an ORAS5 file onto an OceanJAX grid and prints a summary table of
the initial-condition fields (min, max, mean, non-zero count, non-finite count).
Optionally saves PNG figures of the surface T/S and the first zonal T section.

Usage
-----
    # Stats only:
    python diagnose_init.py --oras5_path /path/to/oras5.nc \\
        --nx 20 --ny 20 --nz 10 \\
        --lon_min -10 --lon_max 10 \\
        --lat_min 40 --lat_max 60 \\
        --depth_max 1000

    # With figures saved to ./figs/:
    python diagnose_init.py ... --plot --fig_dir ./figs
"""

from __future__ import annotations

import argparse
import sys

import numpy as np

from OceanJAX.grid import OceanGrid
from OceanJAX.data.oras5 import load_oras5


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _parse_args(argv=None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Diagnose ORAS5 → OceanState initialisation.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--oras5_path",       required=True,
                   help="Path to ORAS5 NetCDF file")
    p.add_argument("--oras5_time_index", type=int,   default=0,
                   help="Time index within the ORAS5 file")
    p.add_argument("--oras5_T_fill",     type=float, default=10.0,
                   help="T fallback for uncovered points [degC]")
    p.add_argument("--oras5_S_fill",     type=float, default=35.0,
                   help="S fallback for uncovered points [psu]")

    g = p.add_argument_group("grid")
    g.add_argument("--nx",        type=int,   default=10)
    g.add_argument("--ny",        type=int,   default=10)
    g.add_argument("--nz",        type=int,   default=5)
    g.add_argument("--lon_min",   type=float, default=0.0)
    g.add_argument("--lon_max",   type=float, default=10.0)
    g.add_argument("--lat_min",   type=float, default=10.0)
    g.add_argument("--lat_max",   type=float, default=20.0)
    g.add_argument("--depth_max", type=float, default=500.0)

    p.add_argument("--plot",    action="store_true",
                   help="Save diagnostic PNG figures")
    p.add_argument("--fig_dir", type=str, default=".",
                   help="Directory for PNG figures (created if absent)")
    return p.parse_args(argv)


# ---------------------------------------------------------------------------
# Stats helper
# ---------------------------------------------------------------------------

def _stats(name: str, arr: np.ndarray, ocean_mask: np.ndarray) -> None:
    """Print one line of statistics for a field, ignoring land cells."""
    ocean = arr[ocean_mask.astype(bool)]
    if ocean.size == 0:
        print(f"  {name:8s}  (no ocean cells)")
        return
    not_finite_count = int(np.sum(~np.isfinite(ocean)))
    valid            = ocean[np.isfinite(ocean)]
    if valid.size == 0:
        print(f"  {name:8s}  ALL non-finite  ({not_finite_count} cells)")
        return
    print(
        f"  {name:8s}  "
        f"min={valid.min():+10.4f}  "
        f"max={valid.max():+10.4f}  "
        f"mean={valid.mean():+10.4f}  "
        f"nonzero={int(np.count_nonzero(valid)):6d}/{valid.size:6d}  "
        f"non-finite={not_finite_count}"
    )


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def _make_figures(state, grid: OceanGrid, fig_dir: str) -> None:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("WARNING: matplotlib not installed — skipping figures.", file=sys.stderr)
        return

    import os
    os.makedirs(fig_dir, exist_ok=True)

    lon = np.array(grid.lon_c)
    lat = np.array(grid.lat_c)
    dep = np.array(grid.z_c)

    T   = np.array(state.T)    # (Nx, Ny, Nz)
    S   = np.array(state.S)
    eta = np.array(state.eta)  # (Nx, Ny)

    # Replace land (zero under mask) with NaN for cleaner plots
    mask2d = np.array(grid.mask_c[:, :, 0], dtype=bool)
    mask3d = np.array(grid.mask_c, dtype=bool)
    T_plot   = np.where(mask3d, T,   np.nan)
    S_plot   = np.where(mask3d, S,   np.nan)
    eta_plot = np.where(mask2d, eta, np.nan)

    # ---- surface T and S ----
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    fig.suptitle("Initial condition — surface fields")

    for ax, field, title, unit in zip(
        axes,
        [T_plot[:, :, 0], S_plot[:, :, 0], eta_plot],
        ["T  k=0", "S  k=0", "eta"],
        ["°C",     "psu",     "m"],
    ):
        im = ax.pcolormesh(lon, lat, field.T, shading="auto")
        plt.colorbar(im, ax=ax, label=unit)
        ax.set_title(title)
        ax.set_xlabel("lon")
        ax.set_ylabel("lat")

    fig.tight_layout()
    surface_path = os.path.join(fig_dir, "diag_surface.png")
    fig.savefig(surface_path, dpi=120)
    plt.close(fig)
    print(f"  Saved: {surface_path}", file=sys.stderr)

    # ---- zonal T section at j=Ny//2 ----
    j_mid = grid.Ny // 2
    T_sec = T_plot[:, j_mid, :]   # (Nx, Nz)

    fig, ax = plt.subplots(figsize=(8, 4))
    im = ax.pcolormesh(lon, dep, T_sec.T, shading="auto")
    plt.colorbar(im, ax=ax, label="°C")
    ax.invert_yaxis()
    ax.set_title(f"Initial T  —  zonal section at lat≈{float(lat[j_mid]):.2f}°")
    ax.set_xlabel("lon")
    ax.set_ylabel("depth [m]")
    fig.tight_layout()
    section_path = os.path.join(fig_dir, "diag_T_section.png")
    fig.savefig(section_path, dpi=120)
    plt.close(fig)
    print(f"  Saved: {section_path}", file=sys.stderr)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(argv=None) -> None:
    args = _parse_args(argv)

    # Build grid
    dz = args.depth_max / args.nz
    depth_levels = (np.arange(args.nz) + 0.5) * dz
    grid = OceanGrid.create(
        lon_bounds=(args.lon_min, args.lon_max),
        lat_bounds=(args.lat_min, args.lat_max),
        depth_levels=depth_levels,
        Nx=args.nx,
        Ny=args.ny,
    )

    print(
        f"\nDiagnosing ORAS5 initialisation\n"
        f"  File  : {args.oras5_path}\n"
        f"  t_idx : {args.oras5_time_index}\n"
        f"  Grid  : {grid.Nx}×{grid.Ny}×{grid.Nz}  "
        f"lon[{args.lon_min},{args.lon_max}] "
        f"lat[{args.lat_min},{args.lat_max}] "
        f"z[0,{args.depth_max}] m\n",
        file=sys.stderr,
    )

    state = load_oras5(
        args.oras5_path,
        grid,
        time_index=args.oras5_time_index,
        T_fill=args.oras5_T_fill,
        S_fill=args.oras5_S_fill,
    )

    # Build masks appropriate for each variable's staggered grid position.
    # Using tracer mask_c for u/v would give wrong cell counts and NaN stats
    # near staggered boundaries.
    mask_c = np.array(grid.mask_c, dtype=bool)          # (Nx, Ny, Nz)  tracer
    mask_u = np.array(grid.mask_u, dtype=bool)          # (Nx, Ny, Nz)  u-face
    mask_v = np.array(grid.mask_v, dtype=bool)          # (Nx, Ny, Nz)  v-face
    mask2d = mask_c[:, :, 0]                            # (Nx, Ny)

    print("\nField statistics  (ocean cells only)")
    print("-" * 72)
    _stats("T",   np.array(state.T),   mask_c)
    _stats("S",   np.array(state.S),   mask_c)
    _stats("u",   np.array(state.u),   mask_u)
    _stats("v",   np.array(state.v),   mask_v)
    _stats("eta", np.array(state.eta), mask2d)
    print("-" * 72)

    # Overall non-finite check (ocean cells only, per staggered mask).
    # isfinite catches both NaN and Inf so blow-up during initialisation
    # is detected before the first time step.
    T_arr   = np.array(state.T)
    S_arr   = np.array(state.S)
    u_arr   = np.array(state.u)
    v_arr   = np.array(state.v)
    eta_arr = np.array(state.eta)
    any_not_finite = bool(
        np.any(~np.isfinite(T_arr[mask_c]))   or
        np.any(~np.isfinite(S_arr[mask_c]))   or
        np.any(~np.isfinite(u_arr[mask_u]))   or
        np.any(~np.isfinite(v_arr[mask_v]))   or
        np.any(~np.isfinite(eta_arr[mask2d]))
    )
    print(f"\nnon-finite present (ocean cells): {any_not_finite}")
    if any_not_finite:
        print("WARNING: non-finite value detected in initial state!", file=sys.stderr)

    if args.plot:
        print("\nGenerating figures …", file=sys.stderr)
        _make_figures(state, grid, args.fig_dir)

    print("\nDone.", file=sys.stderr)


if __name__ == "__main__":
    main()
