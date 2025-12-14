"""
Compare Original TCI vs PatchedTCI error outputs for vortex data.

Usage:
    python compare_tci_vortex.py <errors_file.h5> [--save-prefix PREFIX] [--no-show]

The HDF5 file is expected to follow the format written by the
Julia driver in `FD/velocitymagnitudedata/apply_adaptive_tci.jl`:
  - datasets: xs, ys, fs, errors_original_tci, errors_patched_tci
  - metadata: metadata/tolerance, metadata/maxbonddim, metadata/R, metadata/npoints
  - timing: time_original_tci, time_patched_tci
  - bond dimensions: bonddims_original_tci, bonddims_patched_tci
  - optional: patch_bounds (n_patches x 4, x_lower, x_upper, y_lower, y_upper)
"""

import argparse
import os
import sys
from typing import Dict, Any, Optional

import h5py
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

# Import shared functions to avoid duplication
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "FD", "velocitymagnitudedata"))
try:
    from plot_errors_from_file import (
        read_errors_file, _reshape_to_grid, 
        plot_2d_error_maps, plot_data_with_patches, plot_error_statistics
    )
    USE_SHARED_PLOTS = True
except ImportError:
    USE_SHARED_PLOTS = False
    # Fallback: define locally if import fails
    def read_errors_file(h5file: str) -> Dict[str, Any]:
        """Load error/timing/bond-dimension data from the HDF5 output."""
        if not os.path.exists(h5file):
            raise FileNotFoundError(f"File not found: {h5file}")
        with h5py.File(h5file, "r") as f:
            return {
                "xs": f["xs"][:],
                "ys": f["ys"][:],
                "fs": f["fs"][:],
                "errors_original": f["errors_original_tci"][:],
                "errors_patched": f["errors_patched_tci"][:],
                "time_original": float(f["time_original_tci"][()]),
                "time_patched": float(f["time_patched_tci"][()]),
                "bonddims_original": f["bonddims_original_tci"][:],
                "bonddims_patched": f["bonddims_patched_tci"][:],
                "tolerance": float(f["metadata/tolerance"][()]),
                "maxbonddim": int(f["metadata/maxbonddim"][()]),
                "R": int(f["metadata/R"][()]),
                "npoints": int(f["metadata/npoints"][()]),
                "stats_original": {
                    "mean": float(f["stats_original/mean_error"][()]),
                    "median": float(f["stats_original/median_error"][()]),
                    "max": float(f["stats_original/max_error"][()]),
                    "std": float(f["stats_original/std_error"][()]),
                },
                "stats_patched": {
                    "mean": float(f["stats_patched/mean_error"][()]),
                    "median": float(f["stats_patched/median_error"][()]),
                    "max": float(f["stats_patched/max_error"][()]),
                    "std": float(f["stats_patched/std_error"][()]),
                },
                "patch_bounds": f["patch_bounds"][:] if "patch_bounds" in f else None,
            }
    # Fallback: define _reshape_to_grid locally if import fails
    def _reshape_to_grid(xs: np.ndarray, ys: np.ndarray, values: np.ndarray):
        x_coords = np.sort(np.unique(xs))
        y_coords = np.sort(np.unique(ys))
        Nx, Ny = len(x_coords), len(y_coords)
        x_to_idx = {x: i for i, x in enumerate(x_coords)}
        y_to_idx = {y: i for i, y in enumerate(y_coords)}
        grid = np.zeros((Nx, Ny), dtype=values.dtype)
        for i in range(len(xs)):
            grid[x_to_idx[xs[i]], y_to_idx[ys[i]]] = values[i]
        return x_coords, y_coords, grid


def plot_error_maps(data: Dict[str, Any], save_prefix: Optional[str] = None):
    """Plot error maps using shared function if available."""
    if USE_SHARED_PLOTS:
        output_file = f"{save_prefix}_errmaps.png" if save_prefix else None
        return plot_2d_error_maps(data, output_file)
    
    # Fallback implementation
    xs, ys = data["xs"], data["ys"]
    errors_orig = data["errors_original"]
    errors_patch = data["errors_patched"]
    x_coords, y_coords, err_orig_2d = _reshape_to_grid(xs, ys, errors_orig)
    _, _, err_patch_2d = _reshape_to_grid(xs, ys, errors_patch)
    vmax = max(err_orig_2d.max(), err_patch_2d.max())
    
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    im1 = axes[0].pcolormesh(x_coords, y_coords, err_orig_2d.T, cmap="plasma_r", 
                             shading="auto", vmin=0, vmax=vmax)
    axes[0].set_title("Error Map: Original TCI")
    axes[0].set_xlabel("x")
    axes[0].set_ylabel("y")
    plt.colorbar(im1, ax=axes[0], label="|error|")
    
    im2 = axes[1].pcolormesh(x_coords, y_coords, err_patch_2d.T, cmap="plasma_r",
                             shading="auto", vmin=0, vmax=vmax)
    axes[1].set_title("Error Map: PatchedTCI")
    axes[1].set_xlabel("x")
    axes[1].set_ylabel("y")
    plt.colorbar(im2, ax=axes[1], label="|error|")
    plt.tight_layout()
    
    if save_prefix:
        out = f"{save_prefix}_errmaps.png"
        plt.savefig(out, dpi=150, bbox_inches="tight")
        print(f"Saved: {out}")
    return fig


def plot_statistics(data: Dict[str, Any], save_prefix: Optional[str] = None):
    """Plot statistics using shared function if available."""
    if USE_SHARED_PLOTS:
        output_file = f"{save_prefix}_stats.png" if save_prefix else None
        return plot_error_statistics(data, output_file)
    
    # Fallback implementation
    errors_orig = data["errors_original"]
    errors_patch = data["errors_patched"]
    stats_orig = data["stats_original"]
    stats_patch = data["stats_patched"]
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    # Histogram (log y)
    ax = axes[0, 0]
    ax.hist(errors_orig, bins=50, alpha=0.6, label="Original TCI", color="tab:blue")
    ax.hist(errors_patch, bins=50, alpha=0.6, label="PatchedTCI", color="tab:red")
    ax.set_yscale("log")
    ax.set_xlabel("|error|")
    ax.set_ylabel("freq")
    ax.set_title("Error distribution")
    ax.legend()

    # CDF
    ax = axes[0, 1]
    so = np.sort(errors_orig)
    sp = np.sort(errors_patch)
    ax.plot(so, np.arange(len(so)) / len(so), label="Original", lw=2)
    ax.plot(sp, np.arange(len(sp)) / len(sp), label="Patched", lw=2)
    ax.set_xlabel("|error|")
    ax.set_ylabel("CDF")
    ax.set_title("Error CDF")
    ax.legend()

    # Box
    ax = axes[1, 0]
    bp = ax.boxplot([errors_orig, errors_patch], labels=["Original", "Patched"], patch_artist=True)
    bp["boxes"][0].set_facecolor("tab:blue")
    bp["boxes"][1].set_facecolor("tab:red")
    ax.set_yscale("log")
    ax.set_ylabel("|error|")
    ax.set_title("Error statistics (log scale)")

    # Text summary
    ax = axes[1, 1]
    ax.axis("off")
    text = (
        f"Original TCI:\n"
        f"  mean   = {stats_orig['mean']:.3e}\n"
        f"  median = {stats_orig['median']:.3e}\n"
        f"  max    = {stats_orig['max']:.3e}\n"
        f"  std    = {stats_orig['std']:.3e}\n\n"
        f"PatchedTCI:\n"
        f"  mean   = {stats_patch['mean']:.3e}\n"
        f"  median = {stats_patch['median']:.3e}\n"
        f"  max    = {stats_patch['max']:.3e}\n"
        f"  std    = {stats_patch['std']:.3e}\n\n"
        f"Timing:\n"
        f"  original = {data['time_original']:.2f} s\n"
        f"  patched  = {data['time_patched']:.2f} s\n"
        f"  speedup  = {data['time_original'] / data['time_patched']:.2f}x\n"
    )
    ax.text(0.02, 0.95, text, va="top", ha="left", family="monospace")

    plt.tight_layout()
    if save_prefix:
        out = f"{save_prefix}_stats.png"
        plt.savefig(out, dpi=150, bbox_inches="tight")
        print(f"Saved: {out}")
    return fig


def plot_field_with_patches(data: Dict[str, Any], save_prefix: Optional[str] = None):
    """Plot field with patches using shared function if available."""
    if USE_SHARED_PLOTS:
        output_file = f"{save_prefix}_field.png" if save_prefix else None
        return plot_data_with_patches(data, output_file)
    
    # Fallback implementation
    xs, ys, fs = data["xs"], data["ys"], data["fs"]
    patch_bounds = data.get("patch_bounds")
    x_coords, y_coords, field_2d = _reshape_to_grid(xs, ys, fs)
    
    fig, ax = plt.subplots(figsize=(7, 6))
    im = ax.pcolormesh(x_coords, y_coords, field_2d.T, shading="auto", cmap="viridis")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_title("Field with patch bounds")
    plt.colorbar(im, ax=ax, label="f(x, y)")
    
    if patch_bounds is not None:
        bounds = patch_bounds
        if bounds.ndim == 2 and bounds.shape[1] == 4:
            for (x1, x2, y1, y2) in bounds:
                rect = mpatches.Rectangle((x1, y1), x2 - x1, y2 - y1, 
                                         fill=False, edgecolor="red", lw=1.5)
                ax.add_patch(rect)
        else:
            print(f"Warning: unexpected patch_bounds shape {bounds.shape}, skipping overlays")
    
    plt.tight_layout()
    if save_prefix:
        out = f"{save_prefix}_field.png"
        plt.savefig(out, dpi=150, bbox_inches="tight")
        print(f"Saved: {out}")
    return fig


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("errors_file", help="HDF5 file produced by the Julia TCI comparison")
    parser.add_argument("--save-prefix", default=None, help="If set, save PNGs with this prefix")
    parser.add_argument("--no-show", action="store_true", help="Do not display figures (useful for batch runs)")
    args = parser.parse_args(argv)

    data = read_errors_file(args.errors_file)

    print(f"Loaded {args.errors_file}")
    print(f"  points     : {data['npoints']}")
    print(f"  R          : {data['R']}")
    print(f"  tolerance  : {data['tolerance']}")
    print(f"  maxbonddim : {data['maxbonddim']}")
    print(f"  time orig  : {data['time_original']:.2f} s")
    print(f"  time patch : {data['time_patched']:.2f} s")
    speedup = data['time_original'] / data['time_patched'] if data['time_patched'] > 0 else float('inf')
    print(f"  speedup    : {speedup:.2f}x")

    plot_error_maps(data, args.save_prefix)
    plot_statistics(data, args.save_prefix)
    plot_field_with_patches(data, args.save_prefix)

    if not args.no_show:
        plt.show()


if __name__ == "__main__":
    main()
