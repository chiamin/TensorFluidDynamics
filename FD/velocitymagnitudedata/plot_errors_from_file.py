"""
Python script to read and plot errors saved from Julia TCI analysis.

Usage:
    python plot_errors_from_file.py f_t1e-8_errors.h5
"""

import h5py
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import sys
import os

def _reshape_to_grid(xs, ys, values):
    """
    Reshape 1D coordinate and value arrays to 2D grid.
    Handles unordered data by using coordinate lookups.
    
    Returns:
        x_coords: Sorted unique x coordinates
        y_coords: Sorted unique y coordinates  
        grid: 2D array of shape (Nx, Ny)
    """
    x_coords = np.sort(np.unique(xs))
    y_coords = np.sort(np.unique(ys))
    Nx = len(x_coords)
    Ny = len(y_coords)
    
    x_to_idx = {x: i for i, x in enumerate(x_coords)}
    y_to_idx = {y: i for i, y in enumerate(y_coords)}
    
    grid = np.zeros((Nx, Ny), dtype=values.dtype)
    for i in range(len(xs)):
        grid[x_to_idx[xs[i]], y_to_idx[ys[i]]] = values[i]
    
    return x_coords, y_coords, grid

def read_errors_file(h5file):
    """Read errors and data from HDF5 file."""
    with h5py.File(h5file, 'r') as f:
        # Read coordinates and true values
        xs = f['xs'][:]
        ys = f['ys'][:]
        fs = f['fs'][:]
        
        # Read errors
        errors_original = f['errors_original_tci'][:]
        errors_patched = f['errors_patched_tci'][:]
        
        # Read timing
        time_original = f['time_original_tci'][()]
        time_patched = f['time_patched_tci'][()]
        
        # Read bond dimensions
        bonddims_original = f['bonddims_original_tci'][:]
        bonddims_patched = f['bonddims_patched_tci'][:]
        
        # Read metadata
        tolerance = f['metadata/tolerance'][()]
        maxbonddim = f['metadata/maxbonddim'][()]
        R = f['metadata/R'][()]
        npoints = f['metadata/npoints'][()]
        
        # Read statistics
        stats_original = {
            'mean': f['stats_original/mean_error'][()],
            'median': f['stats_original/median_error'][()],
            'max': f['stats_original/max_error'][()],
            'std': f['stats_original/std_error'][()]
        }
        
        stats_patched = {
            'mean': f['stats_patched/mean_error'][()],
            'median': f['stats_patched/median_error'][()],
            'max': f['stats_patched/max_error'][()],
            'std': f['stats_patched/std_error'][()]
        }

        patch_bounds = None
        if 'patch_bounds' in f:
            patch_bounds = f['patch_bounds'][:]
        
        return {
            'xs': xs,
            'ys': ys,
            'fs': fs,
            'errors_original': errors_original,
            'errors_patched': errors_patched,
            'time_original': time_original,
            'time_patched': time_patched,
            'bonddims_original': bonddims_original,
            'bonddims_patched': bonddims_patched,
            'tolerance': tolerance,
            'maxbonddim': maxbonddim,
            'R': R,
            'npoints': npoints,
            'stats_original': stats_original,
            'stats_patched': stats_patched,
            'patch_bounds': patch_bounds
        }

def plot_data_with_patches(data, output_file=None):
    """Plot the original data fs with patch boundaries overlaid."""
    xs = data['xs']
    ys = data['ys']
    fs = data['fs']
    patch_bounds = data.get('patch_bounds', None)

    # Reshape data to 2D grid
    x_coords, y_coords, fs_2d = _reshape_to_grid(xs, ys, fs)

    fig, ax = plt.subplots(figsize=(8, 6))
    im = ax.pcolormesh(x_coords, y_coords, fs_2d.T, shading='auto', cmap='viridis')
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_title('Field with Patch Boundaries')
    plt.colorbar(im, ax=ax, label='f(x, y)')

    if patch_bounds is not None:
        # Ensure shape is (n_patches, 4)
        pb = patch_bounds
        if pb.shape[1] == 4:
            bounds = pb
        elif pb.shape[0] == 4:
            bounds = pb.T
        else:
            print(f"Warning: unexpected patch_bounds shape {pb.shape}, skipping patch overlay")
            bounds = None

        if bounds is not None:
            for (x1, x2, y1, y2) in bounds:
                rect = mpatches.Rectangle((x1, y1), x2 - x1, y2 - y1,
                                          fill=False, edgecolor='red', linewidth=1.5)
                ax.add_patch(rect)

    plt.tight_layout()

    # Save if requested (display handled in main)
    if output_file:
        plt.savefig(output_file, dpi=150, bbox_inches='tight')
        print(f"Saved plot to: {output_file}")

    return fig

def plot_2d_error_maps(data, output_file=None):
    """Plot 2D error maps for both methods (no difference map)."""
    xs = data['xs']
    ys = data['ys']
    errors_orig = data['errors_original']
    errors_patch = data['errors_patched']
    
    # Reshape errors to 2D grids
    x_coords, y_coords, errors_orig_2d = _reshape_to_grid(xs, ys, errors_orig)
    _, _, errors_patch_2d = _reshape_to_grid(xs, ys, errors_patch)
    
    # Shared scale; use bright-for-small errors colormap
    vmax = max(errors_orig.max(), errors_patch.max())
    cmap_err = 'plasma_r'  # bright yellow for small values, dark for large

    # Create figure (two panels only)
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Original TCI error map
    ax = axes[0]
    im1 = ax.pcolormesh(x_coords, y_coords, errors_orig_2d.T,
                        cmap=cmap_err, shading='auto',
                        vmin=0, vmax=vmax)
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_title('Error Map: Original TCI')
    plt.colorbar(im1, ax=ax, label='Absolute Error')
    
    # PatchedTCI error map
    ax = axes[1]
    im2 = ax.pcolormesh(x_coords, y_coords, errors_patch_2d.T,
                        cmap=cmap_err, shading='auto',
                        vmin=0, vmax=vmax)
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_title('Error Map: PatchedTCI')
    plt.colorbar(im2, ax=ax, label='Absolute Error')
    
    plt.tight_layout()
    
    # Save if requested (display handled in main)
    if output_file:
        plt.savefig(output_file, dpi=150, bbox_inches='tight')
        print(f"Saved plot to: {output_file}")
    
    return fig

def plot_error_statistics(data, output_file=None):
    """Plot error statistics comparison."""
    errors_orig = data['errors_original']
    errors_patch = data['errors_patched']
    stats_orig = data['stats_original']
    stats_patch = data['stats_patched']
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # Histogram comparison
    ax = axes[0, 0]
    ax.hist(errors_orig, bins=50, alpha=0.6, label='Original TCI', color='blue')
    ax.hist(errors_patch, bins=50, alpha=0.6, label='PatchedTCI', color='red')
    ax.set_xlabel('Absolute Error')
    ax.set_ylabel('Frequency')
    ax.set_title('Error Distribution')
    ax.set_yscale('log')
    ax.legend()
    
    # CDF comparison
    ax = axes[0, 1]
    sort_orig = np.sort(errors_orig)
    sort_patch = np.sort(errors_patch)
    n_orig = len(sort_orig)
    n_patch = len(sort_patch)
    ax.plot(sort_orig, np.arange(n_orig) / n_orig, label='Original TCI', linewidth=2, color='blue')
    ax.plot(sort_patch, np.arange(n_patch) / n_patch, label='PatchedTCI', linewidth=2, color='red')
    ax.set_xlabel('Absolute Error')
    ax.set_ylabel('Cumulative Probability')
    ax.set_title('Error CDF')
    ax.legend()
    
    # Box plot
    ax = axes[1, 0]
    bp = ax.boxplot([errors_orig, errors_patch], labels=['Original TCI', 'PatchedTCI'],
                    patch_artist=True)
    bp['boxes'][0].set_facecolor('blue')
    bp['boxes'][1].set_facecolor('red')
    ax.set_ylabel('Absolute Error')
    ax.set_title('Error Statistics Comparison')
    ax.set_yscale('log')
    
    # Statistics text
    ax = axes[1, 1]
    ax.axis('off')
    stats_text = f"""
    Statistics:
    
    Original TCI:
      Mean: {stats_orig['mean']:.6e}
      Median: {stats_orig['median']:.6e}
      Max: {stats_orig['max']:.6e}
      Std: {stats_orig['std']:.6e}
    
    PatchedTCI:
      Mean: {stats_patch['mean']:.6e}
      Median: {stats_patch['median']:.6e}
      Max: {stats_patch['max']:.6e}
      Std: {stats_patch['std']:.6e}
    
    Timing:
      Original TCI: {data['time_original']:.2f} s
      PatchedTCI: {data['time_patched']:.2f} s
      Speedup: {data['time_original'] / data['time_patched']:.2f}x
    """
    ax.text(0.1, 0.5, stats_text, fontsize=11, family='monospace',
            verticalalignment='center', horizontalalignment='left')
    
    plt.tight_layout()
    
    # Show plot first
    plt.show()
    
    # Then save if output file specified
    if output_file:
        plt.savefig(output_file, dpi=150, bbox_inches='tight')
        print(f"Saved plot to: {output_file}")
    
    return fig

def main():
    h5file = "f_t1e-8_errors.h5"
    
    if not os.path.exists(h5file):
        print(f"Error: File '{h5file}' not found!")
        sys.exit(1)
    
    print(f"Reading errors from: {h5file}")
    data = read_errors_file(h5file)
    
    print(f"\nData loaded:")
    print(f"  Points: {data['npoints']}")
    print(f"  Tolerance: {data['tolerance']}")
    print(f"  Max bond dimension: {data['maxbonddim']}")
    print(f"  R: {data['R']}")
    
    # Plot 2D error maps
    base_name = os.path.splitext(h5file)[0]
    fig1 = plot_2d_error_maps(data, f"{base_name}_2d_maps.png")

    # Plot data with patch boundaries
    fig2 = plot_data_with_patches(data, f"{base_name}_data_with_patches.png")
    
    # Plot statistics
    #fig_stats = plot_error_statistics(data, f"{base_name}_statistics.png")

    # Show all figures once at the end
    plt.show()
    
    print("\nDone!")

if __name__ == "__main__":
    main()

