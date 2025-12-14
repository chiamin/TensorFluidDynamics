"""
Plot 2D profile of vortex state from HDF5 file.

Automatically plots 125v_complete.h5 with 128x128 grid size.
Modify h5file and grid_size in main() function to change settings.

Usage:
    python plot_vortex_2d.py
"""

import h5py
import numpy as np
import matplotlib.pyplot as plt
import sys
import os

def plot_vortex_2d(h5file="125v_complete.h5", grid_size=64):
    """
    Read HDF5 file and create 2D plots of vortex profile.
    
    Args:
        h5file: Path to HDF5 file containing complete data
        grid_size: Grid size for plotting (will downsample if needed)
    """
    
    if not os.path.exists(h5file):
        print(f"Error: File '{h5file}' not found!")
        sys.exit(1)
    
    print(f"Reading data from: {h5file}")
    
    with h5py.File(h5file, 'r') as f:
        # Read coordinates and field values
        xs = f['xs'][:]
        ys = f['ys'][:]
        fs = f['fs'][:]
        
        # Read metadata if available
        metadata = {}
        if 'metadata' in f:
            if 'x_range' in f['metadata']:
                metadata['x_range'] = f['metadata']['x_range'][:]
            if 'y_range' in f['metadata']:
                metadata['y_range'] = f['metadata']['y_range'][:]
            if 'N' in f['metadata']:
                metadata['N'] = f['metadata']['N'][()]
    
    print(f"  Loaded {len(xs)} points")
    
    # Reshape to 2D grid using shared utility pattern
    x_coords = np.sort(np.unique(xs))
    y_coords = np.sort(np.unique(ys))
    Nx_orig, Ny_orig = len(x_coords), len(y_coords)
    
    print(f"  Original grid size: {Nx_orig} × {Ny_orig}")
    print(f"  Data range: x ∈ [{x_coords.min():.2f}, {x_coords.max():.2f}], "
          f"y ∈ [{y_coords.min():.2f}, {y_coords.max():.2f}]")
    print(f"  Field range: f ∈ [{fs.min():.6e}, {fs.max():.6e}]")
    
    # Reshape to 2D grid
    x_to_idx = {x: i for i, x in enumerate(x_coords)}
    y_to_idx = {y: i for i, y in enumerate(y_coords)}
    fs_2d_orig = np.zeros((Nx_orig, Ny_orig))
    for i in range(len(xs)):
        fs_2d_orig[x_to_idx[xs[i]], y_to_idx[ys[i]]] = fs[i]
    
    # Downsample to chosen grid size for plotting
    if grid_size >= Nx_orig:
        print(f"  Note: Requested grid size {grid_size} >= original {Nx_orig}, using original data")
        x_plot = x_coords
        y_plot = y_coords
        fs_2d = fs_2d_orig
    else:
        print(f"  Downsampling to {grid_size}×{grid_size} for plotting...")
        # Create evenly spaced indices for downsampling
        x_indices = np.linspace(0, Nx_orig - 1, grid_size, dtype=int)
        y_indices = np.linspace(0, Ny_orig - 1, grid_size, dtype=int)
        
        x_plot = x_coords[x_indices]
        y_plot = y_coords[y_indices]
        fs_2d = fs_2d_orig[np.ix_(x_indices, y_indices)]
    
    print(f"  Plotting grid size: {len(x_plot)} × {len(y_plot)}")
    
    # Create figure with two subplots
    fig, axes = plt.subplots(1, 2, figsize=(16, 7))
    
    # 1. Heatmap
    ax = axes[0]
    im = ax.pcolormesh(x_plot, y_plot, fs_2d.T, 
                       shading='auto', cmap='viridis',
                       vmin=fs.min(), vmax=fs.max())
    ax.set_xlabel('x', fontsize=12)
    ax.set_ylabel('y', fontsize=12)
    ax.set_title(f'Vortex State 2D Profile (Heatmap, {len(x_plot)}×{len(y_plot)})', fontsize=14)
    ax.set_aspect('equal')
    plt.colorbar(im, ax=ax, label='|ψ|²')
    
    # 2. Contour plot
    ax = axes[1]
    contour = ax.contour(x_plot, y_plot, fs_2d.T,
                        levels=20, cmap='viridis', linewidths=1.5)
    ax.clabel(contour, inline=True, fontsize=8)
    ax.set_xlabel('x', fontsize=12)
    ax.set_ylabel('y', fontsize=12)
    ax.set_title(f'Vortex State 2D Profile (Contour, {len(x_plot)}×{len(y_plot)})', fontsize=14)
    ax.set_aspect('equal')
    
    plt.tight_layout()
    
    # Save figure
    base_name = os.path.splitext(h5file)[0]
    output_file = f"{base_name}_plot_{len(x_plot)}x{len(y_plot)}.png"
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    print(f"\nPlot saved to: {output_file}")
    
    # Also create a 3D surface plot
    fig2 = plt.figure(figsize=(10, 8))
    ax2 = fig2.add_subplot(111, projection='3d')
    
    X, Y = np.meshgrid(x_plot, y_plot)
    surf = ax2.plot_surface(X, Y, fs_2d.T, cmap='viridis', 
                           linewidth=0, antialiased=True, alpha=0.8)
    ax2.set_xlabel('x', fontsize=12)
    ax2.set_ylabel('y', fontsize=12)
    ax2.set_zlabel('|ψ|²', fontsize=12)
    ax2.set_title(f'Vortex State 3D Surface ({len(x_plot)}×{len(y_plot)})', fontsize=14)
    fig2.colorbar(surf, ax=ax2, label='|ψ|²', shrink=0.5)
    
    output_file2 = f"{base_name}_3d_{len(x_plot)}x{len(y_plot)}.png"
    plt.savefig(output_file2, dpi=150, bbox_inches='tight')
    print(f"3D plot saved to: {output_file2}")
    
    plt.show()
    
    return fig, fig2

def main():
    # Configuration - modify these values as needed
    h5file = "125v_complete.h5"
    grid_size = 128
    
    plot_vortex_2d(h5file, grid_size)

if __name__ == "__main__":
    main()

