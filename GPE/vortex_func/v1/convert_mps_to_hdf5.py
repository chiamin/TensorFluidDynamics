"""
Convert MPS pickle data to HDF5 format for Julia TCI processing.

Automatically converts all .pkl files in the current directory to .h5 files.

Configuration (modify in main() function):
    x1, x2: Coordinate range (default: -21.0, 21.0)
    kill_sites: Number of sites to kill (default: 9)

Usage:
    python convert_mps_to_hdf5.py

Output:
    Creates .h5 files with the same base name as .pkl files (e.g., 7v.pkl -> 7v.h5)
"""

import argparse
import sys
import os
import pickle
import numpy as np
import h5py

# Add NumpyTensorTools to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../NumpyTensorTools')))
import qtt_tools as qtt
import npmps
import plot_utility as pltut


def convert_mps_to_hdf5(pickle_file, x1=-21.0, x2=21.0, kill_sites=0, output_file=None, store_mps=True):
    """
    Load MPS from pickle and save to HDF5.
    
    Args:
        pickle_file: Path to .pkl file containing MPS
        x1, x2: X coordinate range
        kill_sites: Number of sites to kill (0 = full resolution, 9 = 256×256)
        output_file: Output HDF5 file path (default: replaces .pkl with .h5)
        store_mps: If True, store MPS tensors directly (for on-demand evaluation).
                   If False, evaluate on grid and store (legacy mode).
    """
    print(f"Loading MPS from: {pickle_file}")
    
    # Load MPS
    with open(pickle_file, 'rb') as f:
        mps_original = pickle.load(f)
    
    # Store original MPS info
    original_N = len(mps_original) // 2
    original_Ndx = 2**original_N
    print(f"Original MPS grid size: {original_Ndx} × {original_Ndx} = {original_Ndx**2:,} points")
    
    # Process MPS (kill sites if requested)
    mps = mps_original
    if kill_sites > 0:
        print(f"Killing {kill_sites} sites...")
        for i in range(kill_sites):
            mps = qtt.kill_site_2D(mps, 80, dtype=np.complex128)
    
    # Normalize
    mps = qtt.normalize_MPS_by_integral(mps, x1, x2, Dim=2)
    
    # Get MPS parameters
    N = len(mps) // 2
    Ndx = 2**N
    rescale = (x2 - x1) / Ndx
    shift = x1
    
    print(f"MPS parameters: N = {N}, grid size = {Ndx} × {Ndx} = {Ndx**2:,} points")
    print(f"Coordinate range: x, y ∈ [{x1}, {x2}]")
    print(f"Rescale: {rescale}, Shift: {shift}")
    
    # Determine output file
    if output_file is None:
        output_file = pickle_file.replace('.pkl', '.h5')
    
    print(f"Saving MPS to: {output_file}")
    
    if store_mps:
        # Store MPS tensors directly for on-demand evaluation
        print("Storing MPS tensors (for on-demand evaluation)...")
        with h5py.File(output_file, 'w') as f:
            # Store each MPS tensor
            mps_grp = f.create_group('mps')
            for i, tensor in enumerate(mps):
                mps_grp.create_dataset(f'tensor_{i}', data=tensor, compression='gzip')
            
            # Store metadata
            f['metadata/x_range'] = [x1, x2]
            f['metadata/y_range'] = [x1, x2]
            f['metadata/kill_sites'] = kill_sites
            f['metadata/N'] = N
            f['metadata/Ndx'] = Ndx
            f['metadata/rescale'] = rescale
            f['metadata/shift'] = shift
            f['metadata/original_N'] = original_N
            f['metadata/original_Ndx'] = original_Ndx
            f['metadata/dtype'] = str(mps[0].dtype)
            
            # Store MPS dimensions for verification
            mps_dims = [tensor.shape for tensor in mps]
            f['metadata/mps_shapes'] = [str(shape) for shape in mps_dims]
        
        print(f"Done! Stored MPS with {len(mps)} tensors to {output_file}")
        print(f"MPS can be evaluated on-demand at any (x,y) point in range [{x1}, {x2}]")
    else:
        # Legacy mode: evaluate on grid and store (for backward compatibility)
        print("Legacy mode: Evaluating on grid and storing...")
        print(f"Evaluating MPS on {Ndx} × {Ndx} = {Ndx**2:,} point grid...")
        
        # Generate binary coordinate lists
        bxs = list(pltut.BinaryNumbers(N))
        bys = list(pltut.BinaryNumbers(N))
        
        # Evaluate MPS on grid
        Z = pltut.get_2D_mesh_eles_mps(mps, bxs, bys)
        Z = np.abs(Z)**2  # |psi|^2
        
        # Convert binary to decimal coordinates
        xs_list = pltut.bin_to_dec_list(bxs, rescale, shift)
        ys_list = pltut.bin_to_dec_list(bys, rescale, shift)
        
        # Create meshgrid and flatten
        X, Y = np.meshgrid(xs_list, ys_list, indexing='ij')
        xs_flat = X.flatten()
        ys_flat = Y.flatten()
        fs_flat = Z.flatten()
        
        # Save to HDF5
        with h5py.File(output_file, 'w') as f:
            f['xs'] = xs_flat
            f['ys'] = ys_flat
            f['fs'] = fs_flat
            
            # Save metadata
            f['metadata/x_range'] = [x1, x2]
            f['metadata/y_range'] = [x1, x2]
            f['metadata/kill_sites'] = kill_sites
            f['metadata/N'] = N
            f['metadata/Ndx'] = Ndx
            f['metadata/npoints'] = len(xs_flat)
            f['metadata/original_N'] = original_N
            f['metadata/original_Ndx'] = original_Ndx
        
        print(f"Done! Stored {len(xs_flat):,} pre-evaluated grid points to {output_file}")
        print(f"Data range: x ∈ [{xs_flat.min():.2f}, {xs_flat.max():.2f}], y ∈ [{ys_flat.min():.2f}, {ys_flat.max():.2f}]")
        print(f"Field range: f ∈ [{fs_flat.min():.6e}, {fs_flat.max():.6e}]")
    
    return output_file


def main():
    # Configuration - modify these values as needed
    x1 = -21.0
    x2 = 21.0
    kill_sites = 9  # 0 = full resolution (2^17 × 2^17), 9 = downsampled (256×256)
    store_mps = False  # False = pre-evaluated grid data (for restored code), True = MPS tensors
    
    # Find all .pkl files in the current directory
    script_dir = os.path.dirname(os.path.abspath(__file__))
    pkl_files = [f for f in os.listdir(script_dir) if f.endswith('.pkl')]
    
    if not pkl_files:
        print("No .pkl files found in the current directory.")
        return
    
    print(f"Found {len(pkl_files)} .pkl file(s) to convert:")
    for f in pkl_files:
        print(f"  - {f}")
    print()
    print(f"Configuration:")
    print(f"  kill_sites: {kill_sites} (0 = full resolution, 9 = 256×256)")
    print(f"  store_mps: {store_mps} (store MPS tensors for on-demand evaluation)")
    print()
    
    # Convert each .pkl file
    for pkl_file in sorted(pkl_files):
        pkl_path = os.path.join(script_dir, pkl_file)
        try:
            print("=" * 60)
            convert_mps_to_hdf5(
                pkl_path,
                x1=x1,
                x2=x2,
                kill_sites=kill_sites,
                output_file=None,  # Auto-generate output name
                store_mps=store_mps
            )
            print()
        except Exception as e:
            print(f"Error converting {pkl_file}: {e}")
            import traceback
            traceback.print_exc()
            print()
            continue
    
    print("=" * 60)
    print("Conversion complete!")


if __name__ == '__main__':
    main()

