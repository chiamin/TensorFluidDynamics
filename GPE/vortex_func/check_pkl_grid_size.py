"""
Check the original grid size of pkl files.

This script loads a pkl file and checks:
1. The original MPS length (before any processing)
2. The grid size that would result from the original MPS
3. The grid size after killing sites (as done in plot2D_sec.py and convert_mps_to_hdf5.py)
"""

import sys
import os
import pickle
import numpy as np

# Add NumpyTensorTools to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../NumpyTensorTools')))
import qtt_tools as qtt
import npmps


def check_pkl_grid_size(pkl_file, kill_sites=9):
    """
    Check the grid size of a pkl file.
    
    Args:
        pkl_file: Path to .pkl file
        kill_sites: Number of sites to kill (default: 9, as in plot2D_sec.py)
    """
    print(f"=" * 70)
    print(f"Checking grid size for: {pkl_file}")
    print(f"=" * 70)
    
    # Load original MPS
    with open(pkl_file, 'rb') as f:
        mps_original = pickle.load(f)
    
    # Check original MPS structure
    original_length = len(mps_original)
    print(f"\nOriginal MPS (before processing):")
    print(f"  MPS length: {original_length}")
    print(f"  N = len(mps) // 2 = {original_length // 2}")
    print(f"  Grid size would be: 2^{original_length // 2} × 2^{original_length // 2} = {2**(original_length // 2)} × {2**(original_length // 2)}")
    print(f"  Total points: {2**(original_length // 2) * 2**(original_length // 2):,}")
    
    # Check MPS dimensions
    print(f"\n  MPS dimensions: {npmps.MPS_dims(mps_original)}")
    
    # Now process as in plot2D_sec.py
    print(f"\nProcessing MPS (killing {kill_sites} sites, as in plot2D_sec.py):")
    mps_processed = mps_original
    for i in range(kill_sites):
        mps_processed = qtt.kill_site_2D(mps_processed, 80, dtype=np.complex128)
    
    processed_length = len(mps_processed)
    print(f"  MPS length after killing sites: {processed_length}")
    print(f"  N = len(mps) // 2 = {processed_length // 2}")
    print(f"  Grid size: 2^{processed_length // 2} × 2^{processed_length // 2} = {2**(processed_length // 2)} × {2**(processed_length // 2)}")
    print(f"  Total points: {2**(processed_length // 2) * 2**(processed_length // 2):,}")
    
    print(f"\n  Processed MPS dimensions: {npmps.MPS_dims(mps_processed)}")
    
    # Summary
    print(f"\n" + "=" * 70)
    print(f"SUMMARY:")
    print(f"  Original grid size: {2**(original_length // 2)} × {2**(original_length // 2)}")
    print(f"  Processed grid size: {2**(processed_length // 2)} × {2**(processed_length // 2)}")
    print(f"  Reduction factor: {2**(original_length // 2) / 2**(processed_length // 2):.1f}x")
    print(f"=" * 70)
    
    return {
        'original_length': original_length,
        'original_N': original_length // 2,
        'original_grid_size': 2**(original_length // 2),
        'processed_length': processed_length,
        'processed_N': processed_length // 2,
        'processed_grid_size': 2**(processed_length // 2),
    }


def main():
    # Check one of the pkl files (they should all have the same structure)
    script_dir = os.path.dirname(os.path.abspath(__file__))
    pkl_files = [f for f in os.listdir(script_dir) if f.endswith('.pkl')]
    
    if not pkl_files:
        print("No .pkl files found in the directory.")
        return
    
    # Use the first pkl file found (or 125v.pkl if it exists)
    if '125v.pkl' in pkl_files:
        pkl_file = '125v.pkl'
    else:
        pkl_file = sorted(pkl_files)[0]
    
    pkl_path = os.path.join(script_dir, pkl_file)
    
    print(f"Checking pkl file: {pkl_file}")
    print(f"(All pkl files should have the same grid size)\n")
    
    try:
        result = check_pkl_grid_size(pkl_path, kill_sites=9)
        
        print(f"\n" + "=" * 70)
        print(f"CONCLUSION:")
        if result['processed_grid_size'] == 256:
            print(f"  The processed data (after killing 9 sites) has grid size 256×256.")
            print(f"  The original data has grid size {result['original_grid_size']}×{result['original_grid_size']}.")
            print(f"  So plot2D_sec.py uses the processed (downsampled) data, not the original.")
        else:
            print(f"  The processed data has grid size {result['processed_grid_size']}×{result['processed_grid_size']}.")
            print(f"  The original data has grid size {result['original_grid_size']}×{result['original_grid_size']}.")
        print(f"=" * 70)
        
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == '__main__':
    main()

