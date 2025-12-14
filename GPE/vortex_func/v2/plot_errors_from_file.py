"""
Python script to read and plot errors saved from Julia TCI analysis for vortex data.

Reuses functions from FD/velocitymagnitudedata/plot_errors_from_file.py

Usage:
    python plot_errors_from_file.py [errors_file.h5]
"""

import sys
import os
import argparse
import importlib.util

# Import from the shared velocity magnitude plotting module
shared_module_path = os.path.join(os.path.dirname(__file__), "..", "..", "..", "FD", "velocitymagnitudedata", "plot_errors_from_file.py")
spec = importlib.util.spec_from_file_location("plot_errors_shared", shared_module_path)
plot_errors_shared = importlib.util.module_from_spec(spec)
spec.loader.exec_module(plot_errors_shared)

read_errors_file = plot_errors_shared.read_errors_file
plot_data_with_patches = plot_errors_shared.plot_data_with_patches
plot_2d_error_maps = plot_errors_shared.plot_2d_error_maps

import matplotlib.pyplot as plt

def main():
    parser = argparse.ArgumentParser(description="Plot TCI errors for vortex data")
    parser.add_argument("h5file", nargs="?", default="125v_errors.h5",
                       help="HDF5 file containing error data (default: 125v_errors.h5)")
    args = parser.parse_args()
    
    if not os.path.exists(args.h5file):
        print(f"Error: File '{args.h5file}' not found!")
        sys.exit(1)
    
    print(f"Reading errors from: {args.h5file}")
    data = read_errors_file(args.h5file)
    
    print(f"\nData loaded:")
    print(f"  Points: {data['npoints']}")
    print(f"  Tolerance: {data['tolerance']}")
    print(f"  Max bond dimension: {data['maxbonddim']}")
    print(f"  R: {data['R']}")
    
    base_name = os.path.splitext(args.h5file)[0]
    plot_2d_error_maps(data, f"{base_name}_2d_maps.png")
    plot_data_with_patches(data, f"{base_name}_data_with_patches.png")
    plt.show()
    print("\nDone!")

if __name__ == "__main__":
    main()
