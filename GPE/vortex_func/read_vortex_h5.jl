"""
Read vortex HDF5 files and create interpolator functions.

When run as a script, automatically loads "125v.h5" and creates an interpolator function.

Usage:
    julia read_vortex_h5.jl

Or use as a module:
    using .VortexReader
    f, metadata = load_vortex_interpolator("125v.h5")
    value = f(0.5, 0.3)
"""

using Pkg
import HDF5
using LinearAlgebra

# Module for reading vortex HDF5 files
module VortexReader
    using HDF5
    using LinearAlgebra
    
    """
    Load vortex data from HDF5 file and create an interpolator function.
    
    Args:
        h5file: Path to HDF5 file (e.g., "7v.h5")
    
    Returns:
        interpolator: Function (x, y) -> Float64 that returns the field value at (x, y)
        metadata: Dict with metadata (x_range, y_range, N, Ndx, etc.)
    """
    function load_vortex_interpolator(h5file::String)
        println("Loading vortex data from: $h5file")
        
        HDF5.h5open(h5file, "r") do f
            # Check if file has pre-evaluated grid data or MPS tensors
            if haskey(f, "xs") && haskey(f, "ys") && haskey(f, "fs")
                # Pre-evaluated grid data format
                xs = read(f, "xs")
                ys = read(f, "ys")
                fs = read(f, "fs")
                
                # Read metadata
            metadata = Dict{String, Any}()
            if haskey(f, "metadata")
                metadata["x_range"] = read(f, "metadata/x_range")
                metadata["y_range"] = read(f, "metadata/y_range")
                if haskey(f, "metadata/kill_sites")
                    metadata["kill_sites"] = read(f, "metadata/kill_sites")
                end
                metadata["N"] = read(f, "metadata/N")
                metadata["Ndx"] = read(f, "metadata/Ndx")
                metadata["npoints"] = read(f, "metadata/npoints")
                if haskey(f, "metadata/original_N")
                    metadata["original_N"] = read(f, "metadata/original_N")
                    metadata["original_Ndx"] = read(f, "metadata/original_Ndx")
                end
            end
            
                println("  Loaded $(length(xs)) points")
                println("  Data range: x ∈ [$(minimum(xs)), $(maximum(xs))], y ∈ [$(minimum(ys)), $(maximum(ys))]")
                println("  Field range: f ∈ [$(minimum(fs)), $(maximum(fs))]")
                
                # Create interpolator
                interpolator = create_interpolator(xs, ys, fs)
                
                return interpolator, metadata
            else
                error("HDF5 file must contain pre-evaluated grid data (xs, ys, fs).\nPlease regenerate using convert_mps_to_hdf5.py with store_mps=False")
            end
        end
    end
    
    """
    Create an interpolator function from coordinate and field data.
    Uses nearest-neighbor interpolation for fast lookup.
    """
    function create_interpolator(xs, ys, fs)
        # Get unique sorted coordinates
        x_coords = sort(unique(xs))
        y_coords = sort(unique(ys))
        
        # Get grid dimensions
        Nx = length(x_coords)
        Ny = length(y_coords)
        
        # Reshape data to 2D grid
        fs_2d = reshape(fs, (Nx, Ny))
        
        # Create coordinate index mappings (faster lookup)
        x_to_idx = Dict{Float64, Int}(x => i for (i, x) in enumerate(x_coords))
        y_to_idx = Dict{Float64, Int}(y => i for (i, y) in enumerate(y_coords))
        
        # Create interpolator function
        function interpolate(x::Float64, y::Float64)::Float64
            # Fast lookup if exact match exists
            if haskey(x_to_idx, x) && haskey(y_to_idx, y)
                return fs_2d[x_to_idx[x], y_to_idx[y]]
            end
            
            # Find nearest neighbor using binary search
            x_idx = searchsortednearest(x_coords, x)
            y_idx = searchsortednearest(y_coords, y)
            return fs_2d[x_idx, y_idx]
        end
        
        return interpolate
    end
    
    """
    Binary search to find nearest coordinate index.
    """
    function searchsortednearest(a::Vector{Float64}, x::Float64)::Int
        idx = searchsortedfirst(a, x)
        idx == 1 && return 1
        idx > length(a) && return length(a)
        return abs(a[idx] - x) < abs(a[idx-1] - x) ? idx : idx - 1
    end
    
    """
    Load all vortex HDF5 files in a directory and return a dictionary of interpolators.
    
    Args:
        dir: Directory path (default: current directory)
        pattern: File pattern to match (default: "*v.h5")
    
    Returns:
        Dict mapping filename (without extension) to (interpolator, metadata) tuple
    """
    function load_all_vortex_files(dir::String="."; pattern::String="*v.h5")
        files = filter(f -> endswith(f, ".h5"), readdir(dir))
        vortex_files = filter(f -> occursin(r"\d+v\.h5$", f), files)
        
        result = Dict{String, Tuple{Function, Dict{String, Any}}}()
        
        for file in vortex_files
            filepath = joinpath(dir, file)
            try
                interpolator, metadata = load_vortex_interpolator(filepath)
                # Extract number of vortices from filename (e.g., "7v.h5" -> "7v")
                basename_no_ext = splitext(file)[1]
                result[basename_no_ext] = (interpolator, metadata)
            catch e
                println("Warning: Failed to load $file: $e")
            end
        end
        
        return result
    end
end

using .VortexReader

# Main execution - Load 125v.h5 directly
h5file = "125v.h5"

# Get the directory where this script is located
script_dir = dirname(@__FILE__)
h5file_path = joinpath(script_dir, h5file)

if isfile(h5file_path)
    interpolator, metadata = VortexReader.load_vortex_interpolator(h5file_path)
    
    println("\n" * "="^60)
    println("Interpolator created successfully!")
    println("="^60)
    println("\nTesting the interpolator at a few points:")
    
    # Test at a few points
    test_points = [(0.0, 0.0), (5.0, 5.0), (-10.0, 10.0), (15.0, -15.0)]
    for (x, y) in test_points
        value = interpolator(x, y)
        println("  f($x, $y) = $value")
    end
    
    println("\nThe interpolator function is available as 'interpolator'")
    println("You can use it like: value = interpolator(x, y)")
else
    println("Error: File not found: $h5file_path")
    println("\nAvailable files in directory:")
    if isdir(script_dir)
        h5_files = filter(f -> endswith(f, ".h5"), readdir(script_dir))
        for f in h5_files
            println("  - $f")
        end
    end
end
