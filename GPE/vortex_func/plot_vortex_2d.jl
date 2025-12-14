"""
Generate complete high-resolution grid from vortex data and save to HDF5.

Usage:
    julia plot_vortex_2d.jl
"""

using Pkg
import HDF5

# Helper function for meshgrid
function meshgrid(x, y)
    X = [xi for xi in x, _ in y]
    Y = [yi for _ in x, yi in y]
    return X, Y
end

# Include the vortex reader (this will load 125v.h5 and create interpolator)
include("read_vortex_h5.jl")

# Get coordinate range from metadata or use defaults
x_min, x_max = -21.0, 21.0
y_min, y_max = -21.0, 21.0
if isdefined(Main, :metadata) && haskey(metadata, "x_range")
    x_range, y_range = metadata["x_range"], metadata["y_range"]
    x_min, x_max = x_range[1], x_range[2]
    y_min, y_max = y_range[1], y_range[2]
end

# Create complete high-resolution grid (256x256 to match original resolution)
N_complete = 256
x_complete = range(x_min, x_max, length=N_complete)
y_complete = range(y_min, y_max, length=N_complete)

println("Creating complete $(N_complete)×$(N_complete) grid...")
println("Evaluating interpolator at all grid points...")

Z_complete = zeros(Float64, N_complete, N_complete)
for (i, x) in enumerate(x_complete)
    for (j, y) in enumerate(y_complete)
        Z_complete[i, j] = interpolator(x, y)
    end
    i % 50 == 0 && println("  Processed $i / $N_complete rows...")
end

# Flatten to 1D arrays for HDF5 storage
X_complete, Y_complete = meshgrid(collect(x_complete), collect(y_complete))
xs_flat, ys_flat, fs_flat = vec(X_complete), vec(Y_complete), vec(Z_complete)

output_h5 = joinpath(dirname(@__FILE__), "125v_complete.h5")
println("\nSaving complete data to: $output_h5")

HDF5.h5open(output_h5, "w") do f
    f["xs"] = xs_flat
    f["ys"] = ys_flat
    f["fs"] = fs_flat
    f["metadata/x_range"] = [x_min, x_max]
    f["metadata/y_range"] = [y_min, y_max]
    f["metadata/N"] = N_complete
    f["metadata/npoints"] = length(xs_flat)
    f["metadata/source"] = "125v.h5 (complete interpolated data)"
end

println("Done! Saved $(N_complete)×$(N_complete) = $(length(xs_flat)) points")
println("Data range: x ∈ [$x_min, $x_max], y ∈ [$y_min, $y_max]")
println("Field range: f ∈ [$(minimum(fs_flat)), $(maximum(fs_flat))]")
println("\nUse plot_vortex_2d.py to visualize the data.")

