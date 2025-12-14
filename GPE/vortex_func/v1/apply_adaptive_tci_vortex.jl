"""
Apply Original TCI and Adaptive Patched TCI to vortex data and compare results.

Reuses functions from FD/velocitymagnitudedata/apply_adaptive_tci.jl
Uses the interpolator from read_vortex_h5.jl to get vortex state data at (x,y) coordinates.
"""

using Pkg
# Only add registry if not already present (silent if exists)
try
    Pkg.Registry.add(RegistrySpec(url="https://github.com/tensor4all/T4ARegistry.git"))
catch
    # Registry already exists, continue
end
# Activate the T4AAdaptivePatchedTCI.jl project
Pkg.activate(joinpath(@__DIR__, "..", "..", "..", "FD", "T4AAdaptivePatchedTCI.jl"))
# instantiate() is fast if dependencies are already installed
Pkg.instantiate()

using Random
import QuanticsGrids as QG
import HDF5

# Make sure to load T4AITensorCompat first
# before importing other T4A packages
# to activate the ITensor-related extension
import ITensors: ITensors, Index
import T4AITensorCompat as T4AIT
import T4ATensorCI as TCI
import T4APartitionedTT: Projector
import T4AAdaptivePatchedTCI as PatchedTCI
using Statistics
using LinearAlgebra

Random.seed!(1234)

# Include the vortex reader to get the interpolator
include("read_vortex_h5.jl")

# Include shared functions from velocity magnitude file
# Main execution will be skipped when included (only runs when executed directly)
include(joinpath(@__DIR__, "..", "..", "..", "FD", "velocitymagnitudedata", "apply_adaptive_tci.jl"))

"""
Generate a grid of (x, y, f) points using the interpolator.
Creates a 2^R × 2^R grid.
"""
function generate_grid_points(interpolator, xmin, xmax, ymin, ymax, R)
    N = 2^R
    x_coords = range(xmin, xmax, length=N)
    y_coords = range(ymin, ymax, length=N)
    
    xs = Float64[]
    ys = Float64[]
    fs = Float64[]
    
    println("Generating grid points: $N × $N = $(N^2) points...")
    for (i, x) in enumerate(x_coords)
        for (j, y) in enumerate(y_coords)
            push!(xs, x)
            push!(ys, y)
            push!(fs, interpolator(x, y))
        end
        if i % 32 == 0
            println("  Processed $i / $N rows...")
        end
    end
    
    return xs, ys, fs, (collect(x_coords), collect(y_coords))
end

function create_grid_and_quantics_function(interpolator, xmin, xmax, ymin, ymax, R, ordering, fused_order)
    """Create quantics grid and quantics function based on ordering."""
    localdims = fill(2, 2R)
    
    if ordering == :fused
        grid = FusedGrid2D(R, xmin, xmax, ymin, ymax; order=fused_order)
        println("Created FusedGrid2D with order: $fused_order")
        qf = q -> interpolator(fused_quantics_to_origcoord(grid, q)...)
    else
        grid = QG.DiscretizedGrid{2}(R, (xmin, ymin), (xmax, ymax); unfoldingscheme=ordering)
        println("Created DiscretizedGrid{2} with unfoldingscheme: $ordering")
        qf = q -> interpolator(QG.quantics_to_origcoord(grid, q)...)
    end
    
    return grid, qf, localdims
end

function apply_original_tci_vortex(
    interpolator;
    xmin=-21.0,
    xmax=21.0,
    ymin=-21.0,
    ymax=21.0,
    R=8,
    tol=1e-7,
    maxbonddim=300,
    verbosity=1,
    ordering::Symbol=:fused,
    fused_order::Symbol=:xy,
)
    time_start = time()
    println("="^60)
    println("Original TCI:")
    println("="^60)
    println("Data range: x ∈ [$xmin, $xmax], y ∈ [$ymin, $ymax]")
    println("Using R = $R (grid size: $(2^R) × $(2^R))")
    println("Ordering: $ordering, fused_order: $fused_order")
    
    # Create quantics grid and quantics function
    grid, qf, localdims = create_grid_and_quantics_function(interpolator, xmin, xmax, ymin, ymax, R, ordering, fused_order)
    
    # Create projectable evaluator
    projectable = PatchedTCI.makeprojectable(Float64, qf, localdims)
    
    # Create TensorCI2 directly (original TCI, no patching)
    println("\nCreating TensorCI2 (original TCI)...")
    initialpivots = PatchedTCI.findinitialpivots(projectable, localdims, 10)
    tci = TCI.TensorCI2{Float64}(projectable, localdims, initialpivots)
    
    # Run TCI optimization
    println("Running TCI optimization...")
    ranks, errors = TCI.optimize!(
        tci,
        projectable;
        tolerance=tol,
        maxbonddim=maxbonddim,
        verbosity=verbosity,
        normalizeerror=false,
        loginterval=10,
        nsearchglobalpivot=10,
        maxiter=10,
        ncheckhistory=3,
        tolmarginglobalsearch=10.0,
    )
    
    tt = TCI.TensorTrain(tci)
    elapsed_time = time() - time_start
    
    println("\n" * "="^60)
    println("Original TCI Results:")
    println("="^60)
    @show tt
    println("Computational time: $(round(elapsed_time, digits=2)) seconds")
    
    x_coords = collect(range(xmin, xmax, length=2^R))
    y_coords = collect(range(ymin, ymax, length=2^R))
    return tt, grid, interpolator, (x_coords, y_coords), tci, elapsed_time
end

function apply_adaptive_tci_vortex(
    interpolator;
    xmin=-21.0,
    xmax=21.0,
    ymin=-21.0,
    ymax=21.0,
    R=8,
    tol=1e-7,
    maxbonddim=40,
    verbosity=1,
    ordering::Symbol=:fused,
    fused_order::Symbol=:xy,
)
    time_start = time()
    println("="^60)
    println("Adaptive Patched TCI:")
    println("="^60)
    println("Data range: x ∈ [$xmin, $xmax], y ∈ [$ymin, $ymax]")
    println("Using R = $R (grid size: $(2^R) × $(2^R))")
    println("Ordering: $ordering, fused_order: $fused_order")
    
    # Create quantics grid and quantics function
    grid, qf, localdims = create_grid_and_quantics_function(interpolator, xmin, xmax, ymin, ymax, R, ordering, fused_order)
    
    # Create patch ordering
    pordering = PatchedTCI.PatchOrdering(collect(1:2R))
    
    # Create TCI2PatchCreator
    println("\nCreating TCI2PatchCreator...")
    creator = PatchedTCI.TCI2PatchCreator(
        Float64,
        PatchedTCI.makeprojectable(Float64, qf, localdims),
        localdims,
        maxbonddim=maxbonddim,
        tolerance=tol,
        verbosity=verbosity,
        ntry=10,
    )
    
    println("\nRunning adaptive interpolation...")
    results = PatchedTCI.adaptiveinterpolate(creator, pordering; verbosity=verbosity)
    patched_tt = PatchedTCI.ProjTTContainer(results)
    elapsed_time = time() - time_start
    
    println("\n" * "="^60)
    println("Results:")
    println("="^60)
    @show patched_tt
    println("Computational time: $(round(elapsed_time, digits=2)) seconds")
    
    x_coords = collect(range(xmin, xmax, length=2^R))
    y_coords = collect(range(ymin, ymax, length=2^R))
    return patched_tt, grid, interpolator, (x_coords, y_coords), elapsed_time
end

# ============================================================================
# Main execution
# ============================================================================
# Ordering options for quantics grid representation:
#   :fused_xy  -> Fused ordering with x bits first, then y bits
#                 Layout: [x₁, x₂, ..., x_R, y₁, y₂, ..., y_R]
#                 Good for functions that vary more in x direction
#   
#   :fused_yx  -> Fused ordering with y bits first, then x bits
#                 Layout: [y₁, y₂, ..., y_R, x₁, x₂, ..., x_R]
#                 Good for functions that vary more in y direction
#   
#   :interleaved -> Interleaved ordering, alternating x and y bits
#                   Layout: [x₁, y₁, x₂, y₂, ..., x_R, y_R]
#                   Good for functions with similar variation in both directions
#
# Allowed values: :fused_xy, :fused_yx, :interleaved
order_choice = :interleaved

function order_params(choice::Symbol)
    if choice == :fused_xy
        return (:fused, :xy)
    elseif choice == :fused_yx
        return (:fused, :yx)
    elseif choice == :interleaved
        return (:interleaved, :xy) # fused_order unused in interleaved
    else
        error("Unsupported order_choice=$(choice). Use :fused_xy, :fused_yx, or :interleaved.")
    end
end

(ordering_choice, fused_order_choice) = order_params(order_choice)

if !isdefined(Main, :interpolator)
    error("Interpolator not found! Make sure read_vortex_h5.jl has been executed.")
end

# Get coordinate range from metadata or use defaults
xmin, xmax = -21.0, 21.0
ymin, ymax = -21.0, 21.0
if isdefined(Main, :metadata) && haskey(metadata, "x_range")
    x_range, y_range = metadata["x_range"], metadata["y_range"]
    xmin, xmax = x_range[1], x_range[2]
    ymin, ymax = y_range[1], y_range[2]
end

println("Applying TCI methods to vortex data (125v)")
println("="^60)
println("Using interpolator from read_vortex_h5.jl")
println("Coordinate range: x ∈ [$xmin, $xmax], y ∈ [$ymin, $ymax]")
println("Ordering choice: $order_choice")
println("  -> ordering: $ordering_choice, fused_order: $fused_order_choice")

# Apply original TCI
R = 8  # 2^8 = 256 points per dimension
original_tt, grid_orig, _, (x_coords_orig, y_coords_orig), original_tci, time_original = 
    apply_original_tci_vortex(interpolator; xmin=xmin, xmax=xmax, ymin=ymin, ymax=ymax,
                              R=R, tol=1e-7, maxbonddim=300, verbosity=1,
                              ordering=ordering_choice, fused_order=fused_order_choice)

# Apply adaptive patched TCI
patched_tt, grid_patch, _, (x_coords_patch, y_coords_patch), time_patched = 
    apply_adaptive_tci_vortex(interpolator; xmin=xmin, xmax=xmax, ymin=ymin, ymax=ymax,
                              R=R, tol=1e-7, maxbonddim=40, verbosity=1,
                              ordering=ordering_choice, fused_order=fused_order_choice)

# Compare computational time
println("\n" * "="^60)
println("Computational Time Comparison:")
println("="^60)
println("Original TCI time: $(round(time_original, digits=2)) seconds")
println("PatchedTCI time: $(round(time_patched, digits=2)) seconds")
if time_original > 0
    speedup = time_original / time_patched
    println("Speedup ratio (Original/Patched): $(round(speedup, digits=2))x")
    if speedup > 1
        println("  → PatchedTCI is $(round(speedup, digits=2))x faster")
    else
        println("  → Original TCI is $(round(1/speedup, digits=2))x faster")
    end
end

# Compare bond dimensions (reuse function from included file)
patched_bonddims, original_bonddims = compare_bond_dimensions(patched_tt, original_tt, original_tci)

# Show max bond dimension for each patch
println("\nMax bond dimension per patch:")
for (i, ptt) in enumerate(patched_tt)
    tt = ptt.data
    linkdims = TCI.linkdims(tt)
    max_bond = isempty(linkdims) ? 0 : maximum(linkdims)
    println("  Patch $i: max bond dimension = $max_bond")
end

# Generate grid points for error evaluation
println("\n" * "="^60)
println("Generating grid for error evaluation:")
println("="^60)
xs, ys, fs, (x_coords, y_coords) = generate_grid_points(interpolator, xmin, xmax, ymin, ymax, R)

println("\n" * "="^60)
println("Evaluating errors at all points for 2D maps:")
println("="^60)
errors_original_all = evaluate_all_points(original_tt, grid_orig, xs, ys, fs; method_name="Original TCI")
errors_patched_all = evaluate_all_points(patched_tt, grid_patch, xs, ys, fs; method_name="PatchedTCI")
patch_bounds = collect_patch_bounds(patched_tt, xmin, xmax, ymin, ymax, R;
                                    ordering=ordering_choice, fused_order=fused_order_choice)

# Wrapper to save errors with correct path (shared function uses @__DIR__ from its own file)
function save_errors_to_hdf5_vortex(errors_original, errors_patched, xs, ys, fs,
                                    time_original, time_patched, patched_bonddims, original_bonddims,
                                    output_file::String; tol=1e-7, maxbonddim=30, R=8, patch_bounds=nothing)
    output_path = joinpath(@__DIR__, output_file)
    HDF5.h5open(output_path, "w") do f
        f["xs"] = xs
        f["ys"] = ys
        f["fs"] = fs
        f["errors_original_tci"] = errors_original
        f["errors_patched_tci"] = errors_patched
        f["time_original_tci"] = time_original
        f["time_patched_tci"] = time_patched
        f["bonddims_original_tci"] = original_bonddims
        f["bonddims_patched_tci"] = patched_bonddims
        if patch_bounds !== nothing
            f["patch_bounds"] = patch_bounds
        end
        f["metadata/tolerance"] = tol
        f["metadata/maxbonddim"] = maxbonddim
        f["metadata/R"] = R
        f["metadata/npoints"] = length(xs)
        f["stats_original/mean_error"] = mean(errors_original)
        f["stats_original/median_error"] = median(errors_original)
        f["stats_original/max_error"] = maximum(errors_original)
        f["stats_original/std_error"] = std(errors_original)
        f["stats_patched/mean_error"] = mean(errors_patched)
        f["stats_patched/median_error"] = median(errors_patched)
        f["stats_patched/max_error"] = maximum(errors_patched)
        f["stats_patched/std_error"] = std(errors_patched)
    end
    println("✓ Errors saved to: $output_path")
end

println("\n" * "="^60)
println("Saving errors to file for Python plotting...")
println("="^60)
output_file = "125v_errors.h5"
save_errors_to_hdf5_vortex(errors_original_all, errors_patched_all, xs, ys, fs,
                           time_original, time_patched, patched_bonddims, original_bonddims,
                           output_file; tol=1e-7, maxbonddim=120, R=R, patch_bounds=patch_bounds)

println("\n" * "="^60)
println("Summary:")
println("="^60)
println("Errors saved to: $output_file")
println("\nTo plot in Python, run:")
println("  python plot_errors_from_file.py $output_file")

println("\nDone!")
