using Pkg
# Only add registry if not already present (silent if exists)
try
    Pkg.Registry.add(RegistrySpec(url="https://github.com/tensor4all/T4ARegistry.git"))
catch
    # Registry already exists, continue
end
# Activate the T4AAdaptivePatchedTCI.jl project
Pkg.activate(joinpath(@__DIR__, "..", "T4AAdaptivePatchedTCI.jl"))
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
import T4ATensorCI
import T4AQuanticsTCI as QTCI
import T4APartitionedTT as T4AP
import T4APartitionedTT: PartitionedTT, SubDomainTT, adaptive_patching, Projector
import T4AAdaptivePatchedTCI as PatchedTCI
using CairoMakie
using Statistics
using LinearAlgebra

Random.seed!(1234)

# -------------------------
# Fused 2D grid helper (x bits then y bits)
# -------------------------
struct FusedGrid2D
    R::Int
    xgrid
    ygrid
    order::Symbol # :xy (x bits then y bits) or :yx (y bits then x bits)
end

function FusedGrid2D(R::Int, xmin::Real, xmax::Real, ymin::Real, ymax::Real; order::Symbol=:xy)
    order ∈ (:xy, :yx) || throw(ArgumentError("fused order must be :xy or :yx, got $order"))
    xg = QG.DiscretizedGrid{1}(R, xmin, xmax)
    yg = QG.DiscretizedGrid{1}(R, ymin, ymax)
    return FusedGrid2D(R, xg, yg, order)
end

"""
Convert fused quantics bits (length 2R, values in 1:2) to original (x,y).
Layout: [x₁..x_R, y₁..y_R].
"""
function fused_quantics_to_origcoord(g::FusedGrid2D, q::AbstractVector{<:Integer})
    length(q) == 2g.R || throw(ArgumentError("Expected quantics length $(2g.R), got $(length(q))"))
    if g.order == :xy
        qx = q[1:g.R]
        qy = q[(g.R + 1):(2g.R)]
    else
        qy = q[1:g.R]
        qx = q[(g.R + 1):(2g.R)]
    end
    x = QG.quantics_to_origcoord(g.xgrid, qx)[1]
    y = QG.quantics_to_origcoord(g.ygrid, qy)[1]
    return (x, y)
end

"""
Convert original (x,y) to fused quantics bits (length 2R, values in 1:2).
Layout: [x₁..x_R, y₁..y_R].
"""
function origcoord_to_fused_quantics(g::FusedGrid2D, xy::Tuple{<:Real,<:Real})
    qx = QG.origcoord_to_quantics(g.xgrid, xy[1])
    qy = QG.origcoord_to_quantics(g.ygrid, xy[2])
    return g.order == :xy ? vcat(qx, qy) : vcat(qy, qx)
end

# -------------------------
# Load velocity magnitude data from HDF5
# -------------------------
function load_velocity_data(h5file::String)
    println("Loading data from $h5file...")
    HDF5.h5open(h5file, "r") do f
        xs = read(f, "xs")
        ys = read(f, "ys")
        fs = read(f, "fs")
        return xs, ys, fs
    end
end

# -------------------------
# Create interpolator function from data (optimized)
# -------------------------
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
    
    # Precompute step sizes for binary search optimization
    x_step = length(x_coords) > 1 ? (x_coords[end] - x_coords[1]) / (length(x_coords) - 1) : 1.0
    y_step = length(y_coords) > 1 ? (y_coords[end] - y_coords[1]) / (length(y_coords) - 1) : 1.0
    
    # Optimized interpolation function using binary search
    function interpolate(x, y)
        # Fast lookup if exact match exists
        if haskey(x_to_idx, x) && haskey(y_to_idx, y)
            return fs_2d[x_to_idx[x], y_to_idx[y]]
        end
        
        # Binary search for nearest x coordinate
        x_idx = searchsortednearest(x_coords, x)
        y_idx = searchsortednearest(y_coords, y)
        return fs_2d[x_idx, y_idx]
    end
    
    return interpolate, x_coords, y_coords, (Nx, Ny)
end

# Helper function for binary search (optimized)
function searchsortednearest(a::Vector, x::Real)
    idx = searchsortedfirst(a, x)
    if idx == 1
        return 1
    elseif idx > length(a)
        return length(a)
    else
        # Check which is closer
        return abs(a[idx] - x) < abs(a[idx-1] - x) ? idx : idx - 1
    end
end

# -------------------------
# Apply original TCI (non-patched)
# -------------------------
function apply_original_tci(
    h5file::String;
    R=8,
    tol=1e-7,
    maxbonddim=30,
    verbosity=1,
    ordering::Symbol=:fused,
    fused_order::Symbol=:xy,
)
    # Start timing
    time_start = time()
    
    # Load data
    xs, ys, fs = load_velocity_data(h5file)
    
    # Get coordinate ranges
    xmin, xmax = extrema(xs)
    ymin, ymax = extrema(ys)
    
    println("="^60)
    println("Original TCI:")
    println("="^60)
    println("Data range: x ∈ [$xmin, $xmax], y ∈ [$ymin, $ymax]")
    println("Data points: $(length(xs))")
    
    # Create interpolator
    interpolator, x_coords, y_coords, (Nx, Ny) = create_interpolator(xs, ys, fs)
    println("Grid dimensions: $Nx × $Ny")
    println("Using R = $R (grid size: $(2^R) × $(2^R))")
    
    # Create quantics grid representation and localdims.
    # For the "fused" layout we use 2R binary sites: [x bits..., y bits...].
    # We represent the (x,y) mapping via two separate 1D grids to avoid QuanticsGrids'
    # 2D base-4 (length-R) representation mismatch.
    localdims = fill(2, 2R)
    grid = ordering == :fused ? FusedGrid2D(R, xmin, xmax, ymin, ymax; order=fused_order) :
           QG.DiscretizedGrid{2}(R, (xmin, ymin), (xmax, ymax); unfoldingscheme=ordering)
    
    # Convert interpolator to quantics coordinates
    qf = if ordering == :fused
        q -> interpolator(fused_quantics_to_origcoord(grid, q)...)
    else
        q -> interpolator(QG.quantics_to_origcoord(grid, q)...)
    end
    
    # Create projectable evaluator
    projectable = PatchedTCI.makeprojectable(Float64, qf, localdims)
    
    # Create TensorCI2 directly (original TCI, no patching)
    println("\nCreating TensorCI2 (original TCI)...")
    # Find initial pivots (similar to what TCI2PatchCreator does)
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
    
    # Convert to TensorTrain
    tt = TCI.TensorTrain(tci)
    
    # End timing
    elapsed_time = time() - time_start
    
    println("\n" * "="^60)
    println("Original TCI Results:")
    println("="^60)
    @show tt
    println("Computational time: $(round(elapsed_time, digits=2)) seconds")
    
    return tt, grid, interpolator, (x_coords, y_coords), tci, elapsed_time
end

# -------------------------
# Main function to apply adaptive TCI
# -------------------------
function apply_adaptive_tci(
    h5file::String;
    R=8,
    tol=1e-7,
    maxbonddim=30,
    verbosity=1,
    ordering::Symbol=:fused,
    fused_order::Symbol=:xy,
)
    # Start timing
    time_start = time()
    
    # Load data
    xs, ys, fs = load_velocity_data(h5file)
    
    # Get coordinate ranges
    xmin, xmax = extrema(xs)
    ymin, ymax = extrema(ys)
    
    println("="^60)
    println("Adaptive Patched TCI:")
    println("="^60)
    println("Data range: x ∈ [$xmin, $xmax], y ∈ [$ymin, $ymax]")
    println("Data points: $(length(xs))")
    
    # Create interpolator
    interpolator, x_coords, y_coords, (Nx, Ny) = create_interpolator(xs, ys, fs)
    println("Grid dimensions: $Nx × $Ny")
    println("Using R = $R (grid size: $(2^R) × $(2^R))")
    
    # Create quantics grid representation and localdims (see apply_original_tci)
    localdims = fill(2, 2R)
    grid = ordering == :fused ? FusedGrid2D(R, xmin, xmax, ymin, ymax; order=fused_order) :
           QG.DiscretizedGrid{2}(R, (xmin, ymin), (xmax, ymax); unfoldingscheme=ordering)
    
    # Convert interpolator to quantics coordinates
    qf = if ordering == :fused
        q -> interpolator(fused_quantics_to_origcoord(grid, q)...)
    else
        q -> interpolator(QG.quantics_to_origcoord(grid, q)...)
    end
    
    # Create patch ordering (2R sites; for fused: all x bits then y bits)
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
    
    # Run adaptive interpolation
    println("\nRunning adaptive interpolation...")
    results = PatchedTCI.adaptiveinterpolate(creator, pordering; verbosity=verbosity)
    patched_tt = PatchedTCI.ProjTTContainer(results)
    
    # End timing
    elapsed_time = time() - time_start
    
    println("\n" * "="^60)
    println("Results:")
    println("="^60)
    @show patched_tt
    println("Computational time: $(round(elapsed_time, digits=2)) seconds")
    
    return patched_tt, grid, interpolator, (x_coords, y_coords), elapsed_time
end

# -------------------------
# Compare bond dimensions
# -------------------------
function compare_bond_dimensions(patched_tt, original_tt, original_tci)
    println("\n" * "="^60)
    println("Bond Dimension Comparison:")
    println("="^60)
    
    # Get bond dimensions from patched TCI
    patched_bonddims = Int[]
    for ptt in patched_tt
        tt = ptt.data
        linkdims = TCI.linkdims(tt)
        append!(patched_bonddims, linkdims)
    end
    
    # Get bond dimensions from original TCI
    original_linkdims = TCI.linkdims(original_tt)
    
    println("\nAdaptive Patched TCI:")
    println("  Number of patches: $(length(patched_tt))")
    println("  Total bond dimensions across all patches: $(length(patched_bonddims))")
    println("  Max bond dimension: $(isempty(patched_bonddims) ? 0 : maximum(patched_bonddims))")
    println("  Mean bond dimension: $(isempty(patched_bonddims) ? 0 : mean(patched_bonddims))")
    println("  Median bond dimension: $(isempty(patched_bonddims) ? 0 : median(patched_bonddims))")
    
    println("\nOriginal TCI:")
    println("  Number of bonds: $(length(original_linkdims))")
    println("  Max bond dimension: $(isempty(original_linkdims) ? 0 : maximum(original_linkdims))")
    println("  Mean bond dimension: $(isempty(original_linkdims) ? 0 : mean(original_linkdims))")
    println("  Median bond dimension: $(isempty(original_linkdims) ? 0 : median(original_linkdims))")
    
    # Compare
    if !isempty(patched_bonddims) && !isempty(original_linkdims)
        println("\nComparison:")
        println("  Patched max / Original max: $(maximum(patched_bonddims)) / $(maximum(original_linkdims)) = $(round(maximum(patched_bonddims) / maximum(original_linkdims), digits=2))")
        println("  Patched mean / Original mean: $(round(mean(patched_bonddims), digits=2)) / $(round(mean(original_linkdims), digits=2)) = $(round(mean(patched_bonddims) / mean(original_linkdims), digits=2))")
    end
    
    return patched_bonddims, original_linkdims
end

# -------------------------
# Evaluate errors at all points for 2D plotting (optimized)
# -------------------------
function evaluate_all_points(tt, grid, xs, ys, fs; method_name="TCI")
    println("\nEvaluating errors at all points for 2D plot ($method_name)...")
    
    n_total = length(xs)
    # Preallocate array for better performance
    errors = Vector{Float64}(undef, n_total)
    
    is_patched = (method_name == "PatchedTCI")
    
    @inbounds for idx in 1:n_total
        x, y = xs[idx], ys[idx]
        f_true = fs[idx]
        
        # Convert to quantics and evaluate
        quantics_coord = grid isa FusedGrid2D ? origcoord_to_fused_quantics(grid, (x, y)) :
                        QG.origcoord_to_quantics(grid, (x, y))
        
        if is_patched
            # For patched TCI, use nested format
            nested_coord = [[q] for q in quantics_coord]
            f_approx = tt(nested_coord)
        else
            # For original TCI, use MultiIndex format
            multi_idx = TCI.MultiIndex(quantics_coord)
            f_approx = TCI.evaluate(tt, multi_idx)
        end
        
        errors[idx] = abs(f_approx - f_true)
        
        # Progress indicator (less frequent for performance)
        if idx % 10000 == 0
            println("  Processed $idx / $n_total points...")
        end
    end
    
    println("  Completed evaluation at all $n_total points")
    
    return errors
end

# -------------------------
# Evaluate and compare (optimized)
# -------------------------
function evaluate_and_compare(tt, grid, interpolator, x_coords, y_coords, xs, ys, fs; n_samples=1000, method_name="TCI")
    println("\nEvaluating approximation quality ($method_name)...")
    
    # Sample random points for comparison
    n = min(n_samples, length(xs))
    indices = rand(1:length(xs), n)
    
    # Preallocate arrays
    errors = Vector{Float64}(undef, n)
    sample_xs = Vector{Float64}(undef, n)
    sample_ys = Vector{Float64}(undef, n)
    sample_fs = Vector{Float64}(undef, n)
    
    is_patched = (method_name == "PatchedTCI")
    
    @inbounds for (i, idx) in enumerate(indices)
        x, y = xs[idx], ys[idx]
        f_true = fs[idx]
        
        # Store sample coordinates
        sample_xs[i] = x
        sample_ys[i] = y
        sample_fs[i] = f_true
        
        # Convert to quantics and evaluate
        quantics_coord = grid isa FusedGrid2D ? origcoord_to_fused_quantics(grid, (x, y)) :
                        QG.origcoord_to_quantics(grid, (x, y))
        
        if is_patched
            # For patched TCI, use nested format
            nested_coord = [[q] for q in quantics_coord]
            f_approx = tt(nested_coord)
        else
            # For original TCI, use MultiIndex format
            multi_idx = TCI.MultiIndex(quantics_coord)
            f_approx = TCI.evaluate(tt, multi_idx)
        end
        
        errors[i] = abs(f_approx - f_true)
    end
    
    println("Mean absolute error: $(mean(errors))")
    println("Max absolute error: $(maximum(errors))")
    println("Median absolute error: $(median(errors))")
    
    return errors, sample_xs, sample_ys, sample_fs
end

# -------------------------
# Save errors to HDF5 file
# -------------------------
function save_errors_to_hdf5(errors_original, errors_patched, xs, ys, fs,
                             time_original, time_patched, patched_bonddims, original_bonddims,
                             h5file::String; tol=1e-7, maxbonddim=30, R=8, patch_bounds=nothing)
    # Create output filename
    output_file = replace(h5file, ".h5" => "_errors.h5")
    output_path = joinpath(@__DIR__, output_file)
    
    println("Saving errors to: $output_path")
    
    HDF5.h5open(output_path, "w") do f
        # Save coordinates and true values
        f["xs"] = xs
        f["ys"] = ys
        f["fs"] = fs  # True function values
        
        # Save errors
        f["errors_original_tci"] = errors_original
        f["errors_patched_tci"] = errors_patched
        
        # Save timing information
        f["time_original_tci"] = time_original
        f["time_patched_tci"] = time_patched
        
        # Save bond dimensions
        f["bonddims_original_tci"] = original_bonddims
        f["bonddims_patched_tci"] = patched_bonddims

        # Save patch bounds if available (Nx4: x_lower, x_upper, y_lower, y_upper)
        if patch_bounds !== nothing
            f["patch_bounds"] = patch_bounds
        end
        
        # Save metadata
        f["metadata/tolerance"] = tol
        f["metadata/maxbonddim"] = maxbonddim
        f["metadata/R"] = R
        f["metadata/npoints"] = length(xs)
        
        # Save statistics
        f["stats_original/mean_error"] = mean(errors_original)
        f["stats_original/median_error"] = median(errors_original)
        f["stats_original/max_error"] = maximum(errors_original)
        f["stats_original/std_error"] = std(errors_original)
        
        f["stats_patched/mean_error"] = mean(errors_patched)
        f["stats_patched/median_error"] = median(errors_patched)
        f["stats_patched/max_error"] = maximum(errors_patched)
        f["stats_patched/std_error"] = std(errors_patched)
    end
    
    println("✓ Errors saved successfully!")
    println("  File: $output_path")
    println("\nTo read in Python:")
    println("  import h5py")
    println("  with h5py.File('$output_file', 'r') as f:")
    println("      errors_orig = f['errors_original_tci'][:]")
    println("      errors_patch = f['errors_patched_tci'][:]")
    println("      xs = f['xs'][:]")
    println("      ys = f['ys'][:]")
end

# -------------------------
# Patch bounds utilities (for plotting patches)
# -------------------------
function bound_idx(p::PatchedTCI.Projector, grid1d, sitedims)::Tuple{Float64,Float64}
    # p.data is Vector{Vector{Int}}, where p.data[isite][ilegg] is the projection value
    # 0 means no projection, >0 means projected to that value
    lower_idx = TCI.MultiIndex(undef, sum(length.(sitedims)))
    upper_idx = TCI.MultiIndex(undef, sum(length.(sitedims)))

    idx_pos = 1
    for (isite, site_dim) in enumerate(sitedims)
        for (ilegg, dim) in enumerate(site_dim)
            proj_val = p.data[isite][ilegg]
            if proj_val == 0
                lower_idx[idx_pos] = 1
                upper_idx[idx_pos] = dim
            else
                lower_idx[idx_pos] = proj_val
                upper_idx[idx_pos] = proj_val
            end
            idx_pos += 1
        end
    end

    lower_coord = QG.quantics_to_origcoord(grid1d, lower_idx)[1]
    upper_coord = QG.quantics_to_origcoord(grid1d, upper_idx)[1]
    return (lower_coord, upper_coord)
end

function patch_bound(
    p::PatchedTCI.Projector,
    xmin,
    xmax,
    ymin,
    ymax,
    R;
    ordering::Symbol=:fused,
    fused_order::Symbol=:xy,
)::NTuple{4, Float64}
    xgrid = QG.DiscretizedGrid{1}(R, xmin, xmax)
    ygrid = QG.DiscretizedGrid{1}(R, ymin, ymax)

    x_data = Vector{Int}[]
    x_sitedims = Vector{Int}[]
    y_data = Vector{Int}[]
    y_sitedims = Vector{Int}[]

    if ordering == :interleaved
        # x and y bits are interleaved across sites
        for isite in 1:length(p.data)
            if isodd(isite)
                push!(x_data, p.data[isite])
                push!(x_sitedims, p.sitedims[isite])
            else
                push!(y_data, p.data[isite])
                push!(y_sitedims, p.sitedims[isite])
            end
        end
    elseif ordering == :fused
        # fused layout: either [x bits..., y bits...] or [y bits..., x bits...]
        length(p.data) == 2R || error("Expected projector length $(2R), got $(length(p.data))")
        if fused_order == :xy
            for isite in 1:R
                push!(x_data, p.data[isite])
                push!(x_sitedims, p.sitedims[isite])
            end
            for isite in (R + 1):(2R)
                push!(y_data, p.data[isite])
                push!(y_sitedims, p.sitedims[isite])
            end
        elseif fused_order == :yx
            for isite in 1:R
                push!(y_data, p.data[isite])
                push!(y_sitedims, p.sitedims[isite])
            end
            for isite in (R + 1):(2R)
                push!(x_data, p.data[isite])
                push!(x_sitedims, p.sitedims[isite])
            end
        else
            error("Unsupported fused_order=$fused_order")
        end
    else
        error("Unsupported ordering=$ordering for patch bounds")
    end

    x_proj = PatchedTCI.Projector(x_data, x_sitedims)
    y_proj = PatchedTCI.Projector(y_data, y_sitedims)

    x_lower, x_upper = bound_idx(x_proj, xgrid, x_sitedims)
    y_lower, y_upper = bound_idx(y_proj, ygrid, y_sitedims)

    return (x_lower, x_upper, y_lower, y_upper)
end

function collect_patch_bounds(patched_tt, xmin, xmax, ymin, ymax, R; ordering::Symbol=:fused, fused_order::Symbol=:xy)
    bounds = Matrix{Float64}(undef, length(patched_tt), 4)
    for (i, ptt) in enumerate(patched_tt)
        b = patch_bound(ptt.projector, xmin, xmax, ymin, ymax, R; ordering=ordering, fused_order=fused_order)
        bounds[i, :] = collect(b)
    end
    return bounds
end

# -------------------------
# Plot 2D error maps
# -------------------------
function plot_2d_error_maps(errors_original, errors_patched, xs, ys, h5file::String, 
                          method_name_orig="Original TCI", method_name_patch="PatchedTCI")
    println("\n" * "="^60)
    println("Creating 2D error maps...")
    println("="^60)
    
    # Reshape to 2D grid for plotting (optimized)
    x_coords = sort(unique(xs))
    y_coords = sort(unique(ys))
    Nx = length(x_coords)
    Ny = length(y_coords)
    
    # Create coordinate mappings (faster lookup with typed Dict)
    x_to_idx = Dict{Float64, Int}(x => i for (i, x) in enumerate(x_coords))
    y_to_idx = Dict{Float64, Int}(y => i for (i, y) in enumerate(y_coords))
    
    # Reshape errors to 2D (preallocated)
    errors_orig_2d = zeros(Nx, Ny)
    errors_patch_2d = zeros(Nx, Ny)
    
    @inbounds for i in 1:length(xs)
        x_idx = x_to_idx[xs[i]]
        y_idx = y_to_idx[ys[i]]
        errors_orig_2d[x_idx, y_idx] = errors_original[i]
        errors_patch_2d[x_idx, y_idx] = errors_patched[i]
    end
    
    # Create coordinate grids (for reference, not used in heatmap but available if needed)
    # X, Y = [x for x in x_coords, y in y_coords], [y for x in x_coords, y in y_coords]
    
    # Create figure with 2D error maps
    fig = Figure(size=(1400, 600))
    
    # Original TCI error map
    ax1 = Axis(fig[1, 1], xlabel="x", ylabel="y", title="Error Map: $(method_name_orig)")
    hm1 = heatmap!(ax1, x_coords, y_coords, errors_orig_2d, colormap=:hot, colorrange=(0, maximum(vcat(errors_original, errors_patched))))
    Colorbar(fig[1, 1, Right()], hm1, label="Absolute Error")
    
    # PatchedTCI error map
    ax2 = Axis(fig[1, 2], xlabel="x", ylabel="y", title="Error Map: $(method_name_patch)")
    hm2 = heatmap!(ax2, x_coords, y_coords, errors_patch_2d, colormap=:hot, colorrange=(0, maximum(vcat(errors_original, errors_patched))))
    Colorbar(fig[1, 2, Right()], hm2, label="Absolute Error")
    
    # Difference map (Patched - Original)
    ax3 = Axis(fig[2, 1:2], xlabel="x", ylabel="y", title="Error Difference: $(method_name_patch) - $(method_name_orig)")
    error_diff = errors_patch_2d .- errors_orig_2d
    hm3 = heatmap!(ax3, x_coords, y_coords, error_diff, colormap=:RdBu, 
                   colorrange=(-maximum(abs.(error_diff)), maximum(abs.(error_diff))))
    Colorbar(fig[2, 1:2, Right()], hm3, label="Error Difference")
    
    # Save figure
    save_path = joinpath(@__DIR__, "error_2d_maps_$(h5file).png")
    save(save_path, fig)
    println("2D error maps saved to: $save_path")
    
    return fig
end

# -------------------------
# Plot errors comparison
# -------------------------
function plot_errors_comparison(errors_original, errors_patched, xs_orig, ys_orig, xs_patch, ys_patch, 
                                fs_orig, fs_patch, h5file::String, method_name_orig="Original TCI", method_name_patch="PatchedTCI")
    println("\n" * "="^60)
    println("Creating error plots...")
    println("="^60)
    
    # Create figure with multiple subplots
    fig = Figure(size=(1400, 1000))
    
    # 1. Error histogram comparison
    ax1 = Axis(fig[1, 1], xlabel="Absolute Error", ylabel="Frequency", 
               title="Error Distribution Comparison", yscale=log10)
    hist!(ax1, errors_original, bins=50, label=method_name_orig, alpha=0.6, color=:blue)
    hist!(ax1, errors_patched, bins=50, label=method_name_patch, alpha=0.6, color=:red)
    axislegend(ax1, position=:rt)
    
    # 2. Error CDF comparison
    ax2 = Axis(fig[1, 2], xlabel="Absolute Error", ylabel="Cumulative Probability", 
               title="Error CDF Comparison")
    sort_errors_orig = sort(errors_original)
    sort_errors_patch = sort(errors_patched)
    n_orig = length(sort_errors_orig)
    n_patch = length(sort_errors_patch)
    lines!(ax2, sort_errors_orig, (1:n_orig) ./ n_orig, label=method_name_orig, linewidth=2, color=:blue)
    lines!(ax2, sort_errors_patch, (1:n_patch) ./ n_patch, label=method_name_patch, linewidth=2, color=:red)
    axislegend(ax2, position=:rb)
    
    # 3. Spatial error map for Original TCI
    ax3 = Axis(fig[2, 1], xlabel="x", ylabel="y", title="Spatial Error Map: $(method_name_orig)")
    scatter!(ax3, xs_orig, ys_orig, color=errors_original, colormap=:hot, 
            markersize=8, colorrange=(minimum(vcat(errors_original, errors_patched)), 
                                     maximum(vcat(errors_original, errors_patched))))
    Colorbar(fig[2, 1, Right()], colormap=:hot, label="Absolute Error")
    
    # 4. Spatial error map for PatchedTCI
    ax4 = Axis(fig[2, 2], xlabel="x", ylabel="y", title="Spatial Error Map: $(method_name_patch)")
    scatter!(ax4, xs_patch, ys_patch, color=errors_patched, colormap=:hot, 
            markersize=8, colorrange=(minimum(vcat(errors_original, errors_patched)), 
                                     maximum(vcat(errors_original, errors_patched))))
    Colorbar(fig[2, 2, Right()], colormap=:hot, label="Absolute Error")
    
    # 5. Box plot comparison
    ax5 = Axis(fig[3, 1:2], xlabel="Method", ylabel="Absolute Error", 
               title="Error Statistics Comparison", yscale=log10)
    boxplot!(ax5, [1], errors_original, label=method_name_orig, color=:blue, width=0.4)
    boxplot!(ax5, [2], errors_patched, label=method_name_patch, color=:red, width=0.4)
    ax5.xticks = ([1, 2], [method_name_orig, method_name_patch])
    
    # Add statistics text
    stats_text = """
    Statistics:
    $(method_name_orig):
      Mean: $(round(mean(errors_original), digits=6))
      Median: $(round(median(errors_original), digits=6))
      Max: $(round(maximum(errors_original), digits=6))
      Std: $(round(std(errors_original), digits=6))
    
    $(method_name_patch):
      Mean: $(round(mean(errors_patched), digits=6))
      Median: $(round(median(errors_patched), digits=6))
      Max: $(round(maximum(errors_patched), digits=6))
      Std: $(round(std(errors_patched), digits=6))
    """
    
    Label(fig[3, 1:2, Bottom()], stats_text, fontsize=10, halign=:left, valign=:bottom, 
          padding=(10, 10, 10, 10))
    
    # 6. Error ratio scatter
    ax6 = Axis(fig[4, 1:2], xlabel="Original TCI Error", ylabel="PatchedTCI Error", 
               title="Error Comparison (log scale)", xscale=log10, yscale=log10)
    
    # Match up errors at similar locations (approximate)
    # Use a simple approach: sort by coordinates and match
    # For better matching, we'd need to evaluate at the same points, but this gives an idea
    min_len = min(length(errors_original), length(errors_patched))
    scatter!(ax6, errors_original[1:min_len], errors_patched[1:min_len], 
            alpha=0.5, markersize=5, color=:purple)
    
    # Add diagonal line (y=x)
    max_err = maximum(vcat(errors_original, errors_patched))
    min_err = minimum(vcat(errors_original, errors_patched))
    lines!(ax6, [min_err, max_err], [min_err, max_err], 
           color=:black, linestyle=:dash, linewidth=2, label="y=x")
    axislegend(ax6, position=:rt)
    
    # Add text annotation for ratio
    ratio_mean = mean(errors_patched) / mean(errors_original)
    ratio_median = median(errors_patched) / median(errors_original)
    ratio_text = "Mean ratio (Patch/Orig): $(round(ratio_mean, digits=3))\nMedian ratio: $(round(ratio_median, digits=3))"
    text!(ax6, ratio_text, position=(0.05, 0.95), space=:relative, fontsize=10, 
          align=(:left, :top), color=:black)
    
    # Save figure
    save_path = joinpath(@__DIR__, "error_comparison_$(h5file).png")
    save(save_path, fig)
    println("Error plots saved to: $save_path")
    
    return fig
end

# -------------------------
# Main execution
# -------------------------
# Only run when executed directly, not when included
if abspath(PROGRAM_FILE) == @__FILE__
    h5file = "f_t1e-8.h5"
# Ordering options:
#   :fused_xy  -> fused ordering, x bits then y bits
#   :fused_yx  -> fused ordering, y bits then x bits
#   :interleaved -> interleaved x/y bits
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

    println("Applying TCI methods to: $h5file")
    println("="^60)

    # Load data once for reuse (optimization)
    xs, ys, fs = load_velocity_data(h5file)

    # Apply original TCI
    original_tt, grid_orig, interpolator_orig, (x_coords_orig, y_coords_orig), original_tci, time_original = apply_original_tci(
        h5file;
        R=8,  # 2^8 = 256 points per dimension
        tol=1e-7,
        maxbonddim=500,
        verbosity=1,
        ordering=ordering_choice,
        fused_order=fused_order_choice,
    )

    # Apply adaptive patched TCI
    patched_tt, grid_patch, interpolator_patch, (x_coords_patch, y_coords_patch), time_patched = apply_adaptive_tci(
        h5file;
        R=8,  # 2^8 = 256 points per dimension
        tol=1e-7,
        maxbonddim=60,
        verbosity=1,
        ordering=ordering_choice,
        fused_order=fused_order_choice,
    )

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

    # Compare bond dimensions
    patched_bonddims, original_bonddims = compare_bond_dimensions(patched_tt, original_tt, original_tci)

    # Show max bond dimension for each patch
    println("\nMax bond dimension per patch:")
    for (i, ptt) in enumerate(patched_tt)
        tt = ptt.data
        linkdims = TCI.linkdims(tt)
        max_bond = isempty(linkdims) ? 0 : maximum(linkdims)
        println("  Patch $i: max bond dimension = $max_bond")
    end

    # Evaluate errors at ALL points for 2D plotting
    println("\n" * "="^60)
    println("Evaluating errors at all points for 2D maps:")
    println("="^60)
    errors_original_all = evaluate_all_points(original_tt, grid_orig, xs, ys, fs; method_name="Original TCI")
    errors_patched_all = evaluate_all_points(patched_tt, grid_patch, xs, ys, fs; method_name="PatchedTCI")

    # Collect patch bounds for Python plotting
    xmin, xmax = extrema(xs)
    ymin, ymax = extrema(ys)
    patch_bounds = collect_patch_bounds(
        patched_tt, xmin, xmax, ymin, ymax, 8; ordering=ordering_choice, fused_order=fused_order_choice
    )

    # Skip Julia plotting - save data for Python plotting instead

    # Save errors to HDF5 file for Python plotting
    println("\n" * "="^60)
    println("Saving errors to file for Python plotting...")
    println("="^60)
    save_errors_to_hdf5(errors_original_all, errors_patched_all, xs, ys, fs, 
                         time_original, time_patched, patched_bonddims, original_bonddims,
                         h5file; tol=1e-7, maxbonddim=30, R=8, patch_bounds=patch_bounds)

    println("\n" * "="^60)
    println("Summary:")
    println("="^60)
    println("Errors saved to: $(replace(h5file, ".h5" => "_errors.h5"))")
    println("\nTo plot in Python, run:")
    println("  python plot_errors_from_file.py $(replace(h5file, ".h5" => "_errors.h5"))")

    println("\nDone!")
end

