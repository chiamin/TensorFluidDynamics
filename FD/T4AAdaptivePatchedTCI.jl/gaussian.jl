using Pkg
Pkg.Registry.add(RegistrySpec(url="https://github.com/tensor4all/T4ARegistry.git"))
Pkg.activate(".")
Pkg.instantiate()

using Revise
using Pkg
Pkg.Registry.add(RegistrySpec(url="https://github.com/tensor4all/T4ARegistry.git"))
Pkg.activate(".")
Pkg.instantiate()

using Random
import QuanticsGrids as QG

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
using Test

Random.seed!(1234)

gaussian(x, y) = exp(- ((x-5)^2 + (y-5)^2)) + exp(- ((x+5)^2 + (y+5)^2))

R = 20
xmax = 10.0
grid = QG.DiscretizedGrid{2}(R, (-xmax, -xmax), (xmax, xmax); unfoldingscheme=:interleaved)
localdims = fill(2, 2R)  # interleaved: 2R sites (R for x, R for y)

# Convert gaussian function to quantics coordinates
qf = x -> gaussian(QG.quantics_to_origcoord(grid, x)...)

tol = 1e-7
maxbonddim = 30

# Create patch ordering (interleaved: 2R sites)
pordering = PatchedTCI.PatchOrdering(collect(1:2R))

# Create TCI2PatchCreator
creator = PatchedTCI.TCI2PatchCreator(
    Float64,
    PatchedTCI.makeprojectable(Float64, qf, localdims),
    localdims,
    maxbonddim=maxbonddim,
    tolerance=tol,
    verbosity=1,
    ntry=10,
)

# Run adaptive interpolation
results = PatchedTCI.adaptiveinterpolate(creator, pordering; verbosity=1)
patched_tt = PatchedTCI.ProjTTContainer(results)

@show patched_tt
