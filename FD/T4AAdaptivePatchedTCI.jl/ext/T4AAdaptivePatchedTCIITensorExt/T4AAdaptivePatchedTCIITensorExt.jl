module T4AAdaptivePatchedTCIITensorExt

using T4AAdaptivePatchedTCI: T4AAdaptivePatchedTCI
import T4AAdaptivePatchedTCI:
    Projector, ProjTensorTrain, ProjTTContainer, MultiIndex, MMultiIndex, multii
import T4APartitionedTT: SubDomainTT, PartitionedTT, Projector as PartitionedProjector
import T4AITensorCompat: TensorTrain as ITensorTensorTrain, MPS, siteinds, linkinds
import T4ATensorCI as TCI
using ITensors

# Include the ITensor compatibility code
include(joinpath(@__DIR__, "itensor.jl"))

end
