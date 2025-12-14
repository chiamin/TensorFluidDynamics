using ITensors
using Random
using T4AITensorCompat: TensorTrain
import ITensors: random_itensor

function _random_mpo(
    rng::AbstractRNG, sites::AbstractVector{<:AbstractVector{Index{T}}}; m::Int=1
) where {T}
    N = length(sites)
    links = [Index(m, "Link,n=$n") for n in 1:(N - 1)]
    tensors = Vector{ITensor}(undef, N)
    tensors[1] = random_itensor(rng, sites[1]..., links[1])
    tensors[N] = random_itensor(rng, links[N - 1], sites[N]...)
    for n in 2:(N - 1)
        tensors[n] = random_itensor(rng, links[n - 1], sites[n]..., links[n])
    end
    return TensorTrain(tensors)
end

function _random_mpo(sites::AbstractVector{<:AbstractVector{Index{T}}}; m::Int=1) where {T}
    return _random_mpo(Random.default_rng(), sites; m=m)
end
