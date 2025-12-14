using Test
import T4ATensorCI as TCI
import T4AAdaptivePatchedTCI as TCIA
using T4AQuantics
import T4APartitionedTT: siteinds, SubDomainTT, project as project_subdmps
import T4AITensorCompat: MPS

import T4AAdaptivePatchedTCI: Projector, project, ProjTensorTrain, makeprojectable

using ITensors

# _util.jl is already included in runtests.jl

@testset "itensor" begin
    @testset "SubDomainTT" begin
        N = 3
        sitesx = [Index(2, "x=$n") for n in 1:N]
        sitesy = [Index(2, "y=$n") for n in 1:N]
        sites = collect(collect.(zip(sitesx, sitesy)))
        Ψ = _random_mpo(sites)  # MPO is TensorTrain, MPS is also TensorTrain
        prjΨ = SubDomainTT(Ψ)

        prjΨ1 = project_subdmps(prjΨ, Dict(sitesx[1] => 1))
        prjΨ2 = project_subdmps(prjΨ, Dict(sitesx[1] => 2))

        Ψreconst = MPS(prjΨ1) + MPS(prjΨ2)

        @test Ψreconst ≈ Ψ
    end

    @testset "conversion" begin
        N = 2
        sitesx = [Index(2, "x=$n") for n in 1:N]
        sitesy = [Index(2, "y=$n") for n in 1:N]
        sites = collect(collect.(zip(sitesx, sitesy)))
        sitedims = [dim.(s) for s in sites]
        Ψ = _random_mpo(sites)  # MPO is TensorTrain, MPS is also TensorTrain
        prjΨ = SubDomainTT(Ψ)
        prjΨ1 = project_subdmps(prjΨ, Dict(sitesx[1] => 1))

        prjtt1 = TCIA.ProjTensorTrain{Float64}(prjΨ1)
        @test prjtt1.projector == Projector([[1, 0], [0, 0]], sitedims)

        prjΨ1_reconst = SubDomainTT(Float64, prjtt1, sites)

        @test prjΨ1 ≈ prjΨ1_reconst
    end

    @testset "rearrange_siteinds" begin
        N = 3
        sitesx = [Index(2, "x=$n") for n in 1:N]
        sitesy = [Index(2, "y=$n") for n in 1:N]
        sitesz = [Index(2, "z=$n") for n in 1:N]
        sites = collect(collect.(zip(sitesx, sitesy, sitesz)))

        Ψ = _random_mpo(sites)  # MPO is TensorTrain, MPS is also TensorTrain

        prjΨ = SubDomainTT(Ψ)
        prjΨ1 = project_subdmps(prjΨ, Dict(sitesx[1] => 1))

        sitesxy = collect(collect.(zip(sitesx, sitesy)))
        sites_rearranged = Vector{Index{Int}}[]
        for i in 1:N
            push!(sites_rearranged, sitesxy[i])
            push!(sites_rearranged, [sitesz[i]])
        end
        prjΨ1_rearranged = T4AQuantics.rearrange_siteinds(prjΨ1, sites_rearranged)

        @test reduce(*, MPS(prjΨ1)) ≈ reduce(*, MPS(prjΨ1_rearranged))
        @test siteinds(prjΨ1_rearranged) == sites_rearranged
    end

    @testset "makesitediagonal and extractdiagonal" begin
        N = 3
        sitesx = [Index(2, "x=$n") for n in 1:N]
        sitesy = [Index(2, "y=$n") for n in 1:N]
        sitesz = [Index(2, "z=$n") for n in 1:N]

        sitesxy_vec = [[x, y] for (x, y) in zip(sitesx, sitesy)]
        sitesz_vec = [[z] for z in sitesz]
        sites = [x for pair in zip(sitesxy_vec, sitesz_vec) for x in pair]

        Ψ = _random_mpo(sites)  # MPO is TensorTrain, MPS is also TensorTrain

        prjΨ = SubDomainTT(Ψ)
        prjΨ1 = project_subdmps(prjΨ, Dict(sitesx[1] => 1))

        prjΨ1_diagonalz = T4AQuantics.makesitediagonal(prjΨ1, "y")
        sites_diagonalz = Iterators.flatten(siteinds(prjΨ1_diagonalz))

        psi_diag = prod(MPS(prjΨ1_diagonalz))
        psi = prod(MPS(prjΨ1))

        @test T4AQuantics.extractdiagonal(prjΨ1_diagonalz, "y") ≈ prjΨ1

        for indval in eachindval(sites_diagonalz...)
            ind = first.(indval)
            val = last.(indval)

            index_dict = Dict{Index{Int},Vector{Int}}()
            for (i, el) in enumerate(ind)
                baseind = noprime(el)
                if haskey(index_dict, baseind)
                    push!(index_dict[baseind], i)
                else
                    index_dict[baseind] = [i]
                end
            end
            repeated_indices = [is for is in values(index_dict) if length(is) > 1]

            isdiagonalelement = all(allequal(val[i] for i in is) for is in repeated_indices)

            if isdiagonalelement
                nondiaginds = unique(noprime(i) => v for (i, v) in indval)
                @test psi_diag[indval...] == psi[nondiaginds...]
            else
                @test iszero(psi_diag[indval...])
            end
        end
    end
end
