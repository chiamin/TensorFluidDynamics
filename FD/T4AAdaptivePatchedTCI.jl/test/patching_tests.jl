using Test
using Random

using T4ATensorCI
import T4ATensorCI as TCI
import T4AAdaptivePatchedTCI as TCIA

@testset "patching" begin
    @testset "createpath" begin
        L = 3
        sitedims = [[2], [2], [2]]

        po = TCIA.PatchOrdering(collect(1:L))
        @test TCIA.createpath(TCIA.Projector([[0], [0], [0]], sitedims), po) == Int[]
        @test TCIA.createpath(TCIA.Projector([[1], [1], [0]], sitedims), po) == [1, 1]

        po = TCIA.PatchOrdering(reverse(collect(1:L)))
        @test TCIA.createpath(TCIA.Projector([[0], [0], [0]], sitedims), po) == Int[]
        @test TCIA.createpath(TCIA.Projector([[0], [0], [1]], sitedims), po) == [1]
    end

    @testset "makechildproj" begin
        sitedims = [[2, 2], [2, 2], [2, 2], [2, 2]]
        N = length(sitedims)

        let
            po = TCIA.PatchOrdering(collect(1:N))
            proj = TCIA.Projector([[0, 0], [0, 0], [0, 0], [0, 0]], sitedims)
            @test TCIA.makechildproj(proj, po) == [
                TCIA.Projector([[1, 1], [0, 0], [0, 0], [0, 0]], sitedims),
                TCIA.Projector([[2, 1], [0, 0], [0, 0], [0, 0]], sitedims),
                TCIA.Projector([[1, 2], [0, 0], [0, 0], [0, 0]], sitedims),
                TCIA.Projector([[2, 2], [0, 0], [0, 0], [0, 0]], sitedims),
            ]
        end

        let
            po = TCIA.PatchOrdering(reverse(collect(1:N)))
            proj = TCIA.Projector([[0, 0], [0, 0], [0, 0], [0, 0]], sitedims)
            @test TCIA.makechildproj(proj, po) == [
                TCIA.Projector([[0, 0], [0, 0], [0, 0], [1, 1]], sitedims),
                TCIA.Projector([[0, 0], [0, 0], [0, 0], [2, 1]], sitedims),
                TCIA.Projector([[0, 0], [0, 0], [0, 0], [1, 2]], sitedims),
                TCIA.Projector([[0, 0], [0, 0], [0, 0], [2, 2]], sitedims),
            ]
        end
    end

    @testset "makeproj" begin
        sitedims = [[2, 2], [2, 2], [2, 2], [2, 2]]

        po = TCIA.PatchOrdering([1, 3, 2, 4])
        prefix = [[1, 2], [1, 1]]

        TCIA.makeproj(po, prefix, sitedims) ==
        TCIA.Projector([[1, 2], [0, 0], [1, 1], [0, 0]], sitedims)
    end
end
