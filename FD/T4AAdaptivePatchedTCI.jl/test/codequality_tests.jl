using Test
using Aqua
using JET
using T4AAdaptivePatchedTCI

@testset "Aqua.jl" begin
    Aqua.test_all(T4AAdaptivePatchedTCI; deps_compat=false)
end

#=
@testset "JET.jl" begin
    JET.test_package(T4AAdaptivePatchedTCI; target_defined_modules=true)
end
=#
