using GPUArrayPooling
using Test
using Aqua
using JET

@testset "GPUArrayPooling.jl" begin
    @testset "Code quality (Aqua.jl)" begin
        Aqua.test_all(GPUArrayPooling)
    end
    @testset "Code linting (JET.jl)" begin
        JET.test_package(GPUArrayPooling; target_defined_modules = true)
    end
    # Write your tests here.
end
