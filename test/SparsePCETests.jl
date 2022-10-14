using Test, ModelOrderReduction
using ModelingToolkit, PolyChaos, LinearAlgebra
const MOR = ModelOrderReduction

include("PCETestUtils.jl")
@testset "SparsePCE: constructor test" begin
    @parameters a, b
    @variables x, y
    
    ops = [Uniform01OrthoPoly(3), GaussOrthoPoly(2)]
    indices_x = [[0,0], [1,0], [2,1], [3,0]]
    indices_y = [[0,0], [0,2], [1,1]]
    sp_pce = MOR.SparsePCE([x => indices_x, y => indices_y],
                           [a => ops[1], b => ops[2]])    

    sparse_pc_indices = [[0,0],[1,0],[0,2],[1,1], [3,0], [2,1]]
    @test length(sp_pce.sym_basis) == 6
    @test length(sp_pce.moments[1]) == 4
    @test length(sp_pce.moments[2]) == 3
    @test all(isequal(sp_pce.pc_to_sym[sp_pce.pc_basis.ind[sp_pce.sym_to_pc[baxel], :]], baxel) for baxel in sp_pce.sym_basis)

    sparse_pc_indices_x = [sp_pce.pc_basis.ind[sp_pce.sym_to_pc[baxel], :] for baxel in sp_pce.sparse_sym_basis[1]]
    @test sp_pce.sparse_pc_basis[1] == indices_x
    @test indices_x == sparse_pc_indices_x

    sparse_pc_indices_y = [sp_pce.pc_basis.ind[sp_pce.sym_to_pc[baxel], :] for baxel in sp_pce.sparse_sym_basis[2]]
    @test sp_pce.sparse_pc_basis[2] == indices_y    
    @test indices_y == sparse_pc_indices_y

    # test evaluation 
    a_val, b_val = rand(2)
    x_moments, y_moments = randn(4), randn(3)
    x_true = sum(x_moments[i]*evaluate(indices_x[i][1], a_val, ops[1])*evaluate(indices_x[i][2], b_val, ops[2]) for i in eachindex(x_moments))
    y_true = sum(y_moments[i]*evaluate(indices_y[i][1], a_val, ops[1])*evaluate(indices_y[i][2], b_val, ops[2]) for i in eachindex(y_moments))
    x_val, y_val = sp_pce([x_moments, y_moments], [a_val, b_val])
    @test isapprox(x_val, x_true)
    @test isapprox(y_val, y_true)
end

# test equation for throughout:
@parameters a, b
@variables t, y(t)
D = Differential(t)
test_equation = [D(y) ~ a * y + 4 * b]

# set up pce
n = 5
bases = [a => GaussOrthoPoly(n)]
basis_fxns = [[5],[1],[2],[0]]
pce = SparsePCE([y => basis_fxns], bases)
eq = [eq.rhs for eq in test_equation]
pce_eq = MOR.apply_ansatz(eq, pce)[1]
m = length(pce.sym_basis)

@testset "SparsePCE: apply_ansatz test" begin
    true_eq = expand(pce.sym_basis[2] * dot(pce.moments[1], pce.sym_basis) + 4 * b)
    @test isequal(pce_eq, true_eq)
end

# test extraction of monomial coefficients
coeffs = Dict{Any, Any}(pce.sym_basis[i] * pce.sym_basis[2] => pce.moments[1][i]
                        for i in 1:m)
coeffs[Val(1)] = 4.0 * b
basis_indices = Dict{Any, Any}(pce.sym_basis[i] * pce.sym_basis[2] => ([basis_fxns[i][1], 1],
                                                                       [1, basis_fxns[i][1]])
                               for i in 1:m)
basis_indices[Val(1)] = [[0], [0]]

@testset "SparsePCE: basismonomial extraction test" begin
    extracted_coeffs = MOR.extract_coeffs(pce_eq, pce.sym_basis)
    @test all(isequal(coeffs[mono], extracted_coeffs[mono]) for mono in keys(coeffs))

    extracted_coeffs, extracted_indices = MOR.extract_basismonomial_coeffs([pce_eq], pce)
    extracted_indices = Dict(extracted_indices)
    test1 = [isequal(basis_indices[mono][1], extracted_indices[mono])
             for mono in keys(basis_indices)]
    test2 = [isequal(basis_indices[mono][2], extracted_indices[mono])
             for mono in keys(basis_indices)]
    @test all(test1 + test2 .>= 1)
end

# test Galerkin projection
@testset "SparsePCE: galerkin projection test" begin
    moment_eqs = MOR.pce_galerkin(eq, pce)[1]
    integrator = MOR.bump_degree(pce.pc_basis, n + 1)

    true_moment_eqs = Num[]
    scaling_factors = computeSP2(pce.pc_basis)
    for j in basis_fxns
        j = j[1]
        mom_eq = 0.0
        for mono in keys(basis_indices)
            ind = basis_indices[mono][2]
            c = computeSP(vcat(ind, j), integrator)
            mom_eq += c * coeffs[mono]
        end
        push!(true_moment_eqs, 1/scaling_factors[j+1] * mom_eq)
    end

    @test integrator.deg == n + 1
    @test integrator.measure isa typeof(pce.pc_basis.measure)
    @test integrator.measure.measures[1] isa typeof(pce.pc_basis.measure.measures[1])
    @test all([isapprox_sym(moment_eqs[i], true_moment_eqs[i])
               for i in eachindex(true_moment_eqs)])

    # check generation of moment equations
    @named test_system = ODESystem(test_equation, t, [y], [a, b])
    moment_system, pce_eval = moment_equations(test_system, pce)
    moment_eqs = equations(moment_system)
    moment_eqs = [moment_eqs[i].rhs for i in eachindex(moment_eqs)]
    @test isequal(parameters(moment_system), [b])
    @test nameof(moment_system) == :test_system_pce
    @test isequal(states(moment_system), reduce(vcat, pce.moments))
    @test all([isapprox_sym(moment_eqs[i], true_moment_eqs[i])
               for i in eachindex(true_moment_eqs)])
end
