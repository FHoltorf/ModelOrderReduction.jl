using ModelOrderReduction, ModelingToolkit, PolyChaos
MOR = ModelOrderReduction

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
