import PolyChaos.computeSP2

# getting independent variables
function get_independent_vars(var)
    return []
end
function get_independent_vars(var::Symbolics.Term) where {T}
    if operation(var) isa Symbolics.Sym
        return arguments(var)
    else
        return reduce(vcat, get_independent_vars(arguments(var)))
    end
end
function get_independent_vars(var::Num)
    return get_independent_vars(Symbolics.unwrap(var))
end
function get_independent_vars(vars::AbstractVector)
    return [get_independent_vars(var) for var in vars]
end

# utiltiites to extracting coefficients of a polynomial in monomial basis in variables `vars`
function split_term(term::Symbolics.Mul, vars)
    coeff = term.coeff
    mono = Num(1.0)
    for var in keys(term.dict)
        if var in vars
            mono *= var^term.dict[var]
        else
            coeff *= var^term.dict[var]
        end
    end
    return coeff, mono
end

function split_term(term::Symbolics.Pow, vars)
    if term.base in vars
        return 1.0, term
    else
        return term, Val(1)
    end
end

function split_term(term::T, vars) where {T <: Union{Symbolics.Term, Symbolics.Sym}}
    if term in vars
        return 1.0, term
    else
        return term, Val(1)
    end
end

function extract_coeffs(expr::Symbolics.Add, vars::Set)
    coeffs = Dict()
    if !iszero(expr.coeff)
        coeffs[Val(1)] = expr.coeff
    end
    for term in keys(expr.dict)
        num_coeff = expr.dict[term]
        var_coeff, mono = split_term(term, vars)
        try
            coeffs[mono] += num_coeff * var_coeff
        catch
            coeffs[mono] = num_coeff * var_coeff
        end
    end
    return coeffs
end

function extract_coeffs(expr::T, vars::Set) where {T <: Union{Symbolics.Sym, Symbolics.Mul}}
    coeff, mono = split_term(expr, vars)
    return Dict(mono => coeff)
end

extract_coeffs(expr::Num, vars::Set) = extract_coeffs(Symbolics.unwrap.(expr), vars)
function extract_coeffs(expr::Num, vars::AbstractArray)
    extract_coeffs(Symbolics.unwrap.(expr), vars)
end
function extract_coeffs(expr, vars::AbstractArray{<:Num})
    extract_coeffs(expr, Symbolics.unwrap.(vars))
end
extract_coeffs(expr, vars::AbstractArray) = extract_coeffs(expr, Set(vars))
extract_coeffs(expr::Number, vars::Set) = Dict(Val(1) => expr)

# extracting the indices of the factors of as basismonomial
function get_basis_indices(mono::Symbolics.Mul)
    basis_indices = Int[]
    for (term, pow) in mono.dict
        append!(basis_indices, (arguments(term)[end] - 1) * ones(Int, pow))
    end
    return basis_indices
end
function get_basis_indices(mono::Symbolics.Term)
    return [arguments(mono)[end] - 1]
end
function get_basis_indices(mono::Symbolics.Pow)
    return (arguments(mono.base)[end] - 1) * ones(Int, mono.exp)
end
function get_basis_indices(mono::Num)
    return get_basis_indices(Symbolics.unwrap(mono))
end
function get_basis_indices(::Val{1})
    return [0]
end

# bumping the degree of a PolyChaos OrthoPoly object up to ensure exact integration
# PR to PolyChaos -> remove unnecessarily restrictive constructors and allow construction from measures
#                 -> also expose number of points used for quadrature generation for general orthogonal polys
#

measure_parameters(m::AbstractMeasure) = []
measure_parameters(m::Measure) = m.pars
measure_parameters(m::JacobiMeasure) = [m.ashapeParameter, m.bshapeParameter]
measure_parameters(m::genLaguerreMeasure) = [m.shapeParameter]
measure_parameters(m::genHermiteMeasure) = [m.muParameter]
measure_parameters(m::MeixnerPollaczekMeasure) = [m.λParameter, m.ϕParameter]
measure_parameters(m::Beta01Measure) = [m.ashapeParameter, m.bshapeParameter]
measure_parameters(m::GammaMeasure) = [m.shapeParameter, m.rateParameter]

function bump_degree(op::OrthoPoly, deg::Int)
    return OrthoPoly(op.name, deg, op.measure)
end

function bump_degree(op::JacobiOrthoPoly, deg::Int)
    ps = measure_parameters(op.measure)
    return JacobiOrthoPoly(deg, ps...)
end

function bump_degree(op::genLaguerreOrthoPoly, deg::Int)
    ps = measure_parameters(op.measure)
    return genLaguerreOrthoPoly(deg, ps...)
end

function bump_degree(op::MeixnerPollaczekOrthoPoly, deg::Int)
    ps = measure_parameters(op.measure)
    return MeixnerPollaczekOrthoPoly(deg, ps...)
end

function bump_degree(op::Beta01OrthoPoly, deg::Int)
    ps = measure_parameters(op.measure)
    return Beta01OrthoPoly(deg, ps...)
end

function bump_degree(op::GammaOrthoPoly, deg::Int)
    ps = measure_parameters(op.measure)
    return GammaOrthoPoly(deg, ps...)
end

function bump_degree(op::genHermiteOrthoPoly, deg::Int)
    ps = measure_parameters(op.measure)
    return genHermiteOrthoPoly(deg, ps...)
end

function bump_degree(op::HermiteOrthoPoly, deg::Int)
    ps = measure_parameters(op.measure)
    return HermiteOrthoPoly(deg, ps...)
end

function bump_degree(op::LaguerreOrthoPoly, deg::Int)
    ps = measure_parameters(op.measure)
    return LaguerreOrthoPoly(deg, ps...)
end

function bump_degree(op::Uniform01OrthoPoly, deg::Int)
    ps = measure_parameters(op.measure)
    return Uniform01OrthoPoly(deg, ps...)
end

function bump_degree(op::Uniform_11OrthoPoly, deg::Int)
    ps = measure_parameters(op.measure)
    return Uniform_11OrthoPoly(deg, ps...)
end

function bump_degree(op::GaussOrthoPoly, deg::Int)
    ps = measure_parameters(op.measure)
    return GaussOrthoPoly(deg, ps...)
end

function bump_degree(op::LegendreOrthoPoly, deg::Int)
    ps = measure_parameters(op.measure)
    return LegendreOrthoPoly(deg, ps...)
end

function bump_degree(op::LogisticOrthoPoly, deg::Int)
    ps = measure_parameters(op.measure)
    return LogisticOrthoPoly(deg, ps...)
end

function bump_degree(op::MultiOrthoPoly, deg::Int)
    return MultiOrthoPoly(bump_degree.(op.uni, deg), deg)
end

# extending computeSP2 for multivariate orthogonal polys
function computeSP2(pc::MultiOrthoPoly)
    n = length(pc.uni)
    m = dim(pc)
    uni_SP2 = [computeSP2(op) for op in pc.uni]
    multi_SP2 = [prod(uni_SP2[j][pc.ind[i, j] + 1] for j in 1:n) for i in 1:m]
    return multi_SP2
end

# computing maximum degree
function maximum_degree(mono_indices::AbstractVector, pce::PCE)
    max_degree = 0
    for (mono, ind) in mono_indices
        max_degree = max(max_degree,
                         maximum(sum(pce.pc_basis.ind[ind[i] + 1, :]
                                     for i in eachindex(ind))))
    end
    return max_degree
end

"""
`TensorProductOrthoPoly` objects represent bases formed as the tensor product of univariate `PolyChaos.AbstractOrthoPoly` bases.
By default the basis elements of the tensor product are restricted to polynomials with total degree up to the maximum degree among the 
univariate bases. This maximum degree can be manually specified, however.
"""
struct TensorProductOrthoPoly{M,U}
    ind::Matrix
    measure::M
    deg::Vector{Int}
    uni::U
end
function TensorProductOrthoPoly(ops::AbstractVector{T}) where T <: Union{AbstractOrthoPoly, AbstractCanonicalOrthoPoly}
    n = length(ops)
    degrees = [deg(op) for op in ops]
    ind = grevlex(n, 0:maximum(degrees), degrees)
    measures = [op.measure for op in ops]
    prod_measure = ProductMeasure(t -> prod(measure.w for measure in measures), 
                                  measures) 
    
    return TensorProductOrthoPoly(ind, prod_measure, degrees, ops)
end
function TensorProductOrthoPoly(ops::AbstractVector{T}, max_deg::Int) where T <: Union{AbstractOrthoPoly, AbstractCanonicalOrthoPoly}
    n = length(ops)
    degrees = [deg(op) for op in ops]
    ind = grevlex(n, 0:max_deg, degrees)
    measures = [op.measure for op in ops]
    prod_measure = ProductMeasure(t -> prod(measure.w for measure in measures), 
                                  measures) 
    
    return TensorProductOrthoPoly(ind, prod_measure, degrees, ops)
end

"""
$(TYPEDSIGNATURES)

computes inner product between basis functions of a `TensorProductOrthoPoly` via 
`PolyChaos`'s infrastructure (exploiting the tensor product form). 
"""
function computeSP(basis_fxns, tpop::TensorProductOrthoPoly)
    multi_indices = tpop.ind[basis_fxns, :]
    # columns of multi_indices refer to inner products to be computed
    sp = 1.0
    for k in axes(multi_indices, 2)
        sp *= computeSP(multi_indices[:,k], tpop.uni[k])
    end
    return round(sp, digits = 12)
end

"""
$(TYPEDSIGNATURES)

Compute the an ascending list of `n`-dimensional multi-indices with fixed `grade` (= sum of entries) 
in graded reverse lexicographic order. Constraints on the degrees considered can be incorporated. 
"""
function grevlex(n::Int, grade::Int)
    if n == 1
        return reshape([grade], 1, 1)
    end
    
    if grade == 0
        return zeros(Int, 1, n)
    end

    sub_ind = grevlex(n-1,grade)
    ind = hcat(sub_ind, zeros(Int,size(sub_ind,1)))
    for k in 1:grade
        sub_ind = grevlex(n-1,grade-k)
        ind = vcat(ind, hcat(sub_ind, k*ones(Int,size(sub_ind,1))))
    end
    return ind
end

function grevlex(n::Int, grades::AbstractVector)
    return reduce(vcat, [grevlex(n, grade) for grade in grades])
end

function grevlex(n::Int, grade::Int, max_degrees::Vector{Int})
    return grevlex(n, grade, [0:d for d in max_degrees])
end

function grevlex(n::Int, grades::AbstractVector, max_degrees::Vector{Int})
    return reduce(vcat, [grevlex(n, grade, max_degrees) for grade in grades])
end

function grevlex(n::Int, grade::Int, constrained_degrees::Vector{<:AbstractVector})
    if n == 1
        return grade in constrained_degrees[1] ? reshape([grade], 1, 1) : zeros(Int,0,1)
    end
    
    if grade == 0
        return zeros(Int, 1, n)
    end

    filtered_grades = filter(x -> x <= grade, constrained_degrees[end])
    sub_ind = grevlex(n-1, grade - filtered_grades[1], constrained_degrees[1:end-1])
    ind = hcat(sub_ind, filtered_grades[1]*ones(Int,size(sub_ind,1)))
    for k in filtered_grades[2:end]
        sub_ind = grevlex(n-1, grade-k, constrained_degrees[1:end-1])
        ind = vcat(ind, hcat(sub_ind, k*ones(Int,size(sub_ind,1))))
    end
    return ind
end