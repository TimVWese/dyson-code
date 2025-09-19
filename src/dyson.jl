module Dyson

using LinearAlgebra
using DifferentialEquations
using SparseArrays
using Optim
using Random
using Graphs
using JLD2

export default_initialisation
export get_SIS_H
export find_hermitian
export simulate
export get_steady_state
export infectious_proportion_O
export observe
export shannon_entropy
export renyi_entropy

"""
    default_initialisation()

Load or generate default graph, system matrix, and Dyson transformation, as reported on in the manuscript:
- Graph: Complete graph with 6 nodes
- No disease-free state
- Infection rate β = 0.5
- Healing rate γ = 0.05
- 100,000 iterations of the Dyson optimization
"""
function default_initialisation()
    # Instantiate the considered graph (6 nodes, complete graph)
    name = "full-6-wodf"

    path = joinpath(@__DIR__, "..", "results", "intermediate", "$name.jld2")
    if isfile(path)
        @info "Loaded existing results from $path"
        JLD2.@load path g A H h η β γ incl_disease_free errors
    else
        @info "No existing results found, generating new ones"
        n = 6
        g = complete_graph(6)
        A = adjacency_matrix(g)

        # Instantiate the dynamical system
        β = 0.5
        γ = 0.05

        incl_disease_free = false
        kmax = 50_000

        # Construct H and find the Dyson similarity transformation
        H = Matrix(Dyson.get_SIS_H(A, β, γ; incl_disease_free))
        @assert maximum(abs.(imag.(eigvals(H)))) < 1e-8
        h, η, errors = Dyson.find_hermitian(H; kmax, verbose=true)

        # Store for future use
        isdir(joinpath(@__DIR__, "..", "results", "intermediate")) || mkpath(joinpath(@__DIR__, "..", "results", "intermediate"))
        @info "Saving results to $path"
        JLD2.@save path g A H h η β γ incl_disease_free errors
    end
    @info "Final asymmetry: $(norm(h - h', 2))"
    return name, g, A, H, h, η, β, γ, incl_disease_free, errors
end

"""
    int2state(i::Integer, n::Integer)
Convert an integer `i` that indexes a state to a binary state vector of length `n`.
"""
function int2state(i::Integer, n::Integer)
    return collect(reverse(digits(i-1, base=2, pad=(n))))
end

"""
    get_rate_between(from::Integer, to::Integer, A::BitMatrix, β::Real, γ::Real)

Compute the transition rate between two states in a SIS model.
"""
function get_rate_between(from::Integer, to::Integer, A::AbstractMatrix, β::T, γ::T)::T where T<:Real
    from = int2state(from, size(A, 1))
    to = int2state(to, size(A, 1))
    # Only have a transition in one bit
    if count(from .!= to) != 1
        return zero(T)
    end
    # Find the index of the differing bit
    transition_bit = findfirst(from .!= to)
    if from[transition_bit] == 1
        # Healing transition
        return γ
    else
        # Infection transition, based on adjacency matrix A, with an extra rate for spontaneous infection
        return β * (dot(A[transition_bit, :], from))
    end
end

"""
    get_SIS_H_structure(A::AbstractMatrix)

Get the structure of the sparse SIS system matrix based on the adjacency matrix `A`.
Returns the row indices, column indices, and values for the non-zero entries of the matrix.
"""
function get_SIS_H_structure(A::AbstractMatrix)
    row_indices = Int64[]
    col_indices = Int64[]
    values = Int64[]
    for i in 1:2^size(A, 1)
        for j in 1:2^size(A, 1)
            rate = get_rate_between(j, i, A, 1, -1)  # Example rates
            if rate != 0 || i == j
                push!(row_indices, i)
                push!(col_indices, j)
                push!(values, rate)
            end
        end
    end
    return row_indices, col_indices, values
end

function get_SIS_H(row_indices, col_indices, values, β, γ; incl_disease_free=false)
    f_val = x -> (x < 0)*γ + (x > 0)*x*β
    scaled_values = [f_val(v) for v in values]
    H_sparse = sparse(row_indices, col_indices, scaled_values)
    !incl_disease_free && (H_sparse = H_sparse[2:end, 2:end])
    for i in axes(H_sparse, 1)
        H_sparse[i,i] = -sum(H_sparse[:,i])  # Ensure each row sums to zero
    end
    return -H_sparse
end

function get_SIS_H(A::AbstractMatrix, β, γ; incl_disease_free=false)
    row_indices, col_indices, values = get_SIS_H_structure(A)
    return get_SIS_H(row_indices, col_indices, values, β, γ; incl_disease_free)
end

"""
    c(x, y)

Compute the commutator (Lie bracket) of two matrices: [x,y] = xy - yx.

This operation measures how much two matrices fail to commute. If c(x,y) = 0,
then x and y commute.

# Arguments
- `x::Matrix`: First matrix
- `y::Matrix`: Second matrix

# Returns
- `Matrix`: The commutator [x,y] = xy - yx
"""
c(x,y) = x*y - y*x
# ======
"""
    quant_assymetry(x, H, Λ)

Quantify the asymmetry of a matrix after similarity transformation.

This function measures how far a matrix is from being Hermitian (self-adjoint)
after applying a similarity transformation exp(x[1]*Λ) * H * exp(-x[1]*Λ).
The asymmetry is quantified using the Frobenius norm of the anti-Hermitian part.

# Arguments
- `x::Vector`: Parameter vector (only x[1] is used as the transformation parameter)
- `H::Matrix`: Input matrix to transform
- `Λ::Matrix`: Generator matrix for the similarity transformation

# Returns
- `Float64`: Normalized measure of asymmetry, computed as tr((h-h')†(h-h'))/n
  where h is the transformed matrix and n is the matrix dimension

# Notes
- Returns a real value representing the "distance" from Hermitian
- Smaller values indicate the matrix is closer to being Hermitian
- Used as an objective function for optimization
"""
function quant_assymetry(x,H,Λ)
    n  = size(H)[1]
    h  = exp(x[1]*Λ)*H*exp(-x[1]*Λ)
    eq = h - h'
    return real( tr(eq'eq)/n )
end

"""
    update(H)

Perform one iteration of the algorithm to make matrix more Hermitian.

This function decomposes the input matrix into symmetric and anti-symmetric parts,
then finds the optimal similarity transformation to minimize the asymmetry.

# Arguments
- `H::Matrix`: Input matrix to make more Hermitian

# Returns
- `Tuple{Float64, Float64, Matrix}`: 
  - `x`: Optimal transformation parameter found by optimization
  - `asymmetry`: Residual asymmetry after transformation  
  - `h`: Transformed matrix after applying optimal similarity transformation

# Algorithm
1. Compute trace and decompose H into symmetric (H1) and anti-symmetric (ΔH) parts
2. Form generator A = c(H1, ΔH) for similarity transformations
3. Optimize transformation parameter to minimize asymmetry
4. Apply optimal transformation and return results
"""
function update(H ; x0 = 0.01)
#    q = tr(H)/size(H)[1]
    ΔH = (H - H')/2
#    H1 = (H + H')/2 - q*I
    #    A  = c(H1,ΔH) ;
    A = c(H, ΔH)
    res = optimize(z -> quant_assymetry(z,H,A), [x0], ConjugateGradient())
    x = res.minimizer[1]
    h = exp(x*A)*H*exp(-x*A)
    eq = h - h'
    return exp(x*A), norm(eq[:]) / size(eq, 1), h
end

"""
    find_hermitian(H; tol = 1e-8, kmax = 1000)

Iteratively transform a matrix to make it as Hermitian as possible.

This function repeatedly applies similarity transformations to reduce the 
asymmetry of the input matrix. The algorithm stops when either the asymmetry
falls below the tolerance or the maximum number of iterations is reached.

# Arguments
- `H::Matrix`: Input matrix to make Hermitian
- `tol::Float64`: Convergence tolerance for asymmetry measure (default: 1e-8)
- `kmax::Int`: Maximum number of iterations (default: 1000)

# Returns
- `Matrix`: Transformed matrix that is approximately Hermitian

# Algorithm
The method uses iterative similarity transformations of the form:
H_{k+1} = exp(α_k A_k) H_k exp(-α_k A_k)
where A_k and α_k are chosen to minimally reduce the asymmetry at each step.

# Notes
- Preserves eigenvalues (similarity transformations are spectrum-preserving)
- Convergence is not guaranteed for all matrices
- Final result minimizes ||H - H†||_F among similarity transformations tried
"""
function find_hermitian(H ; tol = 1e-8, kmax = 2500, verbose=false)
    current = 1
    k = 1
    h = zeros(size(H))
    h .= H
    η = I(size(H)[1])
    errors = Array{Float64, 1}(undef, kmax)
    while (k < kmax) * (current > tol)
        ηc, current, h = update(h)
        η = ηc * η
        verbose && println("Iteration $k: Asymmetry = $current")
        errors[k] = current
        k = k+1
    end
    return h, η, errors[1:k-1]
end

"""
    simulate(H, η=I; symmetrize=false, tspan=(0.,25.), saveat=0.1)

Simulate the model given by `H` using a similarity-transformed matrix.
# Arguments
- `H::Matrix`: System matrix to simulate x'(t) = -H x(t)
- `η::Matrix`: Similarity transformation matrix (default: identity matrix)
- `symmetrize::Bool`: Whether to symmetrize the transformed matrix (default: false)
- `tspan::Tuple{Float64, Float64}`: Time span for the simulation (default: (0.0, 25.0))
- `saveat::Float64`: Time step for saving results (default: 0.1)
# Returns
- `ODESolution`: Solution of the ODE problem representing the SIS dynamics
"""
function simulate(H, η=I; symmetrize=false, tspan=(0.,25.), saveat=0.1)
    # Initial state: all susceptible
    initial_state = zeros(size(H, 1))
    initial_state[2] = 1.0  # Start with one infected individual
    initial_state = η * initial_state  # Apply the similarity transformation

    f(du, u, p, t) = (du .= -1 * p[1] * u)
    H_prime = η * H * inv(η)
    symmetrize ? H_prime = (H_prime + H_prime') / 2 : nothing
    prob = ODEProblem(f, initial_state, tspan, [H_prime])
    return solve(prob, Tsit5(); saveat)
end

"""
    get_steady_state(H; η=I, symmetrize=false)

Compute the steady state of the dynamics defined by `H` by forward integration.
# Arguments
- `H::Matrix`: Matrix for the  model
- `η::Matrix`: Similarity transformation matrix (default: identity matrix)
- `symmetrize::Bool`: Whether to symmetrize the transformed system (default: false)
"""
function get_steady_state(H; η=I, symmetrize=false)
    n = size(H, 1)
    x0 = zeros(n)
    x0[end] = 1.0
    x0 = η * x0
    if symmetrize
        H = (H + H') / 2
    end
    f = (du, u, p, t) -> du .= -H * u
    prob = SteadyStateProblem(f, x0)
    sol = solve(prob, DynamicSS(Rodas5()); reltol=1e-6)
    return sol.u
end

"""
    infectious_proportion_O(n; incl_disease_free=false)

Generate the infectious proportion operator for a system with `n` individuals.
If `incl_disease_free` is false, the disease-free state is excluded.
"""
function infectious_proportion_O(n; incl_disease_free=false)
    start_i = incl_disease_free ? 1 : 2
    return diagm([sum(Dyson.int2state(i, n)) for i in start_i:2^n]) ./ n
end

"""
    observe(O, p)

Observe the system using the operator `O` and the state `p`.
"""
function observe(O, p)
    return sum(O[i,j] * p[j] for i in eachindex(p), j in eachindex(p))
end

"""
    observe(O, ϕ, Ω)

Observe the system using the operator `O`, state `ϕ`, and metric `Ω`.
"""
function observe(O, ϕ, Ω)
    return ϕ' * Ω * O * ϕ
end

function shannon_entropy(p)
    -sum(p .* log.(p .+ 1e-10))
end

function renyi_entropy(p)
    -log(sum(p .^ 2))
end

end