exit()

using Random
using Graphs
using LinearAlgebra
using ProgressMeter
using JLD2
using Logging
using SparseArrays
include(joinpath(@__DIR__, "..", "src", "dyson-setup.jl"))

"""
    unpack_network(n, i)

Unpack the integer `i` (<= n * (n-1) ÷ 2) into a network representation for `n` nodes.
"""
function unpack_network(n, i)
    n_edges = n * (n - 1) ÷ 2

    bits = reverse(digits(i - 1, base=2, pad=n_edges))
    adj = zeros(Int, n, n)
    idx = 1
    for row in 1:n-1
        for col in row+1:n
            adj[row, col] = bits[idx]
            adj[col, row] = bits[idx]
            idx += 1
        end
    end
    return Graph(adj)
end

"""
    get_permutations(n)

Get all permutations of `n` nodes, in the form of permutation matrices.
"""
function get_permutations(n)
    if n == 1
        return sparse.([I(1)])
    else
        smaller_perms = get_permutations(n - 1)
        perms = SparseMatrixCSC[]
        for perm in smaller_perms
            for i in 1:n
                P = spzeros(Int, n, n)
                P[i, 1] = 1
                row_idx = 1
                for j in 1:n
                    if j != i
                        P[j, 2:end] = perm[row_idx, :]
                        row_idx += 1
                    end
                end
                push!(perms, P)
            end
        end
        return sparse.(perms)
    end
end

function is_isomorphic(G1::Graph, G2::Graph; permutations=nothing)
    (nv(G1) != nv(G2)) && return false
    (ne(G1) != ne(G2)) && return false
    isnothing(permutations) && (permutations = get_permutations(nv(G1)))
    A1 = adjacency_matrix(G1)
    A2 = adjacency_matrix(G2)
    for P in permutations
        A1_perm = P * A1 * transpose(P)
        if A1_perm == A2
            return true
        end
    end
    return false
end

"""
    find_networks(n; nb_to_find=nothing, max_attempts=nothing, β=0.1, γ=0.05)
"""
function find_networks(n; nb_to_find=nothing, max_attempts=nothing, β=0.1, γ=0.05)
    found_networks = Graph[]
    max_networks = 2^(n * (n - 1) ÷ 2) # Maximum number of graphs
    nb_to_find = isnothing(nb_to_find) ? max_networks : nb_to_find
    max_attempts = isnothing(max_attempts) ? max_networks : max_attempts
    combinations = shuffle(1:max_networks)
    permutations = get_permutations(n)

    @showprogress for i in combinations[1:max_attempts]
        network = unpack_network(n, i)
        !(is_connected(network)) && continue
        if any(is_isomorphic(network, existing; permutations=permutations) for existing in found_networks)
            continue
        end
        H = get_SIS_H(adjacency_matrix(network), β, γ)
        eigs = eigvals(Matrix(H))
        if all(imag.(eigs) .<= sqrt(eps()))  # Check for real eigenvalues
            push!(found_networks, network)
            @info "Found network $(length(found_networks)) with index $i"
            if length(found_networks) >= nb_to_find
                break
            end
        end
    end

    return found_networks
end

"""
    check_permutations(A)

Check how many permutations of matrix A result in the same matrix.
"""
function check_permutations(A; permutations=nothing)
    n = size(A, 1)
    (isnothing(permutations)) && (permutations = get_permutations(n))
    count = 0
    for P in permutations
        A_perm = P * A * transpose(P)
        if A_perm == A
            count += 1
        end
    end
    return count
end

networks = find_networks(6)

network = networks[1]
H = get_SIS_H(adjacency_matrix(network), 0.1, 0.05)
maximum(imag.(eigvals(Matrix(H))))
adjacency_matrix(network)
network10 = deepcopy(network)
add_edge!(network10, 1, 2)
H = get_SIS_H(adjacency_matrix(network10), 0.1, 0.01)
maximum(imag.(eigvals(Matrix(H))))
adjacency_matrix(network10)
