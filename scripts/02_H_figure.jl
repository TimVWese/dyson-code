"""
    scripts/H_figure.jl

This script generates Figure 2B, a heatmap of the structure of the SIS H matrix.

MIT License

Copyright (c) 2025 Tim Van Wesemael, Gilberto Nakamura
"""

include(joinpath(@__DIR__, "..", "src", "dyson-setup.jl"))

using Graphs
using LinearAlgebra
using SparseArrays
using Images


name, g, A, H, _, _, _, _, incl_disease_free, _ = n6e9_initialisation()

row_indices, col_indices, vals = Dyson.get_SIS_H_structure(A)

cs_heal = range(colorant"white", RGB{N0f8}(39/255, 171/255, 173/255), length=3)
cs_inf = range(colorant"white", RGB{N0f8}(232/255, 94/255, 113/255), length=6)

c_vals = RGB{N0f8}[]
for v in vals
    if v < 0
        push!(c_vals, cs_heal[2])
    elseif v > 0
        push!(c_vals, cs_inf[v+1])
    else
        push!(c_vals, RGB{N0f8}(1, 1, 1))
    end
end

vals[vals .> 0] .*= 2
vals[vals .< 0] .*= -1
H = sparse(row_indices, col_indices, vals)
!incl_disease_free && (H = H[2:end, 2:end]) # Remove disease-free state
for i in axes(H, 1)
    H[i, i] = sum(H[i, :])
end
cs_diag = range(colorant"white", RGB{N0f8}(241/255, 164/255, 43/255), length=maximum(diag(H))+1)

H_img = Matrix(sparse(row_indices, col_indices, c_vals))
!incl_disease_free && (H_img = H_img[2:end, 2:end]) # Remove disease-free state
H_img[H_img .== RGB{N0f8}(0,0,0)] .= RGB{N0f8}(1,1,1)
for i in axes(H, 1)
    H_img[i, i] = cs_diag[H[i,i]+1]
end

save(joinpath(FIGURE_DIR, "$(name)-SIS_H_heatmap.pdf"), H_img)
