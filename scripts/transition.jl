"""
    transition.jl

Script to generate Figure 4, showing the transition of various observables as a
function of the infection proportion β/γ.

MIT License

Copyright (c) 2025 Tim Van Wesemael, Gilberto Nakamura
"""

include(joinpath(@__DIR__, "..", "dyson-setup.jl"))

using LinearAlgebra
using SparseArrays
using DifferentialEquations
using Graphs
using ProgressMeter
using JLD2
using Plots, LaTeXStrings

function generate_transition_data(A; γ=5e-2, β_range=(-0.8:0.005:1.5), tol=1e-6, kmax=5000, incl_disease_free=true)
    n = size(A, 1)
    N = 2^n - (1 - incl_disease_free)

    row_indices, col_indices, values = Dyson.get_SIS_H_structure(A)
    O_mean = Dyson.infectious_proportion_O(n; incl_disease_free)
    O_msq = O_mean .^ 2
    βs = γ * 10 .^ β_range

    valid = [maximum(abs.(imag.(eigvals(Matrix(Dyson.get_SIS_H(A, β, γ)))))) < 1e-8 for β in βs]
    @info "$(count(valid)) of $(length(βs)) β values have real eigenvalues"

    ratios = βs ./ γ
    undef_array = () -> zeros(Float64, length(βs))
    props = undef_array()
    entropies = undef_array()
    vars = undef_array()

    ϕ_entropies = undef_array()
    ηs = [zeros(Float64, N, N) for _ in βs]

    pb = Progress(length(βs))
    Threads.@threads for i in eachindex(βs)
        β = βs[i]

        H = get_SIS_H(row_indices, col_indices, values, β, γ; incl_disease_free)
        p = get_steady_state(H)
        props[i] = observe(O_mean, p)
        entropies[i] = renyi_entropy(p)
        I_sq = observe(O_msq, p)
        vars[i] = I_sq - props[i]^2

        @assert maximum(abs.(imag.(eigvals(Matrix(H))))) < 1e-8 "Hermitian matrix has non-zero imaginary eigenvalues: $H"
        try
            h, η, _ = find_hermitian(Matrix(H); tol, kmax)
            ϕ = get_steady_state(h; η=η, symmetrize=false)
            ϕ_entropies[i] = renyi_entropy(ϕ)
            ηs[i] .= η
        catch e
            @warn "Failed to find hermitian matrix for β = $β: $(e)"
            ϕ_entropies[i] = NaN
        end
        next!(pb)
    end

    # Save the results
    return ratios, props, entropies, vars, ϕ_entropies, ηs
end

function plot_transition_data(ratios, props, entropies, vars, ϕ_entropies;
                             marker_step=12, sylim=(-0.05, 4.25),
                             plot_size=(450, 270),
                             output_file="transition_plot.pdf")
    # Load the data
    selection = eachindex(ratios)
    colors = cgrad(:viridis)[10:45:end]
    marker_selection = first(selection):marker_step:last(selection)

    gr()  # GR backend supports custom fonts

    # Set Computer Modern font (assuming it's installed)
    default(
        fontfamily = "Computer Modern",
        guidefont = font("Computer Modern", 12),
        tickfont = font("Computer Modern", 10),
        legendfont = font("Computer Modern", 10)
    )

    plot(;
        plot_size=plot_size, xaxis=:log, xlabel=L"\beta / \gamma", legend=:topright,
        dpi=480, legend_background_color=:transparent, legend_foreground_color=:transparent
    )

    # Plot infection proportion
    plot!(ratios[selection], props[selection]; ylabel=L"\mathrm{mean}, \sigma", c=colors[1], label="", lw=2)
    scatter!(ratios[marker_selection], props[marker_selection]; c=colors[1], label="", marker=:square)
    plot!([], []; label=L"\langle \rho_I \rangle_P", c=colors[1], marker=:square, markersize=4)

    # Plot standard deviation
    plot!(ratios[selection], sqrt.(vars[selection]); c=colors[2], label="", lw=2, style=:solid)
    scatter!(ratios[marker_selection], sqrt.(vars[marker_selection]); c=colors[2], label="", marker=:diamond)
    plot!([], []; label=L"\sigma_P", c=colors[2], style=:solid, marker=:diamond, markersize=4)

    # Plot entropies
    plot!(twinx(), ratios[selection], entropies[selection]; ylabel=L"S", xaxis=:log, c=colors[end-1], label="", lw=2, ylim=sylim)
    scatter!(twinx(), ratios[marker_selection], entropies[marker_selection]; c=colors[end-1], xaxis=:log, label="", marker=:utriangle, yaxis=false, ylim=sylim)
    plot!([], []; label=L"S_\mathrm{Renyi}", c=colors[end-1], marker=:utriangle, markersize=4)

    plot!(twinx(), ratios[selection], ϕ_entropies[selection]; c=colors[end], xaxis=:log, style=:solid, label="", lw=2, yaxis=false, ylim=sylim)
    scatter!(twinx(), ratios[marker_selection], ϕ_entropies[marker_selection]; c=colors[end], xaxis=:log, label="", marker=:dtriangle, yaxis=false, ylim=sylim)
    plot!([], []; label=L"S'_\mathrm{Renyi}", c=colors[end], style=:solid, marker=:dtriangle, markersize=4)

    savefig(output_file)
end

name, _, A, _, _, _, _, γ, incl_disease_free, errors = default_initialisation()
kmax = length(errors)

ratios, props, entropies, vars, ϕ_entropies, ηs = generate_transition_data(A; γ, β_range=(-2.1:0.0125:1.8), kmax, incl_disease_free)
output_file = joinpath(@__DIR__, "..", "results", "intermediate", "transition_points_" * name * ".jld2")
JLD2.@save output_file ratios props entropies vars ϕ_entropies ηs
JLD2.@load output_file ratios props entropies vars ϕ_entropies ηs
plot_transition_data(ratios, props, entropies, vars, ϕ_entropies;
                     marker_step=12, plot_size=(450, 270), output_file=joinpath(FIGURE_DIR, "$(name)_transition.pdf"))
