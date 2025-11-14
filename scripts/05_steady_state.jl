"""
    scripts/steady_state.jl

This script generates Figure 5, comparing the steady states obtained by dynamically evolving the system and
by optimizing the Dyson transformation.

MIT License

Copyright (c) 2025 Tim Van Wesemael, Gilberto Nakamura
"""

include(joinpath(@__DIR__, "..", "src", "dyson-setup.jl"))

using LinearAlgebra
using SparseArrays
using DifferentialEquations
using Graphs
using Plots, LaTeXStrings
using Optim


function KS_distance(p1, p2)
    cp1 = cumsum(p1)
    cp2 = cumsum(p2)
    return maximum(abs.(cp1 .- cp2))
end

function add_KS_annotation!(p, Peq, H_ss)
    ks_dist = KS_distance(Peq, H_ss)
    # Format the KS distance in scientific notation (x.xx ⋅ 10^y)
    ks_mantissa, ks_exponent = Float64(round(ks_dist / 10.0^floor(log10(ks_dist)), digits=2)), Int(floor(log10(ks_dist)))
    ks_formatted = L"KS = %$(ks_mantissa) \cdot 10^{%$(ks_exponent)}"
    annotate!(p, 0.95*64, 0.9*maximum(Peq), text(ks_formatted, 10, :right), font("Computer Modern", 10))
end

"""
    generate_ss_bar_plot(A, β, γ; tol=1e-5, kmax=5000, incl_disease_free=true,
        filename="ss_bar_plot.pdf", m_colors=cgrad(:viridis)[10:180:end],
        plot_size=(450, 270), use_legend=false, bdg=nothing)

Generate a bar plot comparing the steady state obtained by optimizing the Dyson transformation
and by dynamically evolving the system. Also returns the steady states and the optimization result.
Results are in Figure 5.
"""
function generate_ss_bar_plot(A, β, γ;
        tol=1e-5, kmax=5000, incl_disease_free=true,
        filename="ss_bar_plot.pdf", m_colors=cgrad(:viridis)[10:180:end],
        plot_size=(450, 270), use_legend=false, bdg=nothing,
    )

    gr()  # GR backend supports custom fonts
    # Set Computer Modern font (assuming it's installed)
    default(
        fontfamily = "Computer Modern",
        guidefont = font("Computer Modern", 10),
        tickfont = font("Computer Modern", 10),
        legendfont = font("Computer Modern", 8),
        titlefont = font("Computer Modern", 10),
    )

    H = Dyson.get_SIS_H(A, β, γ; incl_disease_free)
    h, η, _ = Dyson.find_hermitian(Matrix(H); kmax)
    iη = inv(η)

    loss = phi -> phi' * h * phi
    x0 = ones(size(H, 1))
    result = optimize(loss, x0, ConjugateGradient(), Optim.Options(iterations=10*kmax, x_abstol=tol))
    Peq = iη * result.minimizer
    Peq ./= sum(Peq)
    ϕ_on = η * Peq

    # Create a figure with two vertically stacked subplots
    isnothing(bdg) && (bdg = β / γ)
    p = plot(;
        ylabel=L"$P_\ell$",
        legend_background_color=:transparent,
        legend_foreground_color=:transparent,
        xmin=0,
        xmax=64,
        title=L"\beta/\gamma=" * string(bdg),
        legend=use_legend,
        xformatter = :none,
    )

    # Get the number of states
    n = length(Peq)
    # Create interleaved x positions
    x_pos_opt = (1:n) .- 0.25
    x_pos_dyn = (1:n) .+ 0.25

    # Plot interleaved bars
    bar!(p, x_pos_opt, Peq, label="Optimization", bar_width=0.5, linecolor=nothing, color=m_colors[1])
    H_ss = Dyson.get_steady_state(H)
    bar!(p, x_pos_dyn, H_ss, label="Dynamic", bar_width=0.5, linecolor=nothing, color=m_colors[2])
    add_KS_annotation!(p, Peq, H_ss)

    # Add the second subplot (phi)
    p2 = plot(
        title="",
        xlabel=L"\ell",
        ylabel=L"$\phi_\ell$",
        legend=false,
        xmin=0,
        xmax=64,
    )

    h_ss = Dyson.get_steady_state(h; η=η, symmetrize=false)
    # Plot interleaved bars for phi
    bar!(p2, x_pos_opt, ϕ_on, label="", bar_width=0.5, linecolor=nothing, color=m_colors[1])
    bar!(p2, x_pos_dyn, h_ss, label="", bar_width=0.5, linecolor=nothing, color=m_colors[2])
    add_KS_annotation!(p2, ϕ_on, h_ss)

    # Combine both subplots into one figure with a shared legend
    final_plot = plot(p, p2, layout=(2, 1), size = plot_size, xlim=(0, 64),)

    # Save the combined figure
    savefig(filename)
    @info "for β = $β, γ = $γ"
    println("Ks P: $(KS_distance(Peq, H_ss))")
    println("Ks ϕ: $(KS_distance(ϕ_on, h_ss))")
    return plot!(), result, Peq, H_ss, ϕ_on, h_ss
end

name, _, A, _, _, _, _, γ, incl_disease_free, errors = default_initialisation()
kmax = length(errors)

generate_ss_bar_plot(A, 1e-2 * γ, γ; kmax, incl_disease_free, filename=joinpath(FIGURE_DIR, "$(name)_SIS_ss_b001.pdf"), use_legend=:bottomright, bdg=L"10^{-2}")
generate_ss_bar_plot(A, 10^-.5 * γ, γ; kmax, incl_disease_free, filename=joinpath(FIGURE_DIR, "$(name)_SIS_ss_b2.pdf"), bdg=L"10^{-0.5}")
generate_ss_bar_plot(A, 2e1 * γ, γ; kmax, incl_disease_free, filename=joinpath(FIGURE_DIR, "$(name)_SIS_ss_b200.pdf"), bdg=L"20")
