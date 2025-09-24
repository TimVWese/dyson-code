"""
    scripts/statistics.jl

Generates Figure 3 of the manuscript, showing the time evolution of the mean and standard deviation

MIT License

Copyright (c) 2025 Tim Van Wesemael, Gilberto Nakamura
"""

include(joinpath(@__DIR__, "..", "src", "dyson-setup.jl"))

using LinearAlgebra
using JLD2
using Plots
using LaTeXStrings

"""
    create_statistics_plot(H, η, γ;
        incl_disease_free=true, m_colors=cgrad(:viridis)[10:180:end],
        plot_size=(450, 270), filename="statistics.pdf"
    )

Generate Figure 3 of the manuscript, showing the time evolution of the mean and standard deviation
in the original and transformed systems.
"""
function create_statistics_plot(H, η, γ;
        incl_disease_free=true, m_colors=cgrad(:viridis)[10:180:end],
        plot_size=(450, 270), filename="statistics.pdf"
    )
    gr()  # GR backend supports custom fonts

    # Set Computer Modern font (assuming it's installed)
    default(
        fontfamily = "Computer Modern",
        guidefont = font("Computer Modern", 12),
        tickfont = font("Computer Modern", 10),
        legendfont = font("Computer Modern", 10)
    )

    N = size(H, 1)
    n = Int64(log2(N + (1 + incl_disease_free)))
    Ξ = ones(N, N)

    Ps = Dyson.simulate(H, I; symmetrize=false, tspan=(0.0, 3.5), saveat=0.05)
    ϕs = Dyson.simulate(H, η; symmetrize=true, tspan=(0.0, 3.5), saveat=0.05)

    O_mean = infectious_proportion_O(n; incl_disease_free)
    O_sq = O_mean^2

    iη = inv(η)
    Opm = η * Ξ * O_mean * iη
    Opsq = η * Ξ * O_sq * iη
    Ωt = inv(η * η')

    Ipm = [Dyson.observe(O_mean, u) for u in Ps.u]
    Ipsq = [Dyson.observe(O_sq, u) for u in Ps.u]
    Ipvar = Ipsq .- Ipm.^2

    Ifm = [Dyson.observe(Opm, u, Ωt) for u in ϕs.u]
    Ifsq = [Dyson.observe(Opsq, u, Ωt) for u in ϕs.u]
    Ifvar = Ifsq .- Ifm.^2

    # Plotting
    plot(; xlabel=L"$\gamma t$", ylabel="mean",size=plot_size, dpi=480)
    # Create plot with the mean values on the left y-axis
    plot!(Ps.t .* γ, Ipm, label=L"\langle \rho_I \rangle_P", color=m_colors[1], lw=2)
    scatter!(Ps.t[1:4:end] .* γ, Ifm[1:4:end], label=L"\langle \rho_I \rangle_\phi", color=m_colors[1], marker=:circle, markersize=4)

    plot!([],[]; label=L"\sigma_P", color=m_colors[2], lw=2)  # Dummy plot for legend spacing
    scatter!([], []; label=L"\sigma_\phi", color=m_colors[2], marker=:square)
    plot!(; legend_columns=2, legend=(0.68, 0.2), legend_background_color=:transparent, legend_foreground_color=:transparent)

    # Create a secondary y-axis on the right for standard deviations
    plot!(twinx(), Ps.t .* γ, sqrt.(Ipvar), label="", color=m_colors[2], ylabel=L"\sigma", lw=2)
    scatter!(twinx(), Ps.t[1:4:end] .* γ, sqrt.(abs.(Ifvar[1:4:end])), label="", color=m_colors[2], marker=:square, markersize=4, yaxis=false)

    savefig(filename)
    return plot!()
end

name, _, _, H, _, η, _, γ, incl_disease_free, _ = default_initialisation()

create_statistics_plot(H, η, γ; incl_disease_free, filename=joinpath(FIGURE_DIR, "$(name)_statistics_full.pdf"))
