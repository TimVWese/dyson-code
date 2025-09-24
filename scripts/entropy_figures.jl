"""
    scripts/entropy_figures.jl

This script generates Figures 1, showing the entropy dynamics of a SIS model in its original and transformed form.

MIT License

Copyright (c) 2025 Tim Van Wesemael, Gilberto Nakamura
"""

include(joinpath(@__DIR__, "..", "src", "dyson-setup.jl"))

using Random
using LinearAlgebra
using Graphs
using Plots, LaTeXStrings


"""
    generate_homotopy(η, sim_f, agg_f; steps=50)

Generates a homotopy of simulations from the identity matrix to the full Dyson transformation matrix `η`
using the simulation function `sim_f`. The aggregation function `agg_f` is applied to extract a variable
from each timestep in the simulation results.
"""
function generate_homotopy(η, sim_f, agg_f; steps=50)
    eta_f = α -> (1 - α) * I + α * η
    trajectories = []
    for t in range(0, 1, length=steps)
        c_η = eta_f(t)
        sol = sim_f(c_η)
        push!(trajectories, agg_f.(sol.u))
    end
    return hcat(trajectories...)'
end

"""
    create_heatmap

Recreate Figure 1B.
"""
function create_heatmap(trajectories, ; T_max=1.25, plot_size=(450, 270), filename="entropy-heatmap.pdf")
    gr()  # GR backend supports custom fonts

    # Set Computer Modern font (assuming it's installed)
    default(
        fontfamily = "Computer Modern",
        guidefont = font("Computer Modern", 12),
        tickfont = font("Computer Modern", 10),
        legendfont = font("Computer Modern", 10)
    )

    # Plot
    heatmap(
        range(0, T_max; length=size(trajectories, 2)), range(0, 1; length=size(trajectories, 1)), trajectories;
        color=:viridis,
        xlabel=L"$\gamma t$",
        ylabel=L"$\alpha$",
        title=L"$S_{\mathrm{Rényi}}$",
        size=plot_size,
        dpi=480,
        framestyle=:box
    )

    # Save as vector graphic
    savefig(filename)
    return Plots.plot!()
end

"""
    create_entropies_figure

Recreate Figure 1A.
"""
function create_entropies_figure(H, η, sim_f, γ;
        m_colors=cgrad(:viridis)[10:180:end], plot_size=(450, 270), filename="entropy-graphs.pdf")
    gr()  # GR backend supports custom fonts

    # Set Computer Modern font (assuming it's installed)
    default(
        fontfamily = "Computer Modern",
        guidefont = font("Computer Modern", 12),
        tickfont = font("Computer Modern", 10),
        legendfont = font("Computer Modern", 10)
    )

    plot(; xlabel=L"$\gamma t$", ylabel="",size=plot_size, dpi=480, legend_background_color=:transparent, legend_foreground_color=:transparent) #, legend_columns=2)

    sol_o = sim_f(I)
    Plots.plot!(sol_o.t .* γ, Dyson.renyi_entropy.(sol_o.u), label="", color=m_colors[1])
    Plots.scatter!(sol_o.t[1:6:end] .* γ, Dyson.renyi_entropy.(sol_o.u[1:6:end]), label="", color=m_colors[1], marker=:circle, markersize=4)
    Plots.plot!([], [], label=L"$S_\mathrm{Renyi}(t)$", color=m_colors[1], marker=:circle, markersize=4)

    sol_o = sim_f(I)
    Plots.plot!(sol_o.t .* γ, Dyson.shannon_entropy.(sol_o.u), label="", color=m_colors[1], style=:dash)
    Plots.scatter!(sol_o.t[1:6:end] .* γ, Dyson.shannon_entropy.(sol_o.u[1:6:end]), label="", color=m_colors[1], marker=:diamond, markersize=4)
    Plots.plot!([], [], label=L"$S_\mathrm{Shannon}(t)$", color=m_colors[1], marker=:diamond, markersize=4, style=:dash)

    sol_h = sim_f(η; symmetrize=true)
    Plots.plot!(sol_h.t .* γ, Dyson.renyi_entropy.(sol_h.u), label="", color=m_colors[2])
    Plots.scatter!(sol_h.t[1:6:end] .* γ, Dyson.renyi_entropy.(sol_h.u[1:6:end]), label="", color=m_colors[2], marker=:square, markersize=4)
    Plots.plot!([], [], label=L"$S'_\mathrm{Renyi}(t)$", color=m_colors[2], marker=:square, markersize=4)

    savefig(filename)
    return Plots.plot!()
end

# Load data with default initialisation
name, _, _, H, h, η, β, γ, _, _ = default_initialisation()

sim_f = (η; symmetrize=false) -> simulate(H, η; symmetrize=symmetrize, tspan=(0.0, 5.), saveat=0.05)
trajectories = generate_homotopy(η, sim_f, Dyson.renyi_entropy; steps=100)

create_heatmap(trajectories; T_max = 0.25, filename=joinpath(FIGURE_DIR, "$(name)_entropy_heatmap.pdf"))
create_entropies_figure(H, η, sim_f, γ, filename=joinpath(FIGURE_DIR, "$(name)_entropy_graphs.pdf"))
