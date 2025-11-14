"""
This script recreates Figure 6, as found in the appendix.
"""

include(joinpath(@__DIR__, "..", "src", "dyson-setup.jl"))
using Plots, LaTeXStrings

"""
    create_convergence_plot

Recreate Figure 6, a convergence plot of the numerical algorithm.
"""
function create_convergence_plot(errors; m_colors=cgrad(:viridis)[10:180:end], plot_size=(450, 270), filename="convergence.pdf")
    gr()  # GR backend supports custom fonts

    # Set Computer Modern font (assuming it's installed)
    default(
        fontfamily = "Computer Modern",
        guidefont = font("Computer Modern", 12),
        tickfont = font("Computer Modern", 10),
        legendfont = font("Computer Modern", 10)
    )

    p = plot(
        collect(eachindex(errors)) ./ 1e3, errors;
        lw=2, c=m_colors[1], xlabel=L"k \quad \left(\cdot 10^3\right)",
        ylabel=L"\tau_k", yaxis=:log, legend=false, size=plot_size,
    )
    savefig(filename)
    return p
end

name, _, _, _, _, _, _, _, _, errors = default_initialisation()
create_convergence_plot(errors; filename=joinpath(FIGURE_DIR, "$(name)_convergence.pdf"))
