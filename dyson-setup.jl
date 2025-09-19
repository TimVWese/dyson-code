"""
    dyson-setup.jl

This file sets up the Dyson.jl package for use within the figure generating scripts.
"""

using Pkg
Pkg.activate(joinpath(@__DIR__))
include(joinpath(@__DIR__, "src", "dyson.jl"))
using .Dyson

using Random
Random.seed!(1234)  # For reproducibility

global FIGURE_DIR = joinpath(@__DIR__, "results", "figures")
isdir(FIGURE_DIR) || mkpath(FIGURE_DIR)
