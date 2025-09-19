# dyson-code
Julia codebase and scripts to accompany the paper *Restoring thermodynamic structure in stochastic dynamics*.

## How to use

In `src/` the main functionality is written.
The `scripts/` folder contains script that use this functionality to generate the figures as shown in the manuscript.
Run `run_all.sh` as a bash script to initialise the Julia project and run all scripts.
The resulting figures can be found under `results/figures`.
To test other set-ups, the `default_initialisation()` function in `src/dyson.jl` can be edited.

## Figure index:

* Figure 1: `scripts/entropy_figures.jl`
* Figure 2: `scripts/H_figure.jl`
* Figure 3: `scripts/statistics.jl`
* Figure 4: `scripts/transition.jl`
* Figure 5: `scripts/steady_state.jl`
* Figure 6: `scripts/convergence.jl`

## Reproducibility

Each figure is generated with a seed started at 1234 as defined in `dyson-setup`.
The results in the manuscript are generated with `julia v1.11.7` aand the package versions as defined in the `[compat]` section of `Project.toml`.
