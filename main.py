#!/usr/bin/env python
import numpy as np
from polymer_code import utilities, simulate, plots, observables

(
    amount_of_chains,
    target_length,
    do_perm,
    w_low,
    w_high,
    next_sides_function,
    to_plot,
    seed,
    threshold,
) = utilities.parse_config("config.json")

if next_sides_function in [
    simulate.get_allowed_sides_2d,
    simulate.get_allowed_sides_2d_free,
    simulate.get_allowed_sides_triangle,
    simulate.get_allowed_sides_hexagon,
]:
    dimension = 2
elif next_sides_function in [
    simulate.get_allowed_sides_3d,
    simulate.get_allowed_sides_3d_free,
]:
    dimension = 3
else:
    raise ValueError("next_sides_function does not have a defined dimension")

max_step, amount_of_chains, chains, alive, weights = simulate.grow_polymers(
    amount_of_chains,
    target_length,
    dimension,
    next_sides_function,
    do_perm,
    (w_low, w_high),
    seed,
    threshold,
)

if "e2e" in to_plot or "gyration" in to_plot:
    end_to_ends, gyrations = observables.find_observables(
        amount_of_chains, max_step, chains, alive
    )

shape = "Squared"
if next_sides_function == simulate.get_allowed_sides_triangle:
    shape = "Triangular"
elif next_sides_function == simulate.get_allowed_sides_hexagon:
    shape = "Hexagonal"

lengths = np.arange(0, max_step)
if "e2e" in to_plot:
    weighted_end_to_end = np.sum(end_to_ends * weights, axis=0) / np.sum(
        weights, axis=0
    )
    plots.plot_end_to_end(
        lengths, end_to_ends, weights, dimension, alive, max_step, shape
    )

if "gyration" in to_plot:
    weighted_gyrations = np.sum(gyrations * weights, axis=0) / np.sum(weights, axis=0)
    plots.plot_gyration(lengths, gyrations, weights, dimension, alive, max_step, shape)

if "animation" in to_plot:
    # the indices to plot (1 is the longest, 2 the second longest, etc.)
    idxs = np.array([i for i in range(1, 17)])
    plots.plot_animation(chains, alive, idxs, dimension)
