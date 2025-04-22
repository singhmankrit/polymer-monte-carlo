#!/usr/bin/env python
import numpy as np
from code import utilities, simulate, plots, observables

(amount_of_chains, target_length, do_perm, w_low, w_high, dimension, next_sides_function) = utilities.parse_config("config.json")

assert next_sides_function != simulate.get_allowed_sides_2d or dimension == 2
assert next_sides_function != simulate.get_allowed_sides_2d_free or dimension == 2
assert next_sides_function != simulate.get_allowed_sides_3d or dimension == 3
assert next_sides_function != simulate.get_allowed_sides_3d_free or dimension == 3

max_step, amount_of_chains, chains, alive, weights = simulate.grow_polymers(
    amount_of_chains,
    target_length,
    dimension,
    next_sides_function,
    do_perm,
    (w_low, w_high),
)

print(max_step, amount_of_chains)

end_to_ends, gyrations = observables.find_observables(amount_of_chains, max_step, chains, alive)

lengths = np.arange(0, max_step)
weighted_end_to_end = np.sum(end_to_ends * weights[:, :max_step], axis=0) / np.sum(
    weights[:, :max_step], axis=0
)
plots.plot_end_to_end(lengths, end_to_ends,
                weights[:, :max_step], dimension, alive, max_step)

weighted_gyrations = np.sum(gyrations * weights[:, :max_step], axis=0) / np.sum(
    weights[:, :max_step], axis=0
)
plots.plot_gyration(lengths, gyrations, weights[:, :max_step], dimension, alive, max_step)
