#!/usr/bin/env python
from random import choice, random
import numpy as np


def get_allowed_sides(chain, step):
    current_position = chain[step, :]
    return (
        [
            new_position
            for new_position in [
                current_position + np.array([1, 0]),
                current_position + np.array([-1, 0]),
                current_position + np.array([0, 1]),
                current_position + np.array([0, -1]),
            ]
            if not (chain[:step] == new_position).all(axis=1).any()
        ]
        if step > 0
        else [
            current_position + np.array([1, 0]),
            current_position + np.array([-1, 0]),
            current_position + np.array([0, 1]),
            current_position + np.array([0, -1]),
        ]
    )


def do_step(chain, weight, alive, step):
    if not alive[step]:
        return
    allowed_sides = get_allowed_sides(chain, step)
    l = len(allowed_sides)
    if l > 0:
        next = choice(allowed_sides)
        chain[step + 1, :] = next
        weight[step + 1] = weight[step] * l
    else:
        alive[step + 1 :] = False


if __name__ == "__main__":
    # TODO: parse configuration
    amount_of_chains = 300
    target_length = 1000
    w_low = 0.33
    w_high = 3.33

    # allow for all three coordinates up to the max length for each chain
    chains = np.zeros((amount_of_chains, target_length, 2))
    # keeps track of whether to keep growing a specific chain or not and the timesteps
    alive = np.tile(True, (amount_of_chains, target_length))
    # weight for each sub-length L for each of the chains
    # uses the long double datatype 'g' to allow for the big number that may appear
    weights = np.zeros((amount_of_chains, target_length), dtype="g")
    weights[:, 0] = 1

    # TODO: iterate on the polymers using PERM instead of just Rosenbluth
    for step in range(target_length - 1):
        for chain in range(amount_of_chains):
            do_step(chains[chain, :, :], weights[chain, :], alive[chain, :], step)

    # TODO: calculate end to end distances

    # TODO: calculate gyrations

    # TODO: visualize things
