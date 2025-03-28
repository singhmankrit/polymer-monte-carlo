#!/usr/bin/env python
from random import choice, random
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm


def get_allowed_sides(chain, step):
    current_position = chain[step, :]
    if step == 0:
        return [
            current_position + np.array([1, 0]),
            current_position + np.array([-1, 0]),
            current_position + np.array([0, 1]),
            current_position + np.array([0, -1]),
        ]
    return [
        new_position
        for new_position in [
            current_position + np.array([1, 0]),
            current_position + np.array([-1, 0]),
            current_position + np.array([0, 1]),
            current_position + np.array([0, -1]),
        ]
        if not (chain[:step] == new_position).all(axis=1).any()
    ]


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
    w_low = 1 / np.sqrt(10)
    w_high = np.sqrt(10)

    # allow for all three coordinates up to the max length for each chain
    chains = np.zeros((amount_of_chains, target_length + 1, 2))
    # keeps track of whether to keep growing a specific chain or not and the timesteps
    alive = np.tile(True, (amount_of_chains, target_length + 1))
    # weight for each sub-length L for each of the chains
    # uses the long double datatype 'g' (probably an 80 bit float) to allow for the big numbers that may appear
    weights = np.zeros((amount_of_chains, target_length + 1), dtype="g")
    weights[:, 0] = 1
    max_step = 1

    for step in tqdm(range(target_length)):
        for chain in range(amount_of_chains):
            do_step(chains[chain, :, :], weights[chain, :], alive[chain, :], step)

        # we use step+1 to get the L'
        if (alive[:, step + 1] == False).all():
            print(f"All chains died by step {step + 1}, skipping other steps")
            break
        mean_weight = np.mean(weights[alive[:, step + 1], step + 1])
        to_add = []
        pruned = 0
        for chain in range(amount_of_chains):
            if not alive[chain, step + 1]:
                continue
            # Pruning
            if weights[chain, step + 1] < w_low * mean_weight:
                if random() < 0.5:
                    pruned += 1
                    # don't grow this chain anymore
                    alive[chain, step + 1] = False
                    # discard the weight at length L'
                    weights[chain, step + 1] = 0
                else:
                    weights[chain, step + 1] *= 2
            # Enrichment
            elif weights[chain, step + 1] > w_high * mean_weight:
                # TODO: double and half weight
                weights[chain, step + 1] /= 2
                to_add.append(chain)
        chains = np.concatenate(
            [chains] + [np.expand_dims(chains[chain, :, :], 0) for chain in to_add],
            axis=0,
        )
        weights = np.concatenate(
            [weights] + [np.expand_dims(weights[chain, :], 0) for chain in to_add],
            axis=0,
        )
        alive = np.concatenate(
            [alive] + [np.expand_dims(alive[chain, :], 0) for chain in to_add], axis=0
        )
        amount_of_chains = chains.shape[0]
        max_step += 1

    print(max_step, amount_of_chains)
    end_to_ends = np.zeros((amount_of_chains, max_step))
    gyrations = np.zeros((amount_of_chains, max_step))
    for chain in range(amount_of_chains):
        start = chains[chain, 0, :]
        end = chains[chain, :, :]
        diff = end - start
        end_to_ends[chain, alive[chain, :max_step]] = np.vecdot(diff, diff)[
            alive[chain]
        ]

        # the middle point (coordinates axis 1) up to length (axis 0)
        center = np.array(
            [
                np.sum(chains[chain, :length, :], axis=0, keepdims=True) / length
                for length in range(1, max_step + 1)
            ]
        )
        # differences from center (coordinates axis 2, length axis 1, particle axis 0)
        cdiffs = np.array(
            [chains[chain, length, :] - center for length in range(0, max_step)]
        )[:, :, 0, :]

        # distance per length (axis 1), per particle (axis 0)
        clens = np.sum(cdiffs * cdiffs, axis=-1)
        # this can probably be done better (using masking or triu maybe) but works for now
        waa = np.array(
            [
                np.sum(clens[: length + 1, length]) / (length + 1)
                for length in range(0, max_step)
            ]
        )
        gyrations[chain, alive[chain, :max_step]] = waa[alive[chain, :max_step]]

    weighted_end_to_end = np.sum(end_to_ends * weights[:, :max_step], axis=0) / np.sum(
        weights[:, :max_step], axis=0
    )

    fig, ax = plt.subplots()
    lengths = np.arange(0, max_step)

    ax.set_xlabel("L (N)")
    ax.set_ylabel("end to end dist^2")
    ax.plot(lengths, weighted_end_to_end)
    ax.plot(lengths, lengths ** (3 / 2))

    ax_right = ax.twinx()
    ax_right.set_ylabel("amount of polymers")
    ax_right.set_yscale("log")
    ax_right.plot(lengths, np.sum(alive[:, :max_step], axis=0))

    plt.show()

    weighted_gyrations = np.sum(gyrations * weights[:, :max_step], axis=0) / np.sum(
        weights[:, :max_step], axis=0
    )

    fig, ax = plt.subplots()
    lengths = np.arange(0, max_step)

    ax.set_xlabel("L (N)")
    ax.set_ylabel("Gyration")
    ax.plot(lengths, weighted_gyrations)
    ax.plot(lengths, 0.1 * lengths ** (3 / 2))

    ax_right = ax.twinx()
    ax_right.set_ylabel("amount of polymers")
    ax_right.set_yscale("log")
    ax_right.plot(lengths, np.sum(alive[:, :max_step], axis=0))

    plt.show()

    # TODO: visualize things
