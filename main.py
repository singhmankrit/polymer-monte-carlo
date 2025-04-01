#!/usr/bin/env python
from random import choice, random
import numpy as np
import matplotlib.pyplot as plt
import scipy.optimize as opt
import logging
from tqdm import trange
from tqdm.contrib.logging import logging_redirect_tqdm

LOG = logging.getLogger(__name__)


def get_allowed_sides_2d(chain, step):
    current_position = chain[step, :]
    return [
        new_position
        for new_position in [
            current_position + np.array([1, 0, 0]),
            current_position + np.array([-1, 0, 0]),
            current_position + np.array([0, 1, 0]),
            current_position + np.array([0, -1, 0]),
            current_position + np.array([0, 0, 1]),
            current_position + np.array([0, 0, -1]),
        ]
        if (not (chain[:step] == new_position).all(axis=1).any() or step == 0)
    ]


def do_step(chain, weight, alive, step, next_sites_function):
    if not alive[step]:
        return
    allowed_sides = next_sites_function(chain, step)
    l = len(allowed_sides)
    if l > 0:
        next = choice(allowed_sides)
        chain[step + 1, :] = next
        weight[step + 1] = weight[step] * l
    else:
        alive[step + 1 :] = False


def init_polymer_storage(amount_of_chains, target_length, dimension):
    # since we want length L we'll have L+1 points
    target_length += 1
    # allow for all three coordinates up to the max length for each chain
    chains = np.zeros((amount_of_chains, target_length, dimension))
    # keeps track of whether to keep growing a specific chain or not and the timesteps
    alive = np.tile(True, (amount_of_chains, target_length))
    # weight for each sub-length L for each of the chains
    # uses the long double datatype 'g' (probably an 80 bit float) to allow for the big numbers that may appear
    weights = np.zeros((amount_of_chains, target_length), dtype="g")
    weights[:, 0] = 1
    return chains, alive, weights


def perm_step(chains, weights, alive, step, amount_of_chains, perm_weights):
    w_low, w_high = perm_weights
    mean_weight = np.mean(weights[alive[:, step + 1], step + 1])
    to_add = []  # keep track of what polymers got duplicated
    pruned = 0  # keep track of how many polymers got pruned this step
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
                # double the weight from L' onwards
                weights[chain, step + 1] *= 2
        # Enrichment
        elif weights[chain, step + 1] > w_high * mean_weight:
            weights[chain, step + 1] /= 2
            to_add.append(chain)
    chains = np.concatenate(
        [chains] + [chains[np.newaxis, chain, :, :] for chain in to_add],
        axis=0,
    )
    weights = np.concatenate(
        [weights] + [weights[np.newaxis, chain, :] for chain in to_add],
        axis=0,
    )
    alive = np.concatenate(
        [alive] + [alive[np.newaxis, chain, :] for chain in to_add],
        axis=0,
    )
    return chains, weights, alive


def grow_polymers(
    amount_of_chains,
    target_length,
    dimension,
    next_sides_function,
    do_perm,
    perm_weights,
):
    chains, alive, weights = init_polymer_storage(
        amount_of_chains, target_length, dimension
    )
    with logging_redirect_tqdm():
        max_step = 1  # we start at 1 point existing (the start)
        for step in trange(target_length):
            for chain in range(amount_of_chains):
                do_step(
                    chains[chain, :, :],
                    weights[chain, :],
                    alive[chain, :],
                    step,
                    next_sides_function,
                )

            # we use step+1 to get the L'
            if (alive[:, step + 1] == False).all():
                LOG.warning(f"All chains died by step {step + 1}, skipping other steps")
                break
            if do_perm:
                chains, weights, alive = perm_step(
                    chains, weights, alive, step, amount_of_chains, perm_weights
                )
            amount_of_chains = chains.shape[0]
            max_step += 1
    return max_step, chains.shape[0], chains, alive, weights


def growth_model(L, A):
    return A * L ** (3 / 2)


def plot_gyration(lengths, weighted_gyrations):
    fig, ax = plt.subplots()

    ax.set_title("Length dependent radius of Gyration")
    ax.set_xlabel(r"length ($\sigma$)")
    ax.set_ylabel(r"Radius of Gyration ($\sigma$)")
    ax.plot(lengths, weighted_gyrations, label="gyration")
    opt_params, _ = opt.curve_fit(growth_model, lengths, weighted_gyrations)
    ax.plot(
        lengths,
        opt_params[0] * lengths ** (3 / 2),
        label=f"${opt_params[0]:.03f} L^{{3 / 2}}$",
    )

    ax_right = ax.twinx()
    ax_right.set_ylabel("amount of polymers")
    ax_right.set_yscale("log")
    ax_right.plot(lengths, np.sum(alive[:, :max_step], axis=0))

    ax.legend()

    plt.tight_layout()
    fig.savefig("gyration.png")


def plot_end_to_end(lengths, weighted_end_to_end):
    fig, ax = plt.subplots()

    ax.set_title("Length dependent end-to-end distance")
    ax.set_xlabel(r"L ($\sigma$)")
    ax.set_ylabel(r"end to end dist ($\sigma^2$)")
    ax.plot(lengths, weighted_end_to_end, label="end-to-end")
    opt_params, _ = opt.curve_fit(growth_model, lengths, weighted_end_to_end)
    ax.plot(
        lengths,
        opt_params[0] * lengths ** (3 / 2),
        label=f"${opt_params[0]:.03f} L^{{3 / 2}}$",
    )

    ax_right = ax.twinx()
    ax_right.set_ylabel("amount of polymers")
    ax_right.set_yscale("log")
    ax_right.plot(lengths, np.sum(alive[:, :max_step], axis=0))

    ax.legend()

    plt.tight_layout()
    fig.savefig("end_to_end.png")


if __name__ == "__main__":
    # TODO: parse configuration
    amount_of_chains = 300
    target_length = 1000
    w_low = 1 / np.sqrt(10)
    w_high = np.sqrt(10)
    do_perm = True
    dimension = 2
    next_sides_function = get_allowed_sides_2d
    assert next_sides_function != get_allowed_sides_2d or dimension == 2
    # example for other assertions to enforce the correct dimension for next_sides_function
    # assert next_sides_function != get_allowed_sides_3d or dimension == 3

    max_step, amount_of_chains, chains, alive, weights = grow_polymers(
        amount_of_chains,
        target_length,
        dimension,
        next_sides_function,
        do_perm,
        (w_low, w_high),
    )

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

    lengths = np.arange(0, max_step)
    weighted_end_to_end = np.sum(end_to_ends * weights[:, :max_step], axis=0) / np.sum(
        weights[:, :max_step], axis=0
    )
    plot_end_to_end(lengths, weighted_end_to_end)

    weighted_gyrations = np.sum(gyrations * weights[:, :max_step], axis=0) / np.sum(
        weights[:, :max_step], axis=0
    )
    plot_gyration(lengths, weighted_gyrations)
