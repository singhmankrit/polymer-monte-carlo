from random import choice, random
from typing import Callable
import numpy as np
from numpy.typing import NDArray
import logging
from tqdm import trange
from tqdm.contrib.logging import logging_redirect_tqdm

LOG = logging.getLogger(__name__)

def get_allowed_sides_2d(
    chain: NDArray[np.float64], step: int
) -> list[NDArray[np.float64]]:
    current_position = chain[step, :]
    return [
        new_position
        for new_position in [
            current_position + np.array([1, 0]),
            current_position + np.array([-1, 0]),
            current_position + np.array([0, 1]),
            current_position + np.array([0, -1]),
        ]
        if (not (chain[:step] == new_position).all(axis=1).any() or step == 0)
    ]


def get_allowed_sides_2d_free(
    chain: NDArray[np.float64], step: int
) -> list[NDArray[np.float64]]:
    current_position = chain[step, :]
    return [
        current_position + np.array([1, 0]),
        current_position + np.array([-1, 0]),
        current_position + np.array([0, 1]),
        current_position + np.array([0, -1]),
    ]


def get_allowed_sides_3d(
    chain: NDArray[np.float64], step: int
) -> list[NDArray[np.float64]]:
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


def get_allowed_sides_3d_free(
    chain: NDArray[np.float64], step: int
) -> list[NDArray[np.float64]]:
    current_position = chain[step, :]
    return [
        current_position + np.array([1, 0, 0]),
        current_position + np.array([-1, 0, 0]),
        current_position + np.array([0, 1, 0]),
        current_position + np.array([0, -1, 0]),
        current_position + np.array([0, 0, 1]),
        current_position + np.array([0, 0, -1]),
    ]


def do_step(
    chain: NDArray[np.float64],
    weight: NDArray[np.longdouble],
    alive: NDArray[np.bool],
    step: int,
    next_sites_function: Callable[
        [NDArray[np.float64], int], list[NDArray[np.float64]]
    ],
):
    if not alive[step]:
        return
    allowed_sides = next_sites_function(chain, step)
    amount_of_allowed_sides = len(allowed_sides)
    if amount_of_allowed_sides > 0:
        next = choice(allowed_sides)
        chain[step + 1, :] = next
        weight[step + 1] = weight[step] * amount_of_allowed_sides
    else:
        alive[step + 1 :] = False


def init_polymer_storage(
    amount_of_chains: int, target_length: int, dimension: int
) -> tuple[NDArray[np.float64], NDArray[np.longdouble], NDArray[np.bool]]:
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
    return chains, weights, alive


def perm_step(
    chains: NDArray[np.float64],
    weights: NDArray[np.longdouble],
    alive: NDArray[np.bool],
    step: int,
    amount_of_chains: int,
    perm_weights: tuple[float, float],
):
    w_low, w_high = perm_weights
    mean_weight = np.mean(weights[alive[:, step + 1], step + 1])
    to_add: list[int] = []  # keep track of what polymers got duplicated
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
    amount_of_chains: int,
    target_length: int,
    dimension: int,
    next_sides_function: Callable[
        [NDArray[np.float64], int], list[NDArray[np.float64]]
    ],
    do_perm: bool,
    perm_weights: tuple[float, float],
):
    chains, weights, alive = init_polymer_storage(
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
            if not alive[:, step + 1].any():
                LOG.warning(f"All chains died by step {step + 1}, skipping other steps")
                break
            if do_perm:
                chains, weights, alive = perm_step(
                    chains, weights, alive, step, amount_of_chains, perm_weights
                )
            amount_of_chains = chains.shape[0]
            max_step += 1
    return max_step, chains.shape[0], chains, alive, weights
