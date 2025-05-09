import random
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
    """
    create a list of allowed next positions for a 2 dimensional grid with self-avoidance

    Parameters
        chain (ndarray): array of positions up to the current step
        step (int): the current step

    Returns
        list: list of valid coordinates for the next step
    """
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
    """
    create a list of allowed next positions for a 2 dimensional grid random walk with self-intersection

    Parameters
        chain (ndarray): array of positions up to the current step
        step (int): the current step

    Returns
        list: list of valid coordinates for the next step
    """
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
    """
    create a list of allowed next positions for a 3 dimensional grid with self-avoidance

    Parameters
        chain (ndarray): array of positions up to the current step
        step (int): the current step

    Returns
        list: list of valid coordinates for the next step
    """
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
    """
    create a list of allowed next positions for a 3 dimensional grid random walk with self-intersection

    Parameters
        chain (ndarray): array of positions up to the current step
        step (int): the current step

    Returns
        list: list of valid coordinates for the next step
    """
    current_position = chain[step, :]
    return [
        current_position + np.array([1, 0, 0]),
        current_position + np.array([-1, 0, 0]),
        current_position + np.array([0, 1, 0]),
        current_position + np.array([0, -1, 0]),
        current_position + np.array([0, 0, 1]),
        current_position + np.array([0, 0, -1]),
    ]


def get_allowed_sides_triangle(
    chain: NDArray[np.float64], step: int
) -> list[NDArray[np.float64]]:
    """
    create a list of allowed next positions for a 2 dimensional triangular grid with self-avoidance

    Parameters
        chain (ndarray): array of positions up to the current step
        step (int): the current step

    Returns
        list: list of valid coordinates for the next step
    """
    current_position = chain[step, :]
    sqrt_3_by_2 = np.sqrt(3).round(4) / 2
    tolerance = 1e-3
    return [
        new_position
        for new_position in [
            current_position + np.array([0, 1]),
            current_position + np.array([0, -1]),
            current_position + np.array([sqrt_3_by_2, 1 / 2]),
            current_position + np.array([sqrt_3_by_2, -1 / 2]),
            current_position + np.array([-sqrt_3_by_2, 1 / 2]),
            current_position + np.array([-sqrt_3_by_2, -1 / 2]),
        ]
        if (
            not np.any(np.all(np.abs(chain[:step] - new_position) < tolerance, axis=1))
            or step == 0
        )
    ]


def get_allowed_sides_hexagon(
    chain: NDArray[np.float64], step: int
) -> list[NDArray[np.float64]]:
    """
    create a list of allowed next positions for a 2 dimensional hexagonal grid with self-avoidance

    Parameters
        chain (ndarray): array of positions up to the current step
        step (int): the current step

    Returns
        list: list of valid coordinates for the next step
    """
    current_position = chain[step, :]
    sqrt_3_by_2 = np.sqrt(3).round(4) / 2
    tolerance = 1e-3
    step_half = 1 / 2 - step % 2
    return [
        new_position
        for new_position in [
            current_position + np.array([0, -2 * step_half]),
            current_position + np.array([sqrt_3_by_2, step_half]),
            current_position + np.array([-sqrt_3_by_2, step_half]),
        ]
        if (
            not np.any(np.all(np.abs(chain[:step] - new_position) < tolerance, axis=1))
            or step == 0
        )
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
    """
    Chooses a valid next site for a chain, modifies the weight and chooses one to append.
    If no valid site exists the Alive for the chain is set to False from the next position onward.

    Parameters
        chain (ndarray): the positions of the chain to step
        weight (ndarray): the weights of the chain to step
        alive (ndarray): whether the chain is alive or not
        step (int): the current step to index, do_step will modify the array at `step + 1`
        next_sites_function (function): A function that returns a list of next valid position

    Returns
        list: list of valid coordinates for the next step
    """
    if not alive[step]:
        return
    allowed_sides = next_sites_function(chain, step)
    amount_of_allowed_sides = len(allowed_sides)
    if amount_of_allowed_sides > 0:
        next = random.choice(allowed_sides)
        chain[step + 1, :] = next
        weight[step + 1] = weight[step] * amount_of_allowed_sides
    else:
        alive[step + 1 :] = False


def init_polymer_storage(
    amount_of_chains: int, target_length: int, dimension: int
) -> tuple[NDArray[np.float64], NDArray[np.longdouble], NDArray[np.bool]]:
    """
    Initialises numpy arrays with the correct size and dimension for the generation of the polymers

    Parameters
        amount_of_chains (int): how many chains to start generating (may increase due to PERM)
        target_length (int): the maximum length L we want to generate
        dimension (int): how many coordinates there are to store for each location

    Returns
        ndarray: array to store the positions of the chains into
        ndarray: array to store the weights of the chains into
        ndarray: array to store the alive status of the chains into
    """
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
) -> int:
    """
    Does a step of pruning and enriching on the chains after they were generated.
    Note: modifies `step + 1`.

    Parameters
        chains (ndarray): array containing the coordinates of the chains at all generated lengths
        weights (ndarray): array containing the weights of the chains at all generated lengths
        alive (ndarray): array containing whether the chains are alive at certain lengths

    Returns
        int: the new amount of chains to grow and work with
    """
    w_low, w_high = perm_weights
    mean_weight = np.mean(
        weights[:amount_of_chains][alive[:amount_of_chains, step + 1], step + 1]
    )
    pruned = 0  # keep track of how many polymers got pruned this step
    new_amount_of_chains = amount_of_chains
    for chain in range(amount_of_chains):
        if not alive[chain, step + 1]:
            continue
        # Pruning
        if weights[chain, step + 1] < w_low * mean_weight:
            if random.random() < 0.5:
                pruned += 1
                # don't grow this chain anymore
                alive[chain, step + 1 :] = False
                # discard the weight at length L'
                weights[chain, step + 1] = 0
            else:
                # double the weight from L' onwards
                weights[chain, step + 1] *= 2
        # Enrichment
        elif weights[chain, step + 1] > w_high * mean_weight:
            weights[chain, step + 1] /= 2
            chains[new_amount_of_chains] = chains[chain]
            weights[new_amount_of_chains] = weights[chain]
            alive[new_amount_of_chains] = alive[chain]
            new_amount_of_chains += 1
    return new_amount_of_chains


def grow_polymers(
    amount_of_chains: int,
    target_length: int,
    dimension: int,
    next_sides_function: Callable[
        [NDArray[np.float64], int], list[NDArray[np.float64]]
    ],
    do_perm: bool,
    perm_weights: tuple[float, float],
    seed: int,
    threshold: int,
) -> tuple[int, int, NDArray[np.float64], NDArray[np.bool], NDArray[np.longdouble]]:
    """
    Grows a number of polymers upto a target length, if all polymers get stuck this function returns early.

    Parameters
        amount_of_chains (int): how many unique polymers to start with
        target_length (int): if this length is reached, stop
        dimension (int): how many coordinates are needed for a point to be specified
        next_sides_function (function): a function that gives the growing possibilities for a certain polymer
        do_perm (bool): whether to include the PERM step or not
        perm_weights (float, float): the factors to determine whether to prune or enrich
        seed (int): seed for the random number generator
        threshold (int): how many polymers need to be alive to continue

    Returns
        int: the maximum chain length that was created
        int: the amount of chains that were created in total (not only those of maximum length)
        ndarray: the positions of all chains at every step
        ndarray: the alive status of all chains at every step
        ndarray: the weights of all chains at every step
    """
    random.seed(seed)
    # 2 is enough with quite a margin for the default config values
    init_chains = amount_of_chains * (2 if do_perm else 1)
    chains, weights, alive = init_polymer_storage(init_chains, target_length, dimension)
    with logging_redirect_tqdm():
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
            if not alive[:amount_of_chains, step + 1].any():
                LOG.warning(f"All chains died by step {step + 1}, skipping other steps")
                break

            if do_perm:
                amount_of_chains = perm_step(
                    chains, weights, alive, step, amount_of_chains, perm_weights
                )
    alive_counts = np.sum(alive[:amount_of_chains], axis=0)
    try:
        max_step = np.where(alive_counts <= threshold)[0][0]
    except IndexError:
        max_step = np.max(np.sum(alive, axis=1))
    return (
        max_step,
        amount_of_chains,
        chains[:amount_of_chains, :max_step, :],
        alive[:amount_of_chains, :max_step],
        weights[:amount_of_chains, :max_step],
    )
