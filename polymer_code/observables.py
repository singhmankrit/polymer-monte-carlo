import numpy as np
from numpy.typing import NDArray
from tqdm import trange


def find_end_to_end(
    chain: NDArray[np.float64], alive: NDArray[np.float64]
) -> NDArray[np.float64]:
    """
    Calculates the end-to-end distance for a chain at every length

    Parameters
        chain (ndarray): the 2D array containing the positions for each chain at every length
        alive (ndarray): the 1D array containing the status of the chain at every length

    Returns
        ndarray(max_step): the end-to-end distance for the chain at each length (where it is alive)
    """
    start = chain[0, :]
    end = chain[:, :]
    diff = end - start
    return np.vecdot(diff, diff)[alive]


def find_gyration(
    chain: NDArray[np.float64], alive: NDArray[np.float64]
) -> NDArray[np.float64]:
    """
    Calculates the gyration for a chain at every length

    Parameters
        chain (ndarray): the 2D array containing the positions for each chain at every length
        alive (ndarray): the 1D array containing the status of the chain at every length

    Returns
        ndarray(max_step): the gyration for the chain at each length (where it is alive)
    """
    # there is no need to calculate further than the first dead spot
    max_steps = np.where(alive == False)[0]
    if max_steps.size > 0:
        max_step = max_steps[0]
    else:
        max_step = alive.size
    # the middle point (coordinates axis 1) up to length (axis 0)
    center = np.array(
        [
            np.sum(chain[:length, :], axis=0, keepdims=True) / length
            for length in range(1, max_step + 1)
        ]
    )
    # differences from center (coordinates axis 2, length axis 1, particle axis 0)
    cdiffs = np.array([chain[length, :] - center for length in range(0, max_step)])[
        :, :, 0, :
    ]

    # distance per length (axis 1), per particle (axis 0)
    clens = np.sum(cdiffs * cdiffs, axis=-1)
    return np.array(
        [
            np.sum(clens[: length + 1, length]) / (length + 1)
            for length in range(0, max_step)
        ]
    )


def find_observables(
    amount_of_chains: int,
    max_step: int,
    chains: NDArray[np.float64],
    alive: NDArray[np.bool],
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """
    Calculates the end-to-end distances and gyrations for all chains at every length they exist.

    Parameters
        amount_of_chains (int): how many chains are in the input
        max_step (int): the size of the longest chain in the input
        chains (ndarray): the 3D array containing the positions of each point of every chain
        alive (ndarray): the 2D array containing whether each chain is "alive" at a certain length

    Returns
        ndarray(amount_of_chains, max_step): the end-to-end distance for each chain at each length (where it is alive)
        ndarray(amount_of_chains, max_step): the gyration for each chain at each length (where it is alive)
    """
    end_to_ends = np.zeros((amount_of_chains, max_step))
    gyrations = np.zeros((amount_of_chains, max_step))
    for chain in trange(amount_of_chains, desc="Observables"):
        end_to_ends[chain, alive[chain]] = find_end_to_end(chains[chain], alive[chain])
        gyrations[chain, alive[chain]] = find_gyration(chains[chain], alive[chain])
    return end_to_ends, gyrations
