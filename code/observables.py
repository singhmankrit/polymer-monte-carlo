import numpy as np
from tqdm import trange


def find_observables(amount_of_chains, max_step, chains, alive):
    end_to_ends = np.zeros((amount_of_chains, max_step))
    gyrations = np.zeros((amount_of_chains, max_step))
    for chain in trange(amount_of_chains):
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
        tempgyrations = np.array(
            [
                np.sum(clens[: length + 1, length]) / (length + 1)
                for length in range(0, max_step)
            ]
        )
        gyrations[chain, alive[chain, :max_step]] = tempgyrations[
            alive[chain, :max_step]
        ]
    return end_to_ends, gyrations

