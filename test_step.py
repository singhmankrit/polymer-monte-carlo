from .main import get_allowed_sides
import numpy as np


def test_allowed_sides_start():
    chain = np.zeros((20, 2))
    for side, target in zip(
        get_allowed_sides(chain, 0),
        [
            np.array([1, 0]),
            np.array([-1, 0]),
            np.array([0, 1]),
            np.array([0, -1]),
        ],
    ):
        assert (side == target).all()


def test_allowed_sides_later():
    chain = np.zeros((20, 2))
    chain[0, :] = [0, 0]
    chain[1, :] = [1, 0]
    for side, target in zip(
        get_allowed_sides(chain, 1),
        [
            np.array([2, 0]),
            np.array([1, 1]),
            np.array([1, -1]),
        ],
    ):
        assert (side == target).all()
