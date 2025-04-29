import numpy as np
from polymer_code.utilities import parse_config, resolve_function
from polymer_code.simulate import get_allowed_sides_2d

_, _, _, _, _, dimension, next_sides_function, _ = parse_config("config.json")

def test_allowed_sides_start():
    chain = np.zeros((20, 2))
    expected = [
        np.array([1, 0]),
        np.array([-1, 0]),
        np.array([0, 1]),
        np.array([0, -1]),
    ]
    actual = next_sides_function(chain, 0)
    for side, target in zip(actual, expected):
        np.testing.assert_array_equal(side, target)

def test_allowed_sides_later():
    chain = np.zeros((20, 2))
    chain[0, :] = [0, 0]
    chain[1, :] = [1, 0]
    expected = [
        np.array([2, 0]),
        np.array([1, 1]),
        np.array([1, -1]),
    ]
    actual = next_sides_function(chain, 1)
    for side, target in zip(actual, expected):
        np.testing.assert_array_equal(side, target)

def test_resolve_function_correct():
    func = resolve_function("simulate.get_allowed_sides_2d")
    assert callable(func)
    assert func == get_allowed_sides_2d

