import numpy as np
from polymer_code.utilities import parse_config, resolve_function
from polymer_code.simulate import get_allowed_sides_2d, init_polymer_storage, do_step, perm_step

_, _, _, _, _, _, next_sides_function, _ = parse_config("config.json")

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

def test_resolve_function():
    func = resolve_function("simulate.get_allowed_sides_2d")
    assert callable(func)
    assert func == get_allowed_sides_2d

def test_perm_step_enrichment_and_pruning():
    step = 2
    num_chains = 5
    target_length = 5
    dimension = 2
    chains = np.zeros((num_chains, target_length, dimension))
    weights = np.ones((num_chains, target_length), dtype=np.longdouble)
    alive = np.ones((num_chains, target_length), dtype=bool)

    # Setup weights to trigger:
    # - pruning for chains 0 and 1 (very low weights)
    # - no change for chain 2
    # - enrichment for chain 3 (very high weight)
    weights[0, step + 1] = 0.1
    weights[1, step + 1] = 0.2
    weights[2, step + 1] = 1.0
    weights[3, step + 1] = 6.0
    weights[4, step + 1] = 1.0

    chains[3, step + 1] = [5.0, 5.0]  # Unique position for enrichment check

    # Copy to compare later
    original_chain3 = chains[3].copy()
    original_weight3 = weights[3].copy()

    chains_new, weights_new, alive_new = perm_step(
        chains, weights, alive, step, num_chains, (0.5, 1.5)
    )

    # Check that some pruning occurred
    assert np.sum(weights_new[:num_chains, step + 1] == 0.0) >= 1, "Pruning did not occur properly"

    # Check that at least one chain was enriched (total chain count increased)
    assert chains_new.shape[0] > num_chains, "Chains not duplicated properly during enrichment"

    # Find duplicates of chain 3 by comparing positions at step + 1
    chain3_position = original_chain3[step + 1]
    duplicate_indices = [
        i for i in range(chains_new.shape[0])
        if np.allclose(chains_new[i, step + 1], chain3_position)
    ]
    assert len(duplicate_indices) >= 2, "Enrichment did not create a duplicate of chain 3"

    # Verify weights of duplicates are halved correctly
    for i in duplicate_indices:
        assert np.isclose(weights_new[i, step + 1], original_weight3[step + 1] / 2), \
            "Duplicate of enriched chain does not have correct weight"

    # Check that once a chain is dead, it stays dead
    for i in range(alive_new.shape[0]):
        false_indices = np.where(~alive_new[i])[0]
        if len(false_indices) > 0:
            first_false = false_indices[0]
            assert np.all(~alive_new[i, first_false:]), f"Chain {i} becomes alive after dying"
