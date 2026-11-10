from itertools import product

import numpy as np

from ebm4puf.error_structure import (
    analyze_device,
    conditional_independent_probabilities,
    leave_one_device_out_prior,
    population_error_statistics,
    reference_polarity_statistics,
    select_shrinkage_strength,
    shrink_marginals,
)


def brute_force_conditional(probabilities: np.ndarray, k: int) -> np.ndarray:
    states = np.asarray(list(product((0, 1), repeat=len(probabilities))))
    weights = np.prod(np.where(states == 1, probabilities, 1 - probabilities), axis=1)
    selected = states.sum(axis=1) == k
    return (
        np.sum(states[selected] * weights[selected, None], axis=0)
        / weights[selected].sum()
    )


def test_conditional_independent_probabilities_match_enumeration():
    probabilities = np.array([0.1, 0.3, 0.6])
    actual = conditional_independent_probabilities(probabilities, np.arange(4))

    for k in range(4):
        np.testing.assert_allclose(
            actual[k], brute_force_conditional(probabilities, k), atol=1e-12
        )
        np.testing.assert_allclose(actual[k].sum(), k, atol=1e-12)


def test_conditional_independent_probabilities_are_stable_in_rare_tail():
    probabilities = np.linspace(0.001, 0.15, 256)
    k = 82
    actual = conditional_independent_probabilities(probabilities, [k])[k]

    assert np.all(np.isfinite(actual))
    assert np.all((actual >= 0) & (actual <= 1))
    np.testing.assert_allclose(actual.sum(), k, atol=1e-10)

    extreme_k = 223
    extreme = conditional_independent_probabilities(
        np.full(256, 0.001), [extreme_k]
    )[extreme_k]
    np.testing.assert_allclose(extreme, extreme_k / 256, atol=1e-10)


def test_family_prior_excludes_target_and_remains_rare():
    target = np.ones((4, 3), dtype=np.uint8)
    other = np.zeros((6, 3), dtype=np.uint8)
    family_counts = target.sum(axis=0) + other.sum(axis=0)

    prior = leave_one_device_out_prior(
        family_counts,
        family_readouts=10,
        excluded_errors=target,
        prior_strength=1,
    )

    assert np.all(prior > 0)
    assert np.all(prior < 0.1)


def test_shrinkage_selection_uses_configured_grid_and_rare_prior():
    enrollment = np.zeros((8, 4), dtype=np.uint8)
    enrollment[0, 0] = 1
    prior = np.full(4, 0.01)
    strengths = np.array([0.1, 1, 10])

    selected = select_shrinkage_strength(enrollment, prior, strengths)
    marginals = shrink_marginals(enrollment, prior, selected)

    assert selected in strengths
    assert np.all((marginals > 0) & (marginals < 1))
    assert marginals.sum() < 0.5


def test_population_statistics_use_equal_device_weighting_and_fixed_seed():
    empirical = np.array([[1, 0, 0], [0, 0, 1]], dtype=float)
    independent = np.array([[0.5, 0.5, 0], [0, 0.5, 0.5]], dtype=float)
    kwargs = {
        "family": "test",
        "target_failure_probabilities": np.array([0.5]),
        "target_device_coverages": np.array([0.5]),
        "bootstrap_repeats": 10,
        "seed": 4,
    }

    first = population_error_statistics(
        empirical, independent, np.array([5, 5]), **kwargs
    )
    second = population_error_statistics(
        empirical, independent, np.array([5, 5]), **kwargs
    )

    population = first[1]
    coverage = first[2]
    assert population[0]["empirical_random_device_tail"] == 0.5
    assert population[1]["empirical_random_device_tail"] == 0.5
    assert coverage[0]["n_readouts"] == 5
    assert coverage[0]["maximum_observed_failures"] == 2
    assert coverage[0]["effective_empirical_probability"] == 0.4
    assert first == second


def test_device_analysis_cross_fits_normalized_distributions():
    errors = np.array(
        [
            [0, 0, 0],
            [1, 0, 0],
            [0, 1, 0],
            [1, 1, 0],
            [0, 0, 1],
            [1, 0, 1],
        ],
        dtype=np.uint8,
    )
    counts, conditionals, marginals, metrics = analyze_device(
        errors,
        family="test",
        device=0,
        family_prior=np.full(3, 0.1),
        shrinkage_strengths=np.array([0.1, 1, 10]),
        n_folds=3,
        seed=7,
    )

    np.testing.assert_allclose(sum(row["empirical_probability"] for row in counts), 1)
    np.testing.assert_allclose(sum(row["independent_probability"] for row in counts), 1)
    assert sum(row["empirical_count"] for row in counts) == len(errors)
    assert len(marginals) == 3 * 3
    assert metrics["n_folds"] == 3
    for k in np.unique(errors.sum(axis=1)):
        rows = [row for row in conditionals if row["k"] == k]
        np.testing.assert_allclose(
            sum(row["empirical_probability"] for row in rows), k, atol=1e-12
        )
        np.testing.assert_allclose(
            sum(row["independent_probability"] for row in rows), k, atol=1e-12
        )
        np.testing.assert_allclose(sum(row["residual"] for row in rows), 0, atol=1e-12)


def test_reference_polarity_statistics_distinguish_transition_directions():
    errors = np.array(
        [
            [1, 0, 0, 0],
            [0, 0, 1, 0],
            [1, 1, 0, 0],
            [1, 0, 1, 0],
        ],
        dtype=np.uint8,
    )
    reference = np.array([0, 0, 1, 1], dtype=np.uint8)
    conditional_rows = []
    for k, empirical in (
        (1, np.array([0.5, 0.0, 0.5, 0.0])),
        (2, np.array([1.0, 0.5, 0.5, 0.0])),
    ):
        independent = np.full(4, k / 4)
        for bit in range(4):
            conditional_rows.append(
                {
                    "k": k,
                    "bit": bit,
                    "support": 2,
                    "empirical_probability": empirical[bit],
                    "independent_probability": independent[bit],
                }
            )

    rows, summary, weighted = reference_polarity_statistics(
        errors,
        reference,
        conditional_rows,
        family="test",
        device=0,
    )

    assert len(rows) == 4
    np.testing.assert_allclose(summary["raw_transition_contrast"], -0.25)
    np.testing.assert_allclose(summary["independent_transition_contrast"], 0)
    np.testing.assert_allclose(summary["residual_transition_contrast"], -0.25)
    np.testing.assert_allclose(weighted, [0.375, -0.125, 0.125, -0.375])
