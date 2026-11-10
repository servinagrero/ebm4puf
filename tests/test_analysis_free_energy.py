import importlib.util
from pathlib import Path
from statistics import NormalDist

import numpy as np

from ebm4puf.artifact import error_matrix
from ebm4puf.authentication import (
    calibrated_anomaly_scores,
    calibrated_directional_scores,
    center_evidence,
    gaussian_sequence_auc,
    required_readouts_for_auc,
)
from ebm4puf.diagnostics import relaxation_metrics
from ebm4puf.probability import poisson_binomial_pmf
from ebm4puf.sampling import (
    bootstrap_sequence_sums,
    bootstrap_sequence_vectors,
    fixed_bit_permutation,
    split_error_matrix,
)

SCRIPT_PATH = Path(__file__).parents[1] / "scripts" / "analysis_free_energy.py"
SPEC = importlib.util.spec_from_file_location("analysis_free_energy", SCRIPT_PATH)
assert SPEC is not None and SPEC.loader is not None
analysis = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(analysis)

parse_training_config = analysis.parse_training_config


def test_error_matrix_xors_each_readout_with_reference():
    responses = np.array([[1, 1, 0], [0, 1, 1]], dtype=np.uint8)
    reference = np.array([1, 0, 1], dtype=np.uint8)

    errors = error_matrix(responses, reference)

    np.testing.assert_array_equal(errors, [[0, 1, 1], [1, 1, 0]])


def test_split_is_deterministic_and_exhaustive():
    errors = np.arange(60).reshape(10, 6)

    first = split_error_matrix(
        errors, train_fraction=0.6, calibration_fraction=0.2, seed=4
    )
    second = split_error_matrix(
        errors, train_fraction=0.6, calibration_fraction=0.2, seed=4
    )

    assert [len(part) for part in first] == [6, 2, 2]
    for left, right in zip(first, second):
        np.testing.assert_array_equal(left, right)
    np.testing.assert_array_equal(
        np.sort(np.concatenate(first), axis=0), np.sort(errors, axis=0)
    )


def test_fixed_permutation_preserves_error_count_per_readout():
    errors = np.array([[1, 0, 0, 1], [0, 1, 0, 0]], dtype=np.uint8)

    permuted = fixed_bit_permutation(errors, np.array([2, 0, 3, 1]))

    np.testing.assert_array_equal(permuted.sum(axis=1), errors.sum(axis=1))


def test_sequence_score_is_additive():
    sums = bootstrap_sequence_sums(
        np.array([3.0]),
        group_size=4,
        repeats=5,
        rng=np.random.default_rng(0),
    )

    np.testing.assert_array_equal(sums, np.full(5, 12.0))


def test_sequence_vector_bootstrap_preserves_paired_channels():
    sums = bootstrap_sequence_vectors(
        np.array([[1.0, 2.0]]),
        group_size=4,
        repeats=5,
        rng=np.random.default_rng(0),
    )

    np.testing.assert_array_equal(sums, np.tile([4.0, 8.0], (5, 1)))


def test_calibrated_anomaly_scores_compare_one_and_two_channels():
    scores = calibrated_anomaly_scores(
        np.array([[0.0, 0.0], [3.0, 0.0], [0.0, 3.0]]),
        per_readout_mean=np.zeros(2),
        per_readout_covariance=np.eye(2),
        group_size=1,
    )

    np.testing.assert_allclose(scores["marginal_anomaly"][0], 0.0)
    np.testing.assert_allclose(scores["residual_anomaly"][0], 0.0)
    np.testing.assert_allclose(scores["joint_mahalanobis"][0], 0.0)
    assert scores["marginal_anomaly"][1] < -8.9
    assert scores["residual_anomaly"][1] == 0.0
    assert scores["joint_mahalanobis"][1] < -8.9
    assert scores["marginal_anomaly"][2] == 0.0
    assert scores["residual_anomaly"][2] < -8.9
    assert scores["joint_mahalanobis"][2] < -8.9


def test_directional_scores_include_calibration_mean_uncertainty():
    scores = calibrated_directional_scores(
        np.array([[0.0, 0.0], [4.0, -4.0]]),
        per_readout_mean=np.zeros(2),
        per_readout_covariance=np.eye(2),
        group_size=4,
        n_calibration=4,
    )

    expected = 4 / np.sqrt(8)
    np.testing.assert_allclose(scores["marginal_directional"], [0.0, expected])
    np.testing.assert_allclose(scores["residual_directional"], [0.0, -expected])
    np.testing.assert_allclose(scores["joint_bonferroni"], [0.0, -expected])


def test_poisson_binomial_pmf_is_exact_and_normalized():
    pmf = poisson_binomial_pmf(np.array([0.5, 0.5]))

    np.testing.assert_allclose(pmf, [0.25, 0.5, 0.25])
    np.testing.assert_allclose(pmf.sum(), 1.0)


def test_center_evidence_uses_matched_marginal_median():
    rows = [
        {
            "scenario": scenario,
            "group_size": 2,
            **{f"evidence_{q}": value for q in ("q05", "q25", "q50", "q75", "q95")},
        }
        for scenario, value in (("marginal", 4.0), ("genuine", 7.0))
    ]

    center_evidence(rows)

    assert rows[0]["centered_evidence_q50"] == 0.0
    assert rows[1]["centered_evidence_q50"] == 3.0


def test_relaxation_compares_checkpoint_with_immediately_previous_state():
    class FakeModel:
        def gibbs_chain(self, values, *, checkpoints, seed):
            assert checkpoints == [1, 4, 5]
            states = {
                1: np.array([[1, 0]], dtype=np.uint8),
                4: np.array([[1, 1]], dtype=np.uint8),
                5: np.array([[0, 1]], dtype=np.uint8),
            }
            return {
                step: (np.zeros_like(states[step], dtype=float), states[step])
                for step in checkpoints
            }

    rows = relaxation_metrics(
        model=FakeModel(),
        scenarios={"genuine": np.array([[0, 0]], dtype=np.uint8)},
        checkpoints=[1, 5],
        max_samples=10,
        seed=0,
        source_gibbs_steps=0,
    )

    assert [row["relaxation_steps"] for row in rows] == [1, 5]
    assert [row["mean_transition_hamming_fraction"] for row in rows] == [0.5, 0.5]
    assert [row["unchanged_transition_fraction"] for row in rows] == [0.0, 0.0]


def test_training_config_parser_accepts_cd_and_pcd():
    assert parse_training_config("CD:5") == ("cd", 5)
    assert parse_training_config("pcd:1") == ("pcd", 1)


def test_gaussian_auc_scales_with_square_root_of_group_size():
    genuine = np.array([1.0, 3.0])
    alternative = np.array([0.0, 2.0])

    np.testing.assert_allclose(
        gaussian_sequence_auc(genuine, alternative, 1),
        NormalDist().cdf(0.5),
    )
    np.testing.assert_allclose(
        gaussian_sequence_auc(genuine, alternative, 4),
        NormalDist().cdf(1.0),
    )


def test_required_readouts_is_directional():
    genuine = np.array([1.0, 3.0])
    alternative = np.array([0.0, 2.0])

    assert (
        required_readouts_for_auc(
            genuine,
            alternative,
            NormalDist().cdf(0.5),
        )
        == 1
    )
    assert required_readouts_for_auc(alternative, genuine, 0.8) is None
