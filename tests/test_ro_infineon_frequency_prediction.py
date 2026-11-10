import importlib.util
import sys
from pathlib import Path

import numpy as np

SCRIPTS = Path(__file__).parents[1] / "scripts"
sys.path.insert(0, str(SCRIPTS))
MODULE_PATH = SCRIPTS / "analysis_ro_infineon_frequency_prediction.py"
SPEC = importlib.util.spec_from_file_location(
    "analysis_ro_infineon_frequency_prediction", MODULE_PATH
)
assert SPEC is not None and SPEC.loader is not None
analysis = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(analysis)


def test_leave_one_condition_out_uses_only_other_conditions_for_reference():
    differences = np.array([[[1.0, 2.0, 3.0, -4.0]]])
    pair_mean = np.full_like(differences, 10.0)
    relative = differences / pair_mean
    responses = (differences > 0).astype(np.uint8).transpose(0, 2, 1)

    dataset = analysis.leave_one_condition_out_dataset(
        pair_mean,
        differences,
        relative,
        responses,
        np.array([0]),
    )

    np.testing.assert_array_equal(dataset["bitflip"], [0, 0, 0, 1])
    assert dataset["log_absolute_frequency_difference"][3] == np.log1p(2)
    assert dataset["log_absolute_frequency_difference"][0] == np.log1p(2)


def test_relative_enrollment_feature_is_invariant_to_frequency_scale():
    differences = np.array([[[1.0, 2.0, 3.0, -4.0]]])
    pair_mean = np.full_like(differences, 10.0)
    relative = differences / pair_mean
    responses = (differences > 0).astype(np.uint8).transpose(0, 2, 1)
    original = analysis.leave_one_condition_out_dataset(
        pair_mean, differences, relative, responses, np.array([0])
    )
    scaled = analysis.leave_one_condition_out_dataset(
        10 * pair_mean,
        10 * differences,
        relative,
        responses,
        np.array([0]),
    )

    np.testing.assert_allclose(
        original["log_absolute_relative_frequency_difference"],
        scaled["log_absolute_relative_frequency_difference"],
    )
    assert not np.allclose(
        original["log_absolute_frequency_difference"],
        scaled["log_absolute_frequency_difference"],
    )


def test_batched_poisson_binomial_matches_two_bernoulli_variables():
    actual = analysis.batched_poisson_binomial(np.array([[0.2, 0.3]]))

    np.testing.assert_allclose(actual, [[0.56, 0.38, 0.06]])
    np.testing.assert_allclose(actual.sum(axis=1), 1)
