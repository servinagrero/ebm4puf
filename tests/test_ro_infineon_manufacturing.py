import importlib.util
from pathlib import Path

import numpy as np
import polars as pl
import pytest

MODULE_PATH = (
    Path(__file__).parents[1] / "scripts" / "analysis_ro_infineon_manufacturing.py"
)
SPEC = importlib.util.spec_from_file_location(
    "analysis_ro_infineon_manufacturing", MODULE_PATH
)
assert SPEC is not None and SPEC.loader is not None
analysis = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(analysis)


def synthetic_inputs():
    frequencies = 1 + np.arange(2 * 6 * 255 * 2, dtype=float).reshape(2, 6, 255, 2)
    mapping = pl.DataFrame(
        {
            "artifact_bit": np.arange(256),
            "group_a": np.r_[np.zeros(255, dtype=int), 2],
            "group_b": np.r_[np.ones(255, dtype=int), 3],
            "design_id": np.r_[np.arange(255), 0],
        }
    )
    group_a = mapping["group_a"].to_numpy()
    group_b = mapping["group_b"].to_numpy()
    design = mapping["design_id"].to_numpy()
    differences = (
        frequencies[:, group_a, design, :] - frequencies[:, group_b, design, :]
    )
    responses = (differences > 0).astype(np.uint8).transpose(0, 2, 1)
    metadata = pl.DataFrame(
        {
            "device_id": [0, 1],
            "x": [1, 1],
            "y": [2, 2],
            "wafer_id": [3, 3],
            "lot_id": [4, 5],
        }
    )
    conditions = pl.DataFrame({"condition_id": [0, 1]})
    return metadata, conditions, frequencies, mapping, responses, differences


def test_validate_inputs_reconstructs_selected_responses():
    metadata, conditions, frequencies, mapping, responses, expected = synthetic_inputs()

    actual = analysis.validate_inputs(
        metadata, conditions, frequencies, mapping, responses
    )

    np.testing.assert_array_equal(actual, expected)


def test_validate_inputs_requires_full_manufacturing_identity_to_be_unique():
    metadata, conditions, frequencies, mapping, responses, _ = synthetic_inputs()
    metadata = metadata.with_columns(pl.lit(4).alias("lot_id"))

    with pytest.raises(ValueError, match="uniquely identify"):
        analysis.validate_inputs(metadata, conditions, frequencies, mapping, responses)


def test_grouped_zscore_is_centered_within_groups():
    values = np.array([1.0, 3.0, 10.0, 14.0])
    groups = np.array(["a", "a", "b", "b"])

    actual = analysis.grouped_zscore(values, groups)

    np.testing.assert_allclose(actual[[0, 1]].mean(), 0)
    np.testing.assert_allclose(actual[[2, 3]].mean(), 0)
    np.testing.assert_allclose(actual[[0, 1]].std(ddof=1), 1)
    np.testing.assert_allclose(actual[[2, 3]].std(ddof=1), 1)


def test_stratified_permutation_is_reproducible():
    values = np.arange(12, dtype=float)
    outcomes = values + np.tile([0.0, 1.0], 6)
    strata = np.repeat(["a", "b", "c"], 4)

    first = analysis.stratified_permutation_pvalue(
        values,
        outcomes,
        strata,
        repeats=100,
        rng=np.random.default_rng(7),
    )
    second = analysis.stratified_permutation_pvalue(
        values,
        outcomes,
        strata,
        repeats=100,
        rng=np.random.default_rng(7),
    )

    assert first == second


def test_selected_pair_quantities_separate_absolute_and_relative_scale():
    _, _, frequencies, mapping, _, _ = synthetic_inputs()

    mean, difference, relative, log_ratio = analysis.selected_pair_quantities(
        frequencies, mapping
    )
    scaled = analysis.selected_pair_quantities(10 * frequencies, mapping)

    np.testing.assert_allclose(scaled[0], 10 * mean)
    np.testing.assert_allclose(scaled[1], 10 * difference)
    np.testing.assert_allclose(scaled[2], relative)
    np.testing.assert_allclose(scaled[3], log_ratio)
    np.testing.assert_allclose(relative, difference / mean)


def test_selected_difference_frame_preserves_bit_and_condition_alignment():
    pair_mean = np.array([[[10.0, 20.0], [30.0, 40.0]]])
    differences = np.array([[[1.0, -2.0], [3.0, -4.0]]])
    relative = differences / pair_mean
    log_ratio = np.array([[[0.1, -0.2], [0.3, -0.4]]])
    errors = np.array([[[0, 1], [1, 0]]], dtype=np.uint8)

    frame = analysis.selected_difference_frame(
        pair_mean, differences, relative, log_ratio, errors
    )

    row = frame.filter(
        (pl.col("artifact_bit") == 1) & (pl.col("condition_id") == 0)
    ).row(0, named=True)
    assert row["pair_mean_frequency"] == 30
    assert row["frequency_difference"] == 3
    assert row["relative_frequency_difference"] == 0.1
    assert row["bitflip"] == 1


def test_surface_uses_equal_device_weighting_within_bins():
    differences = np.array([[[0.0, 0.1], [0.2, 0.3]], [[0.05, 1.0], [2.0, 3.0]]])
    pair_mean = differences + 1
    errors = np.array([[[1, 1], [1, 1]], [[0, 0], [0, 0]]], dtype=np.uint8)

    rows = analysis.device_balanced_surface(
        differences,
        pair_mean,
        errors,
        measure="test",
        bins=2,
    )
    low_bin = next(
        row for row in rows if row["difference_bin"] == 0 and row["frequency_bin"] == 0
    )

    assert low_bin["n_observations"] == 4
    assert low_bin["n_devices"] == 2
    assert low_bin["device_balanced_bitflip_probability"] == 0.5


def test_weighted_residual_summary_preserves_sign_and_support():
    conditionals = pl.DataFrame(
        {
            "device": [0, 0],
            "bit": [0, 0],
            "k": [1, 2],
            "support": [3, 1],
            "residual": [0.2, -0.4],
        }
    )

    actual = analysis.weighted_bit_residuals(conditionals).row(0, named=True)

    assert actual["total_readout_support"] == 4
    np.testing.assert_allclose(actual["weighted_signed_residual"], 0.05)
    np.testing.assert_allclose(actual["weighted_absolute_residual"], 0.25)


def test_row_spearman_and_rank_groups_follow_within_row_order():
    values = np.array([[1.0, 2.0, 3.0, 4.0], [4.0, 3.0, 2.0, 1.0]])
    outcomes = np.array([[2.0, 4.0, 6.0, 8.0], [1.0, 2.0, 3.0, 4.0]])

    correlations = analysis.row_spearman(values, outcomes)

    np.testing.assert_allclose(correlations, [1, -1])
    np.testing.assert_array_equal(analysis.rank_groups(values[0], 2), [0, 0, 1, 1])
