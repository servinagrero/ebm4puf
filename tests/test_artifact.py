from pathlib import Path

import h5py
import numpy as np
import pytest

from ebm4puf.artifact import error_matrix, read_family, selected_family_names


def test_error_matrix_rejects_nonbinary_responses():
    with pytest.raises(ValueError, match="responses must be binary"):
        error_matrix(np.array([[0, 2]]), np.array([0, 1]))


def test_family_selection_filters_non_room_temperature(tmp_path: Path):
    artifact = tmp_path / "artifact.h5"
    with h5py.File(artifact, "w") as h5:
        h5.create_group("room")
        hot = h5.create_group("hot")
        hot.attrs["environment"] = "hot"

    assert selected_family_names(
        artifact, None, include_non_room_temperature=False
    ) == ["room"]
    assert selected_family_names(
        artifact, ["hot"], include_non_room_temperature=True
    ) == ["hot"]


def test_family_selection_rejects_missing_family(tmp_path: Path):
    artifact = tmp_path / "artifact.h5"
    with h5py.File(artifact, "w"):
        pass

    with pytest.raises(ValueError, match="missing requested families"):
        selected_family_names(
            artifact, ["missing"], include_non_room_temperature=False
        )


def test_read_family_validates_shapes(tmp_path: Path):
    artifact = tmp_path / "artifact.h5"
    with h5py.File(artifact, "w") as h5:
        family = h5.create_group("test")
        family.create_dataset("responses", data=np.zeros((2, 3, 4), dtype=np.uint8))
        family.create_dataset("reference", data=np.zeros((2, 5), dtype=np.uint8))

    with (
        h5py.File(artifact, "r") as h5,
        pytest.raises(ValueError, match="bit counts do not match"),
    ):
        read_family(h5, "test")


def test_read_family_returns_validated_arrays(tmp_path: Path):
    artifact = tmp_path / "artifact.h5"
    responses = np.zeros((2, 3, 4), dtype=np.uint8)
    reference = np.zeros((2, 4), dtype=np.uint8)
    with h5py.File(artifact, "w") as h5:
        family = h5.create_group("test")
        family.create_dataset("responses", data=responses)
        family.create_dataset("reference", data=reference)

    with h5py.File(artifact, "r") as h5:
        actual_responses, actual_reference = read_family(h5, "test")

    np.testing.assert_array_equal(actual_responses, responses)
    np.testing.assert_array_equal(actual_reference, reference)


def test_read_family_checks_values_before_integer_conversion(tmp_path: Path):
    artifact = tmp_path / "artifact.h5"
    with h5py.File(artifact, "w") as h5:
        family = h5.create_group("test")
        family.create_dataset("responses", data=np.full((1, 1, 1), 0.5))
        family.create_dataset("reference", data=np.zeros((1, 1), dtype=np.uint8))

    with (
        h5py.File(artifact, "r") as h5,
        pytest.raises(ValueError, match="responses must be binary"),
    ):
        read_family(h5, "test")
