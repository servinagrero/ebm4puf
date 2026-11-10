"""Shared access and validation for PUF dataset artifacts."""

import logging
from collections.abc import Sequence
from pathlib import Path

import h5py
import numpy as np

LOGGER = logging.getLogger(__name__)
NON_ROOM_ENVIRONMENTS = {"freezer", "hot"}


def decode_attribute(value: object) -> str | None:
    """Return an HDF5 attribute as text while preserving missing values."""
    if value is None:
        return None
    if isinstance(value, bytes):
        return value.decode()
    return str(value)


def error_matrix(responses: np.ndarray, reference: np.ndarray) -> np.ndarray:
    """Return binary bit-flip vectors for one device."""
    responses = np.asarray(responses)
    reference = np.asarray(reference)
    if responses.ndim != 2 or reference.ndim != 1:
        raise ValueError("responses must be 2D and reference must be 1D")
    if responses.shape[1] != reference.size:
        raise ValueError("response and reference bit counts do not match")
    if not np.all((responses == 0) | (responses == 1)):
        raise ValueError("responses must be binary")
    if not np.all((reference == 0) | (reference == 1)):
        raise ValueError("reference must be binary")
    return np.logical_xor(responses, reference).astype(np.uint8)


def selected_family_names(
    artifact: Path,
    requested: Sequence[str] | None,
    *,
    include_non_room_temperature: bool,
) -> list[str]:
    """Return requested artifact families after environment filtering."""
    with h5py.File(artifact, "r") as h5:
        available = sorted(h5.keys())
        if requested:
            missing = sorted(set(requested) - set(available))
            if missing:
                raise ValueError(f"artifact is missing requested families: {missing}")
            available = list(requested)

        selected = []
        for family in available:
            environment = decode_attribute(h5[family].attrs.get("environment"))
            if (
                not include_non_room_temperature
                and environment in NON_ROOM_ENVIRONMENTS
            ):
                LOGGER.info("Skipping non-room-temperature family %s", family)
                continue
            selected.append(family)
        return selected


def read_family(
    artifact: h5py.File,
    family: str,
) -> tuple[np.ndarray, np.ndarray]:
    """Read and validate a family's response and reference arrays."""
    group = artifact[family]
    if "responses" not in group or "reference" not in group:
        raise ValueError(f"{family} must contain responses and reference datasets")

    responses = np.asarray(group["responses"])
    reference = np.asarray(group["reference"])
    if responses.ndim != 3:
        raise ValueError(f"{family} responses must have shape (devices, samples, bits)")
    if reference.ndim != 2:
        raise ValueError(f"{family} reference must have shape (devices, bits)")
    if responses.shape[0] != reference.shape[0]:
        raise ValueError(f"{family} response and reference device counts do not match")
    if responses.shape[2] != reference.shape[1]:
        raise ValueError(f"{family} response and reference bit counts do not match")
    if not np.all((responses == 0) | (responses == 1)):
        raise ValueError(f"{family} responses must be binary")
    if not np.all((reference == 0) | (reference == 1)):
        raise ValueError(f"{family} reference must be binary")
    return responses.astype(np.uint8, copy=False), reference.astype(np.uint8, copy=False)
