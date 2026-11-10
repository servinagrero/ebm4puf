"""Sampling and resampling utilities for PUF studies."""

import numpy as np


def split_error_matrix(
    errors: np.ndarray,
    *,
    train_fraction: float,
    calibration_fraction: float,
    seed: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Deterministically split a device's readouts into three nonempty sets."""
    errors = np.asarray(errors)
    if errors.ndim != 2 or len(errors) < 3:
        raise ValueError("at least three error vectors are required")
    if train_fraction <= 0 or calibration_fraction <= 0:
        raise ValueError("split fractions must be positive")
    if train_fraction + calibration_fraction >= 1:
        raise ValueError("train and calibration fractions must sum to less than 1")

    rng = np.random.default_rng(seed)
    indices = rng.permutation(len(errors))
    n_train = max(1, int(np.floor(len(errors) * train_fraction)))
    n_calibration = max(1, int(np.floor(len(errors) * calibration_fraction)))
    if n_train + n_calibration >= len(errors):
        n_train = len(errors) - n_calibration - 1
    train_end = n_train
    calibration_end = n_train + n_calibration
    return (
        errors[indices[:train_end]],
        errors[indices[train_end:calibration_end]],
        errors[indices[calibration_end:]],
    )


def fixed_bit_permutation(errors: np.ndarray, permutation: np.ndarray) -> np.ndarray:
    """Permute columns identically for every readout, preserving error counts."""
    errors = np.asarray(errors)
    permutation = np.asarray(permutation)
    if errors.ndim != 2 or permutation.shape != (errors.shape[1],):
        raise ValueError("permutation must contain one entry per bit")
    if not np.array_equal(np.sort(permutation), np.arange(errors.shape[1])):
        raise ValueError("permutation must contain every bit index exactly once")
    return errors[:, permutation]


def bootstrap_sequence_sums(
    values: np.ndarray,
    *,
    group_size: int,
    repeats: int,
    rng: np.random.Generator,
) -> np.ndarray:
    """Bootstrap additive scores for sequences of independent readouts."""
    values = np.asarray(values, dtype=float)
    if values.ndim != 1 or values.size == 0:
        raise ValueError("values must be a nonempty vector")
    if group_size < 1 or repeats < 1:
        raise ValueError("group_size and repeats must be positive")
    indices = rng.integers(0, values.size, size=(repeats, group_size))
    return values[indices].sum(axis=1)


def bootstrap_sequence_vectors(
    values: np.ndarray,
    *,
    group_size: int,
    repeats: int,
    rng: np.random.Generator,
) -> np.ndarray:
    """Bootstrap paired additive score vectors using shared readout indices."""
    values = np.asarray(values, dtype=float)
    if values.ndim != 2 or values.shape[0] == 0:
        raise ValueError("values must be a nonempty matrix")
    if group_size < 1 or repeats < 1:
        raise ValueError("group_size and repeats must be positive")
    indices = rng.integers(0, len(values), size=(repeats, group_size))
    return values[indices].sum(axis=1)


def make_static_scenarios(
    *,
    train: np.ndarray,
    test: np.ndarray,
    impostor_errors: np.ndarray | None,
    n_samples: int,
    seed: int,
) -> dict[str, np.ndarray]:
    """Construct null/tampered scenarios that do not depend on an RBM."""
    rng = np.random.default_rng(seed)
    marginals = train.mean(axis=0)
    permutation = rng.permutation(train.shape[1])
    permuted_base = test[rng.integers(0, len(test), size=n_samples)]
    scenarios = {
        "marginal": rng.binomial(1, marginals, size=(n_samples, train.shape[1])),
        "mean_ber": rng.binomial(
            1, float(marginals.mean()), size=(n_samples, train.shape[1])
        ),
        "uniform": rng.binomial(1, 0.5, size=(n_samples, train.shape[1])),
        "permuted": fixed_bit_permutation(permuted_base, permutation),
    }
    if impostor_errors is not None:
        scenarios["cross_device"] = impostor_errors
    return {name: values.astype(np.uint8) for name, values in scenarios.items()}


def select_devices(n_devices: int, limit: int | None, seed: int) -> np.ndarray:
    if limit is None or limit >= n_devices:
        return np.arange(n_devices)
    if limit < 1:
        raise ValueError("devices-per-family must be positive")
    return np.sort(np.random.default_rng(seed).choice(n_devices, limit, replace=False))

