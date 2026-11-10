"""Model-free analysis of PUF error counts and error locations."""

from collections.abc import Sequence

import numpy as np

from ebm4puf.probability import poisson_binomial_pmf


def leave_one_device_out_prior(
    family_error_counts: np.ndarray,
    family_readouts: int,
    excluded_errors: np.ndarray,
    *,
    prior_strength: float,
) -> np.ndarray:
    """Estimate a rare-error family bit prior without the target device."""
    family_error_counts = np.asarray(family_error_counts, dtype=float)
    excluded_errors = np.asarray(excluded_errors, dtype=float)
    if family_error_counts.ndim != 1 or excluded_errors.ndim != 2:
        raise ValueError("family counts must be 1D and excluded errors must be 2D")
    if family_error_counts.size != excluded_errors.shape[1]:
        raise ValueError("family and excluded-device bit counts do not match")
    remaining_readouts = family_readouts - len(excluded_errors)
    if remaining_readouts < 1:
        raise ValueError("a leave-one-device-out prior requires another device")
    if prior_strength <= 0:
        raise ValueError("family prior strength must be positive")

    remaining_counts = family_error_counts - excluded_errors.sum(axis=0)
    if np.any(remaining_counts < 0):
        raise ValueError("excluded errors exceed family error counts")
    family_ber = remaining_counts.sum() / (
        remaining_readouts * family_error_counts.size
    )
    prior = (remaining_counts + prior_strength * family_ber) / (
        remaining_readouts + prior_strength
    )
    floor = 0.5 / (remaining_readouts * family_error_counts.size + 1)
    return np.clip(prior, floor, 1 - floor)


def select_shrinkage_strength(
    enrollment_errors: np.ndarray,
    family_prior: np.ndarray,
    strengths: np.ndarray,
) -> float:
    """Select family-to-device shrinkage by leave-one-readout-out likelihood."""
    enrollment_errors = np.asarray(enrollment_errors, dtype=float)
    family_prior = np.asarray(family_prior, dtype=float)
    strengths = np.asarray(strengths, dtype=float)
    if enrollment_errors.ndim != 2 or len(enrollment_errors) < 2:
        raise ValueError("at least two enrollment readouts are required")
    if family_prior.shape != (enrollment_errors.shape[1],):
        raise ValueError("family prior must contain one probability per bit")
    if strengths.ndim != 1 or strengths.size == 0 or np.any(strengths <= 0):
        raise ValueError("shrinkage strengths must be a nonempty positive vector")

    counts = enrollment_errors.sum(axis=0)
    scores = []
    for strength in strengths:
        probabilities = (
            counts[None, :] - enrollment_errors + strength * family_prior
        ) / (len(enrollment_errors) - 1 + strength)
        probabilities = np.clip(probabilities, 1e-12, 1 - 1e-12)
        scores.append(
            np.sum(
                enrollment_errors * np.log(probabilities)
                + (1 - enrollment_errors) * np.log1p(-probabilities)
            )
        )
    return float(strengths[int(np.argmax(scores))])


def shrink_marginals(
    enrollment_errors: np.ndarray,
    family_prior: np.ndarray,
    strength: float,
) -> np.ndarray:
    """Shrink device bit marginals toward a rare-error family prior."""
    enrollment_errors = np.asarray(enrollment_errors, dtype=float)
    family_prior = np.asarray(family_prior, dtype=float)
    if enrollment_errors.ndim != 2:
        raise ValueError("enrollment errors must be two-dimensional")
    if family_prior.shape != (enrollment_errors.shape[1],):
        raise ValueError("family prior must contain one probability per bit")
    if strength <= 0:
        raise ValueError("shrinkage strength must be positive")
    return (enrollment_errors.sum(axis=0) + strength * family_prior) / (
        len(enrollment_errors) + strength
    )


def poisson_prefix_suffix(
    probabilities: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Return forward and backward Poisson-binomial probability tables."""
    n_bits = probabilities.size
    prefix = np.zeros((n_bits + 1, n_bits + 1), dtype=float)
    suffix = np.zeros((n_bits + 1, n_bits + 1), dtype=float)
    prefix[0, 0] = 1.0
    suffix[n_bits, 0] = 1.0
    for bit, probability in enumerate(probabilities):
        prefix[bit + 1] = prefix[bit] * (1 - probability)
        prefix[bit + 1, 1:] += prefix[bit, :-1] * probability
    for bit in range(n_bits - 1, -1, -1):
        probability = probabilities[bit]
        suffix[bit] = suffix[bit + 1] * (1 - probability)
        suffix[bit, 1:] += suffix[bit + 1, :-1] * probability
    return prefix, suffix


def tilt_probabilities_to_count(
    probabilities: np.ndarray,
    count: int,
) -> np.ndarray:
    """Exponentially tilt Bernoulli odds so their expected sum is count."""
    logits = np.log(probabilities) - np.log1p(-probabilities)
    lower = -1_000.0
    upper = 1_000.0
    for _ in range(100):
        midpoint = (lower + upper) / 2
        shifted = np.clip(logits + midpoint, -700, 700)
        tilted = 1 / (1 + np.exp(-shifted))
        if tilted.sum() < count:
            lower = midpoint
        else:
            upper = midpoint
    shifted = np.clip(logits + (lower + upper) / 2, -700, 700)
    return 1 / (1 + np.exp(-shifted))


def conditional_independent_probabilities(
    probabilities: np.ndarray,
    counts: np.ndarray | Sequence[int],
) -> dict[int, np.ndarray]:
    """Return P(E_i=1 | K=k) under independent Bernoulli marginals."""
    probabilities = np.asarray(probabilities, dtype=float)
    counts = np.unique(np.asarray(counts, dtype=int))
    if probabilities.ndim != 1:
        raise ValueError("probabilities must be one-dimensional")
    if np.any((probabilities <= 0) | (probabilities >= 1)):
        raise ValueError("probabilities must lie strictly between zero and one")
    if np.any((counts < 0) | (counts > probabilities.size)):
        raise ValueError("counts must lie between zero and the number of bits")

    n_bits = probabilities.size
    result = {int(k): np.zeros(n_bits, dtype=float) for k in counts}
    positive_counts = counts[counts > 0]
    if positive_counts.size == 0:
        return result

    # Prefix/suffix distributions avoid the catastrophic cancellation caused
    # by polynomial division in rare, high-count strata.
    prefix, suffix = poisson_prefix_suffix(probabilities)

    for k in positive_counts:
        if k == n_bits:
            result[int(k)].fill(1)
            continue
        working_probabilities = probabilities
        working_prefix = prefix
        working_suffix = suffix
        denominator = prefix[n_bits, k]
        if denominator < 1e-250:
            # Exponential tilting multiplies every K=k vector by the same
            # factor, preserving the conditional distribution while moving k
            # near the numerical center of the Poisson-binomial PMF.
            working_probabilities = tilt_probabilities_to_count(probabilities, int(k))
            working_prefix, working_suffix = poisson_prefix_suffix(
                working_probabilities
            )
            denominator = working_prefix[n_bits, k]
        if not np.isfinite(denominator) or denominator <= 0:
            raise FloatingPointError(
                f"cannot resolve independent conditional probability for K={k}"
            )
        target = int(k) - 1
        row = result[int(k)]
        for bit, probability in enumerate(working_probabilities):
            lower = max(0, target - (n_bits - bit - 1))
            upper = min(target, bit)
            left = working_prefix[bit, lower : upper + 1]
            right = working_suffix[bit + 1, target - upper : target - lower + 1][::-1]
            row[bit] = probability * np.dot(left, right) / denominator

        if not np.all(np.isfinite(row)):
            raise FloatingPointError(f"nonfinite conditional probabilities for K={k}")
        if np.any((row < -1e-12) | (row > 1 + 1e-12)):
            raise FloatingPointError(
                f"conditional probabilities outside [0, 1] for K={k}"
            )
        np.clip(row, 0, 1, out=row)
        total = row.sum()
        if total <= 0:
            raise FloatingPointError(f"zero conditional probability mass for K={k}")
        row *= int(k) / total
        if np.any(row > 1 + 1e-10) or not np.isclose(row.sum(), int(k), atol=1e-10):
            raise FloatingPointError(f"invalid conditional normalization for K={k}")
    return result


def cross_validation_folds(
    n_readouts: int,
    n_folds: int,
    seed: int,
) -> list[np.ndarray]:
    """Return deterministic, nonempty held-out index folds."""
    if n_folds < 2 or n_folds > n_readouts:
        raise ValueError("folds must lie between 2 and the number of readouts")
    indices = np.random.default_rng(seed).permutation(n_readouts)
    return [fold for fold in np.array_split(indices, n_folds) if len(fold)]


def analyze_device(
    errors: np.ndarray,
    *,
    family: str,
    device: int,
    family_prior: np.ndarray,
    shrinkage_strengths: np.ndarray,
    n_folds: int,
    seed: int,
) -> tuple[list[dict], list[dict], list[dict], dict]:
    """Cross-fit count and conditional-location statistics for one device."""
    errors = np.asarray(errors, dtype=np.uint8)
    if errors.ndim != 2 or len(errors) < 2:
        raise ValueError("errors must contain at least two readouts")
    if not np.all((errors == 0) | (errors == 1)):
        raise ValueError("errors must be binary")

    n_readouts, n_bits = errors.shape
    folds = cross_validation_folds(n_readouts, n_folds, seed)
    all_indices = np.arange(n_readouts)
    empirical_counts = np.bincount(errors.sum(axis=1), minlength=n_bits + 1)
    independent_pmf = np.zeros(n_bits + 1, dtype=float)
    conditional_null_sum: dict[int, np.ndarray] = {}
    marginal_rows = []
    selected_strengths = []

    for fold_index, held_out in enumerate(folds):
        enrollment = np.setdiff1d(all_indices, held_out, assume_unique=True)
        enrollment_errors = errors[enrollment]
        strength = select_shrinkage_strength(
            enrollment_errors,
            family_prior,
            shrinkage_strengths,
        )
        marginals = shrink_marginals(enrollment_errors, family_prior, strength)
        selected_strengths.append(strength)
        enrollment_counts = enrollment_errors.sum(axis=0)
        for bit in range(n_bits):
            marginal_rows.append(
                {
                    "family": family,
                    "device": device,
                    "fold": fold_index,
                    "bit": bit,
                    "enrollment_size": len(enrollment),
                    "enrollment_error_count": int(enrollment_counts[bit]),
                    "family_prior_probability": float(family_prior[bit]),
                    "shrinkage_strength": strength,
                    "marginal_probability": float(marginals[bit]),
                }
            )
        independent_pmf += len(held_out) / n_readouts * poisson_binomial_pmf(marginals)

        held_out_errors = errors[held_out]
        held_out_counts = held_out_errors.sum(axis=1).astype(int)
        conditional_null = conditional_independent_probabilities(
            marginals, np.unique(held_out_counts)
        )
        for k, probabilities in conditional_null.items():
            support = int(np.count_nonzero(held_out_counts == k))
            conditional_null_sum.setdefault(k, np.zeros(n_bits, dtype=float))
            conditional_null_sum[k] += support * probabilities

    count_rows = []
    empirical_pmf = empirical_counts / n_readouts
    for k in range(n_bits + 1):
        count_rows.append(
            {
                "family": family,
                "device": device,
                "k": k,
                "n_readouts": n_readouts,
                "empirical_count": int(empirical_counts[k]),
                "empirical_probability": float(empirical_pmf[k]),
                "independent_probability": float(independent_pmf[k]),
                "probability_difference": float(empirical_pmf[k] - independent_pmf[k]),
            }
        )

    error_counts = errors.sum(axis=1).astype(int)
    conditional_rows = []
    conditional_distance = 0.0
    conditional_mae = 0.0
    for k in np.flatnonzero(empirical_counts):
        selected = errors[error_counts == k]
        support = len(selected)
        flip_counts = selected.sum(axis=0)
        empirical = flip_counts / support
        independent = conditional_null_sum[int(k)] / support
        residual = empirical - independent
        if not np.all(np.isfinite(independent)) or np.any(
            (independent < -1e-12) | (independent > 1 + 1e-12)
        ):
            raise FloatingPointError(
                f"invalid independent probabilities for {family} device {device}, K={k}"
            )
        if not np.isclose(independent.sum(), int(k), atol=1e-8):
            raise FloatingPointError(
                f"independent probabilities do not sum to K for "
                f"{family} device {device}, K={k}"
            )
        if not np.isclose(residual.sum(), 0, atol=1e-8):
            raise FloatingPointError(
                f"conditional residuals do not sum to zero for "
                f"{family} device {device}, K={k}"
            )
        conditional_mae += empirical_pmf[k] * np.mean(np.abs(residual))
        denominator = 2 * min(int(k), n_bits - int(k))
        if denominator:
            conditional_distance += empirical_pmf[k] * (
                np.sum(np.abs(residual)) / denominator
            )
        for bit in range(n_bits):
            conditional_rows.append(
                {
                    "family": family,
                    "device": device,
                    "k": int(k),
                    "bit": bit,
                    "support": support,
                    "flip_count": int(flip_counts[bit]),
                    "empirical_probability": float(empirical[bit]),
                    "independent_probability": float(independent[bit]),
                    "residual": float(residual[bit]),
                }
            )

    metrics = {
        "family": family,
        "device": device,
        "n_readouts": n_readouts,
        "n_bits": n_bits,
        "n_folds": len(folds),
        "shrinkage_strength_mean": float(np.mean(selected_strengths)),
        "shrinkage_strength_min": float(np.min(selected_strengths)),
        "shrinkage_strength_max": float(np.max(selected_strengths)),
        "family_prior_mean_error_count": float(family_prior.sum()),
        "mean_error_count": float(errors.sum(axis=1).mean()),
        "independent_mean_error_count": float(
            np.dot(np.arange(n_bits + 1), independent_pmf)
        ),
        "count_cdf_distance": float(
            np.max(np.abs(np.cumsum(empirical_pmf - independent_pmf)))
        ),
        "count_total_variation": float(
            0.5 * np.sum(np.abs(empirical_pmf - independent_pmf))
        ),
        "conditional_residual_distance": float(conditional_distance),
        "conditional_residual_mae": float(conditional_mae),
    }
    return count_rows, conditional_rows, marginal_rows, metrics


def reference_polarity_statistics(
    errors: np.ndarray,
    reference: np.ndarray,
    conditional_rows: list[dict],
    *,
    family: str,
    device: int,
) -> tuple[list[dict], dict, np.ndarray]:
    """Summarize 0-to-1 versus 1-to-0 conditional error behavior."""
    errors = np.asarray(errors, dtype=np.uint8)
    reference = np.asarray(reference, dtype=np.uint8)
    if errors.ndim != 2 or reference.shape != (errors.shape[1],):
        raise ValueError("reference must contain one value per error bit")
    if not np.all((reference == 0) | (reference == 1)):
        raise ValueError("reference must be binary")
    masks = {value: reference == value for value in (0, 1)}
    if not all(mask.any() for mask in masks.values()):
        raise ValueError("reference-polarity analysis requires both bit values")

    rows = []
    weighted_independent = {0: 0.0, 1: 0.0}
    weighted_residual_by_bit = np.zeros(reference.size, dtype=float)
    for k in sorted({int(row["k"]) for row in conditional_rows}):
        selected = [row for row in conditional_rows if int(row["k"]) == k]
        selected.sort(key=lambda row: int(row["bit"]))
        support = int(selected[0]["support"])
        empirical = np.array([row["empirical_probability"] for row in selected])
        independent = np.array([row["independent_probability"] for row in selected])
        residual = empirical - independent
        weighted_residual_by_bit += support * residual / len(errors)
        statistics = {}
        for value, mask in masks.items():
            statistics[value] = {
                "empirical": float(empirical[mask].mean()),
                "independent": float(independent[mask].mean()),
                "residual": float(residual[mask].mean()),
            }
            weighted_independent[value] += (
                support / len(errors) * statistics[value]["independent"]
            )
        empirical_contrast = statistics[1]["empirical"] - statistics[0]["empirical"]
        independent_contrast = (
            statistics[1]["independent"] - statistics[0]["independent"]
        )
        residual_contrast = statistics[1]["residual"] - statistics[0]["residual"]
        for value, mask in masks.items():
            rows.append(
                {
                    "family": family,
                    "device": device,
                    "k": k,
                    "reference_value": value,
                    "reference_bit_count": int(mask.sum()),
                    "reference_bit_fraction": float(mask.mean()),
                    "support": support,
                    "empirical_flip_probability": statistics[value]["empirical"],
                    "independent_flip_probability": statistics[value]["independent"],
                    "residual": statistics[value]["residual"],
                    "empirical_contrast": empirical_contrast,
                    "independent_contrast": independent_contrast,
                    "residual_contrast": residual_contrast,
                }
            )

    empirical_by_bit = errors.mean(axis=0)
    raw_contrast = float(
        empirical_by_bit[masks[1]].mean() - empirical_by_bit[masks[0]].mean()
    )
    independent_contrast = weighted_independent[1] - weighted_independent[0]
    summary = {
        "family": family,
        "device": device,
        "n_readouts": len(errors),
        "reference_zero_bits": int(masks[0].sum()),
        "reference_one_bits": int(masks[1].sum()),
        "reference_one_fraction": float(masks[1].mean()),
        "raw_transition_contrast": raw_contrast,
        "independent_transition_contrast": independent_contrast,
        "residual_transition_contrast": raw_contrast - independent_contrast,
    }
    return rows, summary, weighted_residual_by_bit


def summarize_reference_polarity_family(
    device_rows: list[dict],
    weighted_residuals: list[np.ndarray],
    references: np.ndarray,
    *,
    bootstrap_repeats: int,
    permutation_repeats: int,
    seed: int,
) -> dict:
    """Return family polarity effects with device bootstrap and label null."""
    if not device_rows or len(device_rows) != len(weighted_residuals):
        raise ValueError("polarity summaries require aligned device results")
    if references.shape != (len(device_rows), len(weighted_residuals[0])):
        raise ValueError("reference matrix does not match polarity summaries")
    raw = np.array([row["raw_transition_contrast"] for row in device_rows])
    residual = np.array([row["residual_transition_contrast"] for row in device_rows])
    rng = np.random.default_rng(seed)
    bootstrap_raw = np.empty(bootstrap_repeats)
    bootstrap_residual = np.empty(bootstrap_repeats)
    for repeat in range(bootstrap_repeats):
        selected = rng.integers(0, len(device_rows), len(device_rows))
        bootstrap_raw[repeat] = raw[selected].mean()
        bootstrap_residual[repeat] = residual[selected].mean()

    observed = float(residual.mean())
    exceedances = 0
    for _ in range(permutation_repeats):
        contrasts = []
        for values, reference in zip(weighted_residuals, references):
            permuted = rng.permutation(reference)
            contrasts.append(
                values[permuted == 1].mean() - values[permuted == 0].mean()
            )
        exceedances += abs(np.mean(contrasts)) >= abs(observed)
    raw_bounds = np.quantile(bootstrap_raw, [0.025, 0.975])
    residual_bounds = np.quantile(bootstrap_residual, [0.025, 0.975])
    return {
        "family": device_rows[0]["family"],
        "n_devices": len(device_rows),
        "mean_reference_one_fraction": float(
            np.mean([row["reference_one_fraction"] for row in device_rows])
        ),
        "mean_raw_transition_contrast": float(raw.mean()),
        "raw_bootstrap_q025": float(raw_bounds[0]),
        "raw_bootstrap_q975": float(raw_bounds[1]),
        "mean_residual_transition_contrast": observed,
        "residual_bootstrap_q025": float(residual_bounds[0]),
        "residual_bootstrap_q975": float(residual_bounds[1]),
        "residual_contrast_q25": float(np.quantile(residual, 0.25)),
        "residual_contrast_median": float(np.median(residual)),
        "residual_contrast_q75": float(np.quantile(residual, 0.75)),
        "fraction_devices_positive_residual": float(np.mean(residual > 0)),
        "reference_label_permutation_p_value": (exceedances + 1)
        / (permutation_repeats + 1),
    }


def tail_probabilities(pmfs: np.ndarray) -> np.ndarray:
    """Return P(K > tau) for every PMF row and correction capacity tau."""
    pmfs = np.asarray(pmfs, dtype=float)
    if pmfs.ndim != 2 or np.any(pmfs < 0):
        raise ValueError("pmfs must be a nonnegative matrix")
    if not np.allclose(pmfs.sum(axis=1), 1, atol=1e-8):
        raise ValueError("each PMF must sum to one")
    reverse_cdf = np.cumsum(pmfs[:, ::-1], axis=1)[:, ::-1]
    return np.clip(reverse_cdf - pmfs, 0, 1)


def first_capacity(values: np.ndarray, threshold: float) -> int | None:
    """Return the first capacity whose value is at most the threshold."""
    indices = np.flatnonzero(np.asarray(values) <= threshold)
    return int(indices[0]) if indices.size else None


def population_error_statistics(
    empirical_pmfs: np.ndarray,
    independent_pmfs: np.ndarray,
    readout_counts: np.ndarray,
    *,
    family: str,
    target_failure_probabilities: np.ndarray,
    target_device_coverages: np.ndarray,
    bootstrap_repeats: int,
    seed: int,
) -> tuple[list[dict], list[dict], list[dict], list[dict]]:
    """Summarize random-device tails, device coverage, and capacities."""
    empirical_pmfs = np.asarray(empirical_pmfs, dtype=float)
    independent_pmfs = np.asarray(independent_pmfs, dtype=float)
    readout_counts = np.asarray(readout_counts, dtype=int)
    if empirical_pmfs.shape != independent_pmfs.shape or empirical_pmfs.ndim != 2:
        raise ValueError("empirical and independent PMFs must have the same shape")
    if readout_counts.shape != (len(empirical_pmfs),) or np.any(readout_counts < 1):
        raise ValueError("readout counts must contain one positive value per device")
    if bootstrap_repeats < 0:
        raise ValueError("bootstrap repeats cannot be negative")

    empirical_tails = tail_probabilities(empirical_pmfs)
    independent_tails = tail_probabilities(independent_pmfs)
    n_devices, n_counts = empirical_pmfs.shape
    if not np.all(readout_counts == readout_counts[0]):
        raise ValueError(
            "device coverage requires equal readout counts within a family"
        )
    n_readouts = int(readout_counts[0])
    observed_max = int(max(np.flatnonzero(pmf > 0)[-1] for pmf in empirical_pmfs))

    bootstrap_tails = np.empty((bootstrap_repeats, n_counts), dtype=float)
    rng = np.random.default_rng(seed)
    for repeat in range(bootstrap_repeats):
        selected = rng.integers(0, n_devices, size=n_devices)
        mixture = np.zeros(n_counts, dtype=float)
        for device in selected:
            sampled = rng.multinomial(
                int(readout_counts[device]), empirical_pmfs[device]
            )
            mixture += sampled / readout_counts[device]
        bootstrap_tails[repeat] = tail_probabilities((mixture / n_devices)[None, :])[0]

    device_rows = []
    population_rows = []
    coverage_rows = []
    capacity_rows = []
    empirical_mean = empirical_tails.mean(axis=0)
    independent_mean = independent_tails.mean(axis=0)
    empirical_quantiles = np.quantile(
        empirical_tails, [0.25, 0.5, 0.75, 0.9, 0.95], axis=0
    )
    independent_quantiles = np.quantile(independent_tails, [0.5, 0.9, 0.95], axis=0)
    if bootstrap_repeats:
        bootstrap_bounds = np.quantile(bootstrap_tails, [0.025, 0.975], axis=0)
    else:
        bootstrap_bounds = np.full((2, n_counts), np.nan)

    for device in range(n_devices):
        for tau in range(n_counts):
            device_rows.append(
                {
                    "family": family,
                    "device": device,
                    "tau": tau,
                    "empirical_tail_probability": float(empirical_tails[device, tau]),
                    "independent_tail_probability": float(
                        independent_tails[device, tau]
                    ),
                }
            )
    for tau in range(n_counts):
        population_rows.append(
            {
                "family": family,
                "tau": tau,
                "n_devices": n_devices,
                "empirical_random_device_tail": float(empirical_mean[tau]),
                "independent_random_device_tail": float(independent_mean[tau]),
                "empirical_device_q25": float(empirical_quantiles[0, tau]),
                "empirical_device_q50": float(empirical_quantiles[1, tau]),
                "empirical_device_q75": float(empirical_quantiles[2, tau]),
                "empirical_device_q90": float(empirical_quantiles[3, tau]),
                "empirical_device_q95": float(empirical_quantiles[4, tau]),
                "independent_device_q50": float(independent_quantiles[0, tau]),
                "independent_device_q90": float(independent_quantiles[1, tau]),
                "independent_device_q95": float(independent_quantiles[2, tau]),
                "empirical_bootstrap_q025": float(bootstrap_bounds[0, tau]),
                "empirical_bootstrap_q975": float(bootstrap_bounds[1, tau]),
                "observed_max_k": observed_max,
            }
        )
        for target in target_failure_probabilities:
            maximum_failures = int(np.floor(target * n_readouts + 1e-12))
            coverage_rows.append(
                {
                    "family": family,
                    "tau": tau,
                    "target_failure_probability": float(target),
                    "n_readouts": n_readouts,
                    "maximum_observed_failures": maximum_failures,
                    "effective_empirical_probability": (maximum_failures / n_readouts),
                    "empirical_device_coverage": float(
                        np.mean(empirical_tails[:, tau] <= target)
                    ),
                    "independent_device_coverage": float(
                        np.mean(independent_tails[:, tau] <= target)
                    ),
                }
            )

    for target in target_failure_probabilities:
        empirical_population_capacity = first_capacity(empirical_mean, float(target))
        independent_population_capacity = first_capacity(
            independent_mean, float(target)
        )
        for coverage in target_device_coverages:
            empirical_coverage_curve = np.mean(empirical_tails <= target, axis=0)
            independent_coverage_curve = np.mean(independent_tails <= target, axis=0)
            empirical_device_capacity = np.flatnonzero(
                empirical_coverage_curve >= coverage
            )
            independent_device_capacity = np.flatnonzero(
                independent_coverage_curve >= coverage
            )
            capacity_rows.append(
                {
                    "family": family,
                    "target_failure_probability": float(target),
                    "target_device_coverage": float(coverage),
                    "empirical_population_capacity": empirical_population_capacity,
                    "independent_population_capacity": independent_population_capacity,
                    "empirical_device_coverage_capacity": (
                        int(empirical_device_capacity[0])
                        if empirical_device_capacity.size
                        else None
                    ),
                    "independent_device_coverage_capacity": (
                        int(independent_device_capacity[0])
                        if independent_device_capacity.size
                        else None
                    ),
                }
            )
    return device_rows, population_rows, coverage_rows, capacity_rows

