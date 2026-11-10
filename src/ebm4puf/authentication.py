"""Authentication scores and sequence tests for PUF error vectors."""

from collections.abc import Iterable
from statistics import NormalDist
from typing import Protocol

import numpy as np
from sklearn.covariance import LedoitWolf
from sklearn.metrics import roc_auc_score

from ebm4puf.sampling import bootstrap_sequence_sums, bootstrap_sequence_vectors


class ScoreModel(Protocol):
    """Energy model interface required by authentication statistics."""

    def score_samples(self, values: np.ndarray) -> np.ndarray: ...


def bernoulli_log_probability(
    errors: np.ndarray, probabilities: np.ndarray
) -> np.ndarray:
    """Log probability under independent per-bit Bernoulli variables."""
    probabilities = np.clip(np.asarray(probabilities, dtype=float), 1e-6, 1 - 1e-6)
    errors = np.asarray(errors)
    if errors.ndim != 2 or probabilities.shape != (errors.shape[1],):
        raise ValueError("probabilities must contain one value per error bit")
    return np.sum(
        errors * np.log(probabilities) + (1 - errors) * np.log1p(-probabilities),
        axis=1,
    )


def score_quantiles(values: np.ndarray, prefix: str) -> dict[str, float]:
    """Summarize a bootstrapped sequence-score distribution."""
    quantiles = np.quantile(values, [0.05, 0.25, 0.5, 0.75, 0.95])
    return {
        f"{prefix}_q05": float(quantiles[0]),
        f"{prefix}_q25": float(quantiles[1]),
        f"{prefix}_q50": float(quantiles[2]),
        f"{prefix}_q75": float(quantiles[3]),
        f"{prefix}_q95": float(quantiles[4]),
    }


def model_scores(
    model: ScoreModel, errors: np.ndarray, marginals: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Return unnormalized RBM and RBM-vs-independent evidence scores."""
    scores = score_channels(model, errors, marginals)
    rbm_score = scores["rbm"]
    evidence = scores["residual"]
    return rbm_score, evidence


def score_channels(
    model: ScoreModel,
    errors: np.ndarray,
    marginals: np.ndarray,
) -> dict[str, np.ndarray]:
    """Return marginal, raw RBM, and residual per-readout scores."""
    independent = bernoulli_log_probability(errors, marginals)
    rbm = model.score_samples(errors)
    return {
        "independent": independent,
        "rbm": rbm,
        "residual": rbm - independent,
    }


def joint_score_matrix(
    model: ScoreModel,
    errors: np.ndarray,
    marginals: np.ndarray,
) -> np.ndarray:
    """Return paired marginal and residual scores for each readout."""
    scores = score_channels(model, errors, marginals)
    return np.column_stack((scores["independent"], scores["residual"]))


def joint_score_statistics_rows(
    *,
    model: ScoreModel,
    populations: dict[str, np.ndarray],
    marginals: np.ndarray,
) -> list[dict[str, float | int | str]]:
    """Summarize paired marginal and residual per-readout scores."""
    rows = []
    for population, values in populations.items():
        scores = joint_score_matrix(model, values, marginals)
        covariance = (
            np.cov(scores, rowvar=False, ddof=1)
            if len(scores) > 1
            else np.zeros((2, 2), dtype=float)
        )
        denominator = np.sqrt(covariance[0, 0] * covariance[1, 1])
        rows.append(
            {
                "population": population,
                "n_vectors": len(scores),
                "marginal_mean": float(scores[:, 0].mean()),
                "residual_mean": float(scores[:, 1].mean()),
                "marginal_variance": float(covariance[0, 0]),
                "residual_variance": float(covariance[1, 1]),
                "marginal_residual_covariance": float(covariance[0, 1]),
                "marginal_residual_correlation": (
                    float(covariance[0, 1] / denominator)
                    if denominator > 0
                    else float("nan")
                ),
            }
        )
    return rows


def calibrated_anomaly_scores(
    sequence_scores: np.ndarray,
    *,
    per_readout_mean: np.ndarray,
    per_readout_covariance: np.ndarray,
    group_size: int,
) -> dict[str, np.ndarray]:
    """Return genuine-centered marginal, residual, and joint anomaly scores."""
    sequence_scores = np.asarray(sequence_scores, dtype=float)
    per_readout_mean = np.asarray(per_readout_mean, dtype=float)
    per_readout_covariance = np.asarray(per_readout_covariance, dtype=float)
    if sequence_scores.ndim != 2 or sequence_scores.shape[1] != 2:
        raise ValueError("sequence_scores must have shape (n_sequences, 2)")
    if per_readout_mean.shape != (2,) or per_readout_covariance.shape != (2, 2):
        raise ValueError("calibration moments must describe two score channels")
    if group_size < 1:
        raise ValueError("group_size must be positive")

    centered = sequence_scores - group_size * per_readout_mean
    sequence_covariance = group_size * per_readout_covariance
    ridge = max(float(np.trace(sequence_covariance)), 1.0) * 1e-9
    regularized = sequence_covariance + ridge * np.eye(2)
    inverse = np.linalg.inv(regularized)
    return {
        "marginal_anomaly": -(centered[:, 0] ** 2) / regularized[0, 0],
        "residual_anomaly": -(centered[:, 1] ** 2) / regularized[1, 1],
        "joint_mahalanobis": -np.einsum(
            "ni,ij,nj->n",
            centered,
            inverse,
            centered,
        ),
    }


def calibrated_directional_scores(
    sequence_scores: np.ndarray,
    *,
    per_readout_mean: np.ndarray,
    per_readout_covariance: np.ndarray,
    group_size: int,
    n_calibration: int,
) -> dict[str, np.ndarray]:
    """Return lower-tail scores including calibration-mean uncertainty."""
    sequence_scores = np.asarray(sequence_scores, dtype=float)
    per_readout_mean = np.asarray(per_readout_mean, dtype=float)
    per_readout_covariance = np.asarray(per_readout_covariance, dtype=float)
    if sequence_scores.ndim != 2 or sequence_scores.shape[1] != 2:
        raise ValueError("sequence_scores must have shape (n_sequences, 2)")
    if per_readout_mean.shape != (2,) or per_readout_covariance.shape != (2, 2):
        raise ValueError("calibration moments must describe two score channels")
    if group_size < 1 or n_calibration < 2:
        raise ValueError("group_size must be positive and calibration needs two rows")

    centered = sequence_scores - group_size * per_readout_mean
    predictive_variance = np.diag(per_readout_covariance) * (
        group_size + group_size**2 / n_calibration
    )
    floor = max(float(np.max(predictive_variance)), 1.0) * 1e-12
    standardized = centered / np.sqrt(np.maximum(predictive_variance, floor))
    return {
        "marginal_directional": standardized[:, 0],
        "residual_directional": standardized[:, 1],
        "joint_bonferroni": np.minimum(
            standardized[:, 0],
            standardized[:, 1],
        ),
    }


def gaussian_sequence_auc(
    genuine: np.ndarray,
    alternative: np.ndarray,
    group_size: int,
) -> float:
    """Approximate sequence AUROC from per-readout Gaussian moments."""
    genuine = np.asarray(genuine, dtype=float)
    alternative = np.asarray(alternative, dtype=float)
    if genuine.size < 2 or alternative.size < 2:
        raise ValueError("at least two scores per population are required")
    if group_size < 1:
        raise ValueError("group_size must be positive")

    mean_difference = float(genuine.mean() - alternative.mean())
    variance_sum = float(genuine.var(ddof=1) + alternative.var(ddof=1))
    if variance_sum == 0:
        if mean_difference > 0:
            return 1.0
        if mean_difference < 0:
            return 0.0
        return 0.5

    standardized_difference = (
        np.sqrt(group_size) * mean_difference / np.sqrt(variance_sum)
    )
    return float(NormalDist().cdf(standardized_difference))


def required_readouts_for_auc(
    genuine: np.ndarray,
    alternative: np.ndarray,
    target_auc: float,
) -> int | None:
    """Return the Gaussian-approximate N needed for a directional AUROC."""
    if not 0.5 < target_auc < 1:
        raise ValueError("target_auc must lie strictly between 0.5 and 1")

    genuine = np.asarray(genuine, dtype=float)
    alternative = np.asarray(alternative, dtype=float)
    if genuine.size < 2 or alternative.size < 2:
        raise ValueError("at least two scores per population are required")

    mean_difference = float(genuine.mean() - alternative.mean())
    if mean_difference <= 0:
        return None

    variance_sum = float(genuine.var(ddof=1) + alternative.var(ddof=1))
    if variance_sum == 0:
        return 1

    z_score = NormalDist().inv_cdf(target_auc)
    estimate = (z_score * np.sqrt(variance_sum) / mean_difference) ** 2
    return max(1, int(np.ceil(estimate)))


def score_statistics_rows(
    *,
    model: ScoreModel,
    populations: dict[str, np.ndarray],
    marginals: np.ndarray,
) -> list[dict[str, float | int | str]]:
    """Summarize per-readout score distributions for named populations."""
    rows = []
    for population, values in populations.items():
        for score_name, scores in score_channels(model, values, marginals).items():
            quantiles = np.quantile(scores, [0.05, 0.5, 0.95])
            rows.append(
                {
                    "population": population,
                    "score": score_name,
                    "n_vectors": len(scores),
                    "mean": float(scores.mean()),
                    "standard_deviation": float(
                        scores.std(ddof=1) if len(scores) > 1 else 0
                    ),
                    "q05": float(quantiles[0]),
                    "q50": float(quantiles[1]),
                    "q95": float(quantiles[2]),
                }
            )
    return rows


def evaluate_score_tests(
    *,
    model: ScoreModel,
    genuine: np.ndarray,
    scenarios: dict[str, np.ndarray],
    marginals: np.ndarray,
    group_sizes: Iterable[int],
    repeats: int,
    target_aucs: Iterable[float],
    seed: int,
) -> tuple[list[dict], list[dict]]:
    """Compare empirical and Gaussian AUROC for all three score channels."""
    rng = np.random.default_rng(seed)
    genuine_scores = score_channels(model, genuine, marginals)
    test_rows = []
    required_rows = []

    for scenario, values in scenarios.items():
        alternative_scores = score_channels(model, values, marginals)
        for score_name in genuine_scores:
            genuine_score = genuine_scores[score_name]
            alternative_score = alternative_scores[score_name]
            mean_difference = float(genuine_score.mean() - alternative_score.mean())
            genuine_std = float(genuine_score.std(ddof=1))
            alternative_std = float(alternative_score.std(ddof=1))

            for target_auc in target_aucs:
                required_rows.append(
                    {
                        "scenario": scenario,
                        "score": score_name,
                        "target_auc": target_auc,
                        "required_readouts": required_readouts_for_auc(
                            genuine_score,
                            alternative_score,
                            target_auc,
                        ),
                        "genuine_mean": float(genuine_score.mean()),
                        "alternative_mean": float(alternative_score.mean()),
                        "mean_difference": mean_difference,
                        "genuine_standard_deviation": genuine_std,
                        "alternative_standard_deviation": alternative_std,
                    }
                )

            for group_size in group_sizes:
                if group_size > len(genuine):
                    continue
                genuine_group = bootstrap_sequence_sums(
                    genuine_score,
                    group_size=group_size,
                    repeats=repeats,
                    rng=rng,
                )
                alternative_group = bootstrap_sequence_sums(
                    alternative_score,
                    group_size=group_size,
                    repeats=repeats,
                    rng=rng,
                )
                labels = np.concatenate(
                    [
                        np.ones(repeats, dtype=np.uint8),
                        np.zeros(repeats, dtype=np.uint8),
                    ]
                )
                test_rows.append(
                    {
                        "scenario": scenario,
                        "score": score_name,
                        "group_size": group_size,
                        "empirical_auc": float(
                            roc_auc_score(
                                labels,
                                np.concatenate([genuine_group, alternative_group]),
                            )
                        ),
                        "gaussian_auc": gaussian_sequence_auc(
                            genuine_score,
                            alternative_score,
                            group_size,
                        ),
                        "genuine_mean": float(genuine_score.mean()),
                        "alternative_mean": float(alternative_score.mean()),
                        "mean_difference": mean_difference,
                        "genuine_standard_deviation": genuine_std,
                        "alternative_standard_deviation": alternative_std,
                    }
                )

    return test_rows, required_rows


def evaluate_joint_score_tests(
    *,
    model: ScoreModel,
    calibration: np.ndarray,
    genuine: np.ndarray,
    scenarios: dict[str, np.ndarray],
    marginals: np.ndarray,
    group_sizes: Iterable[int],
    repeats: int,
    false_reject_rate: float,
    seed: int,
) -> list[dict]:
    """Evaluate genuine-centered one- and two-channel anomaly tests."""
    rng = np.random.default_rng(seed)
    calibration_scores = joint_score_matrix(model, calibration, marginals)
    genuine_scores = joint_score_matrix(model, genuine, marginals)
    calibration_fit = LedoitWolf().fit(calibration_scores)
    per_readout_mean = calibration_fit.location_
    per_readout_covariance = calibration_fit.covariance_
    rows = []

    for group_size in group_sizes:
        if group_size > len(genuine):
            continue
        calibration_group = bootstrap_sequence_vectors(
            calibration_scores,
            group_size=group_size,
            repeats=repeats,
            rng=rng,
        )
        genuine_group = bootstrap_sequence_vectors(
            genuine_scores,
            group_size=group_size,
            repeats=repeats,
            rng=rng,
        )
        calibration_anomaly = calibrated_anomaly_scores(
            calibration_group,
            per_readout_mean=per_readout_mean,
            per_readout_covariance=per_readout_covariance,
            group_size=group_size,
        )
        genuine_anomaly = calibrated_anomaly_scores(
            genuine_group,
            per_readout_mean=per_readout_mean,
            per_readout_covariance=per_readout_covariance,
            group_size=group_size,
        )
        calibration_directional = calibrated_directional_scores(
            calibration_group,
            per_readout_mean=per_readout_mean,
            per_readout_covariance=per_readout_covariance,
            group_size=group_size,
            n_calibration=len(calibration_scores),
        )
        genuine_directional = calibrated_directional_scores(
            genuine_group,
            per_readout_mean=per_readout_mean,
            per_readout_covariance=per_readout_covariance,
            group_size=group_size,
            n_calibration=len(calibration_scores),
        )
        calibration_tests = {
            **calibration_anomaly,
            **calibration_directional,
        }
        genuine_tests = {
            **genuine_anomaly,
            **genuine_directional,
        }

        for scenario, values in scenarios.items():
            alternative_group = bootstrap_sequence_vectors(
                joint_score_matrix(model, values, marginals),
                group_size=group_size,
                repeats=repeats,
                rng=rng,
            )
            alternative_anomaly = calibrated_anomaly_scores(
                alternative_group,
                per_readout_mean=per_readout_mean,
                per_readout_covariance=per_readout_covariance,
                group_size=group_size,
            )
            alternative_directional = calibrated_directional_scores(
                alternative_group,
                per_readout_mean=per_readout_mean,
                per_readout_covariance=per_readout_covariance,
                group_size=group_size,
                n_calibration=len(calibration_scores),
            )
            alternative_tests = {
                **alternative_anomaly,
                **alternative_directional,
            }
            labels = np.concatenate(
                [
                    np.ones(repeats, dtype=np.uint8),
                    np.zeros(repeats, dtype=np.uint8),
                ]
            )
            for test_name, calibration_values in calibration_tests.items():
                genuine_values = genuine_tests[test_name]
                alternative_values = alternative_tests[test_name]
                if test_name == "joint_bonferroni":
                    threshold = NormalDist().inv_cdf(false_reject_rate / 2)
                    calibration_method = "gaussian_predictive_bonferroni"
                elif test_name.endswith("_directional"):
                    threshold = NormalDist().inv_cdf(false_reject_rate)
                    calibration_method = "gaussian_predictive"
                else:
                    threshold = float(
                        np.quantile(calibration_values, false_reject_rate)
                    )
                    calibration_method = "empirical_quantile"
                rows.append(
                    {
                        "scenario": scenario,
                        "test": test_name,
                        "group_size": group_size,
                        "empirical_auc": float(
                            roc_auc_score(
                                labels,
                                np.concatenate([genuine_values, alternative_values]),
                            )
                        ),
                        "threshold": threshold,
                        "calibration_method": calibration_method,
                        "target_false_reject_rate": false_reject_rate,
                        "calibration_acceptance_rate": float(
                            np.mean(calibration_values >= threshold)
                        ),
                        "genuine_acceptance_rate": float(
                            np.mean(genuine_values >= threshold)
                        ),
                        "observed_false_reject_rate": float(
                            np.mean(genuine_values < threshold)
                        ),
                        "alternative_acceptance_rate": float(
                            np.mean(alternative_values >= threshold)
                        ),
                    }
                )
    return rows


def center_evidence(
    rows: list[dict[str, float | int | str]],
    marginal_baselines: dict[int, float] | None = None,
) -> dict[int, float]:
    """Center evidence quantiles on the matched-marginal median at each N."""
    if marginal_baselines is None:
        marginal_baselines = {
            int(row["group_size"]): float(row["evidence_q50"])
            for row in rows
            if row["scenario"] == "marginal"
        }
    for row in rows:
        baseline = marginal_baselines[int(row["group_size"])]
        for quantile in ("q05", "q25", "q50", "q75", "q95"):
            row[f"centered_evidence_{quantile}"] = (
                float(row[f"evidence_{quantile}"]) - baseline
            )
    return marginal_baselines


def evaluate_authentication(
    *,
    model: ScoreModel,
    calibration: np.ndarray,
    test: np.ndarray,
    scenarios: dict[str, np.ndarray],
    marginals: np.ndarray,
    group_sizes: Iterable[int],
    repeats: int,
    false_reject_rate: float,
    seed: int,
) -> list[dict[str, float | int | str]]:
    """Evaluate calibrated additive evidence and free-energy sequence scores."""
    rng = np.random.default_rng(seed)
    calibration_rbm, calibration_evidence = model_scores(model, calibration, marginals)
    test_rbm, test_evidence = model_scores(model, test, marginals)
    rows = []
    for group_size in group_sizes:
        if group_size > min(len(calibration), len(test)):
            continue
        calibration_evidence_group = bootstrap_sequence_sums(
            calibration_evidence,
            group_size=group_size,
            repeats=repeats,
            rng=rng,
        )
        calibration_rbm_group = bootstrap_sequence_sums(
            calibration_rbm,
            group_size=group_size,
            repeats=repeats,
            rng=rng,
        )
        genuine_evidence = bootstrap_sequence_sums(
            test_evidence,
            group_size=group_size,
            repeats=repeats,
            rng=rng,
        )
        genuine_rbm = bootstrap_sequence_sums(
            test_rbm,
            group_size=group_size,
            repeats=repeats,
            rng=rng,
        )
        evidence_threshold = float(
            np.quantile(calibration_evidence_group, false_reject_rate)
        )
        free_energy_threshold = float(
            np.quantile(calibration_rbm_group, false_reject_rate)
        )
        genuine_row = {
            "scenario": "genuine",
            "group_size": group_size,
            "free_energy_threshold": free_energy_threshold,
            "free_energy_acceptance_rate": float(
                np.mean(genuine_rbm >= free_energy_threshold)
            ),
            "evidence_threshold": evidence_threshold,
            "evidence_acceptance_rate": float(
                np.mean(genuine_evidence >= evidence_threshold)
            ),
            "evidence_auc": float("nan"),
            "free_energy_auc": float("nan"),
            "mean_evidence": float(genuine_evidence.mean()),
            "mean_rbm_score": float(genuine_rbm.mean()),
        }
        genuine_row.update(score_quantiles(genuine_rbm, "rbm_score"))
        genuine_row.update(score_quantiles(genuine_evidence, "evidence"))
        rows.append(genuine_row)
        for scenario, values in scenarios.items():
            alternative_rbm, alternative_evidence = model_scores(
                model, values, marginals
            )
            alternative_evidence_group = bootstrap_sequence_sums(
                alternative_evidence,
                group_size=group_size,
                repeats=repeats,
                rng=rng,
            )
            alternative_rbm_group = bootstrap_sequence_sums(
                alternative_rbm,
                group_size=group_size,
                repeats=repeats,
                rng=rng,
            )
            labels = np.concatenate(
                [np.ones(repeats, dtype=np.uint8), np.zeros(repeats, dtype=np.uint8)]
            )
            alternative_row = {
                "scenario": scenario,
                "group_size": group_size,
                "free_energy_threshold": free_energy_threshold,
                "free_energy_acceptance_rate": float(
                    np.mean(alternative_rbm_group >= free_energy_threshold)
                ),
                "evidence_threshold": evidence_threshold,
                "evidence_acceptance_rate": float(
                    np.mean(alternative_evidence_group >= evidence_threshold)
                ),
                "evidence_auc": float(
                    roc_auc_score(
                        labels,
                        np.concatenate([genuine_evidence, alternative_evidence_group]),
                    )
                ),
                "free_energy_auc": float(
                    roc_auc_score(
                        labels,
                        np.concatenate([genuine_rbm, alternative_rbm_group]),
                    )
                ),
                "mean_evidence": float(alternative_evidence_group.mean()),
                "mean_rbm_score": float(alternative_rbm_group.mean()),
            }
            alternative_row.update(score_quantiles(alternative_rbm_group, "rbm_score"))
            alternative_row.update(
                score_quantiles(alternative_evidence_group, "evidence")
            )
            rows.append(alternative_row)
    return rows

