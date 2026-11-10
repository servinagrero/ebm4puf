"""Distribution-fidelity metrics for generated PUF error vectors."""

import numpy as np
from sklearn.covariance import LedoitWolf


def cdf_distance(reference: np.ndarray, candidate: np.ndarray, n_bits: int) -> float:
    """Maximum CDF distance between two error-count distributions."""
    reference_pmf = np.bincount(reference.sum(axis=1), minlength=n_bits + 1)
    candidate_pmf = np.bincount(candidate.sum(axis=1), minlength=n_bits + 1)
    reference_cdf = np.cumsum(reference_pmf / reference_pmf.sum())
    candidate_cdf = np.cumsum(candidate_pmf / candidate_pmf.sum())
    return float(np.max(np.abs(reference_cdf - candidate_cdf)))


def matrix_fidelity(reference: np.ndarray, candidate: np.ndarray) -> dict[str, float]:
    """Compare Ledoit-Wolf covariance and precision estimates."""
    reference_fit = LedoitWolf().fit(reference)
    candidate_fit = LedoitWolf().fit(candidate)
    metrics = {}
    for name in ("covariance", "precision"):
        reference_matrix = getattr(reference_fit, f"{name}_")
        candidate_matrix = getattr(candidate_fit, f"{name}_")
        denominator = np.linalg.norm(reference_matrix)
        metrics[f"{name}_relative_frobenius"] = float(
            np.linalg.norm(reference_matrix - candidate_matrix)
            / max(denominator, np.finfo(float).eps)
        )
        mask = ~np.eye(reference_matrix.shape[0], dtype=bool)
        left = reference_matrix[mask]
        right = candidate_matrix[mask]
        if np.std(left) == 0 or np.std(right) == 0:
            metrics[f"{name}_offdiagonal_correlation"] = float("nan")
        else:
            metrics[f"{name}_offdiagonal_correlation"] = float(
                np.corrcoef(left, right)[0, 1]
            )
    return metrics


def fidelity_metrics(
    reference: np.ndarray, candidate: np.ndarray, *, source: str
) -> dict[str, float | str]:
    """Return marginal, error-count, covariance, and precision discrepancies."""
    result: dict[str, float | str] = {
        "source": source,
        "marginal_mae": float(
            np.mean(np.abs(reference.mean(axis=0) - candidate.mean(axis=0)))
        ),
        "error_count_cdf_distance": cdf_distance(
            reference, candidate, reference.shape[1]
        ),
        "mean_error_count_real": float(reference.sum(axis=1).mean()),
        "mean_error_count_generated": float(candidate.sum(axis=1).mean()),
    }
    result.update(matrix_fidelity(reference, candidate))
    return result

