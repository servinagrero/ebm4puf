"""Probability utilities shared by PUF analyses."""

import numpy as np


def poisson_binomial_pmf(probabilities: np.ndarray) -> np.ndarray:
    """Return the exact PMF of a sum of independent Bernoulli variables."""
    probabilities = np.asarray(probabilities, dtype=float)
    if probabilities.ndim != 1:
        raise ValueError("probabilities must be one-dimensional")
    if np.any((probabilities < 0) | (probabilities > 1)):
        raise ValueError("probabilities must lie in [0, 1]")

    pmf = np.zeros(probabilities.size + 1, dtype=float)
    pmf[0] = 1.0
    for probability in probabilities:
        previous = pmf.copy()
        pmf *= 1 - probability
        pmf[1:] += previous[:-1] * probability
    return pmf
