import numpy as np
import pytest

from ebm4puf.probability import poisson_binomial_pmf


def test_poisson_binomial_pmf_handles_deterministic_probabilities():
    np.testing.assert_allclose(poisson_binomial_pmf(np.array([0, 1])), [0, 1, 0])


def test_poisson_binomial_pmf_rejects_invalid_probabilities():
    with pytest.raises(ValueError, match=r"probabilities must lie in \[0, 1\]"):
        poisson_binomial_pmf(np.array([1.1]))
