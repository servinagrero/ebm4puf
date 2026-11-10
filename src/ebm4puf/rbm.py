import warnings

import numpy as np
import torch
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.covariance import GraphicalLasso, LedoitWolf
from sklearn.exceptions import ConvergenceWarning
from torch import nn


class PyTorchRBM(nn.Module):
    def __init__(self, n_visible: int, n_hidden: int):
        super().__init__()
        self.W = nn.Parameter(torch.randn(n_visible, n_hidden) * 0.01)
        self.v_bias = nn.Parameter(torch.zeros(n_visible))
        self.h_bias = nn.Parameter(torch.zeros(n_hidden))
        self.score_offset = nn.Parameter(torch.zeros(()))

    def sample_h(self, v: torch.Tensor):
        """P(h=1 | v) and samples the hidden state."""
        h_prob = torch.sigmoid(torch.matmul(v, self.W) + self.h_bias)
        h_sample = torch.bernoulli(h_prob)
        return h_prob, h_sample

    def sample_v(self, h: torch.Tensor):
        """P(v=1 | h) and samples the visible state."""
        v_prob = torch.sigmoid(torch.matmul(h, self.W.t()) + self.v_bias)
        v_sample = torch.bernoulli(v_prob)
        return v_prob, v_sample

    def interaction_score(self, v: torch.Tensor) -> torch.Tensor:
        """Return the hidden-interaction score plus a scalar offset."""
        return (
            torch.sum(
                nn.functional.softplus(torch.matmul(v, self.W) + self.h_bias),
                dim=1,
            )
            + self.score_offset
        )

    def free_energy(self, v: torch.Tensor):
        r"""
        Computes the Free Energy of a visible state.

        .. math::

            F(v) = -v^T v_{bias} - \sum(\log(1 + \exp(v W + h_{bias})))

        Softplus is used for numerical stability of log(1 + exp(x))
        """
        v_term = torch.matmul(v, self.v_bias)
        return -(v_term + self.interaction_score(v))


class BaseRBMPUFModel(BaseEstimator, TransformerMixin):
    def __init__(
        self,
        n_hidden: int = 64,
        learning_rate: float = 0.01,
        epochs: int = 100,
        batch_size: int = 32,
        device: str = "cpu",
        random_state: int | None = None,
        verbose: bool = True,
    ):
        """Common sklearn-style wrapper behavior for binary RBMs."""
        self.n_hidden = n_hidden
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.batch_size = batch_size
        self.device = device
        self.random_state = random_state
        self.verbose = verbose
        self.model = None

    def _initialize_model(self, X: np.ndarray) -> np.ndarray:
        """Validate data and initialize the shared PyTorch energy module."""
        X = self._validate_binary_matrix(X)
        if self.n_hidden < 1 or self.epochs < 1 or self.batch_size < 1:
            raise ValueError("n_hidden, epochs, and batch_size must be positive")
        if self.learning_rate <= 0:
            raise ValueError("learning_rate must be positive")
        if self.random_state is not None:
            torch.manual_seed(self.random_state)

        _, n_visible = X.shape
        self.n_visible_ = n_visible
        self.model = PyTorchRBM(n_visible, self.n_hidden).to(self.device)
        return X

    def transform(self, X: np.ndarray) -> np.ndarray:
        """
        Maps the visible PUF responses to the latent hidden space.
        Returns the probabilities P(h=1|v).
        """
        model = self._check_fitted()
        X = self._validate_binary_matrix(X, n_visible=self.n_visible_)
        model.eval()
        with torch.no_grad():
            v = torch.tensor(X, dtype=torch.float32).to(self.device)
            h_prob, _ = model.sample_h(v)
        return h_prob.cpu().numpy()

    def free_energy(self, X: np.ndarray) -> np.ndarray:
        """Return the RBM free energy for each binary visible vector."""
        model = self._check_fitted()
        X = self._validate_binary_matrix(X, n_visible=self.n_visible_)
        model.eval()
        with torch.no_grad():
            visible = torch.tensor(X, dtype=torch.float32, device=self.device)
            energy = model.free_energy(visible)
        return energy.cpu().numpy()

    def score_samples(self, X: np.ndarray) -> np.ndarray:
        """Return unnormalized log-probability scores ``-free_energy(X)``."""
        return -self.free_energy(X)

    def gibbs_chain(
        self,
        X: np.ndarray,
        *,
        checkpoints: list[int],
        seed: int | None = None,
    ) -> dict[int, tuple[np.ndarray, np.ndarray]]:
        """Run one visible Gibbs chain and retain probabilities/samples at checkpoints."""
        model = self._check_fitted()
        X = self._validate_binary_matrix(X, n_visible=self.n_visible_)
        checkpoints = sorted(set(checkpoints))
        if not checkpoints or checkpoints[0] < 1:
            raise ValueError("checkpoints must contain positive step counts")
        if seed is not None:
            torch.manual_seed(seed)

        retained = {}
        visible = torch.tensor(X, dtype=torch.float32, device=self.device)
        model.eval()
        with torch.no_grad():
            for step in range(1, checkpoints[-1] + 1):
                _, hidden = model.sample_h(visible)
                probabilities, visible = model.sample_v(hidden)
                if step in checkpoints:
                    retained[step] = (
                        probabilities.cpu().numpy().copy(),
                        visible.cpu().numpy().copy(),
                    )
        return retained

    def generate(
        self,
        X: np.ndarray | None = None,
        *,
        n_samples: int = 1,
        gibbs_steps: int = 1,
        from_hidden: bool = False,
        seed: int | None = None,
        return_probabilities: bool = False,
    ) -> np.ndarray:
        """
        Generate binary visible samples from the learned RBM.

        If ``X`` is provided and ``from_hidden=False``, rows are visible seeds and
        the method runs a Gibbs chain from them. If ``from_hidden=True``, ``X`` is
        interpreted as hidden states/probabilities and decoded to visible samples.

        If ``X`` is omitted, hidden states are sampled from Bernoulli(0.5).
        """
        model = self._check_fitted()
        if seed is not None:
            torch.manual_seed(seed)

        model.eval()
        with torch.no_grad():
            if X is None:
                v_sample = torch.bernoulli(
                    torch.full(
                        (n_samples, self.n_visible_),
                        0.5,
                        dtype=torch.float32,
                        device=self.device,
                    )
                )
                v_prob = v_sample
                for _ in range(gibbs_steps):
                    _, h_sample = model.sample_h(v_sample)
                    v_prob, v_sample = model.sample_v(h_sample)
            elif from_hidden:
                h = torch.tensor(X, dtype=torch.float32, device=self.device)
                v_prob, v_sample = model.sample_v(h)
            else:
                v_sample = torch.tensor(X, dtype=torch.float32, device=self.device)
                v_prob = v_sample
                for _ in range(gibbs_steps):
                    _, h_sample = model.sample_h(v_sample)
                    v_prob, v_sample = model.sample_v(h_sample)

        generated = v_prob if return_probabilities else v_sample
        return generated.cpu().numpy()

    def get_hamiltonian_parameters(self):
        r"""
        Extracts the learned weights and biases for Spin Glass analysis.
        Note: If your spin-glass code assumes {-1, 1} instead of {0, 1},
        you will need to apply a linear transformation here.

        The mathematical mapping from a {0,1} RBM to a {−1,1} Ising model is:

        .. math::

            W_{Ising}=\frac{1}{4} W_{RBM}

        .. math::

            v\_bias_{Ising}=\frac{1}{2}v\_bias_{RBM} + \frac{1}{4}\sum_j W_{RBM}^{:,j}
        """
        model = self._check_fitted()

        return {
            "W": model.W.detach().cpu().numpy(),
            "v_bias": model.v_bias.detach().cpu().numpy(),
            "h_bias": model.h_bias.detach().cpu().numpy(),
        }

    def generate_precision_matrix(
        self,
        *,
        n_samples: int = 10_000,
        gibbs_steps: int = 10,
        alpha: float | None = None,
        seed: int | None = None,
    ):
        """Generate synthetic visible samples and estimate a precision matrix.

        The goal is to train the RBM, generatize idealized visible responses, estimate the covariance and finally invert it.

        If ``alpha`` is provided, Graphical Lasso is used to provide a sparse precision matrix; otherwise Ledoit-Wolf shrinkage provides a stable dense precision estimate.
        """
        samples = self.generate(
            n_samples=n_samples,
            gibbs_steps=gibbs_steps,
            seed=seed,
        )

        if alpha is None:
            estimator = LedoitWolf().fit(samples)
        else:
            try:
                with warnings.catch_warnings():
                    warnings.filterwarnings("ignore", category=ConvergenceWarning)
                    estimator = GraphicalLasso(alpha=alpha).fit(samples)
            except Exception:  # noqa: BLE001 - fall back if sparse fitting fails
                estimator = LedoitWolf().fit(samples)

        self.generated_samples_ = samples
        self.generated_covariance_ = estimator.covariance_
        self.generated_precision_ = estimator.precision_
        return self.generated_precision_

    def _check_fitted(self) -> PyTorchRBM:
        if self.model is None:
            raise ValueError("Model is not fitted yet.")
        return self.model

    @staticmethod
    def _validate_binary_matrix(
        X: np.ndarray, *, n_visible: int | None = None
    ) -> np.ndarray:
        X = np.asarray(X)
        if X.ndim != 2 or X.shape[0] == 0 or X.shape[1] == 0:
            raise ValueError("X must be a non-empty two-dimensional array")
        if n_visible is not None and X.shape[1] != n_visible:
            raise ValueError(f"X must contain {n_visible} visible features")
        if not np.all((X == 0) | (X == 1)):
            raise ValueError("X must contain only binary values 0 and 1")
        return X.astype(np.float32, copy=False)
