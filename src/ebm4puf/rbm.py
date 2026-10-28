import warnings
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.covariance import GraphicalLasso, LedoitWolf
from sklearn.exceptions import ConvergenceWarning
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm


class PyTorchRBM(nn.Module):
    def __init__(self, n_visible: int, n_hidden: int):
        super(PyTorchRBM, self).__init__()
        # Standard initialization for RBM weights
        self.W = nn.Parameter(torch.randn(n_visible, n_hidden) * 0.01)
        self.v_bias = nn.Parameter(torch.zeros(n_visible))
        self.h_bias = nn.Parameter(torch.zeros(n_hidden))

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

    def free_energy(self, v: torch.Tensor):
        """
        Computes the Free Energy of a visible state.

        .. math::

            F(v) = -v^T v_{bias} - \sum(\log(1 + \exp(v W + h_{bias})))

        Softplus is used for numerical stability of log(1 + exp(x))
        """
        v_term = torch.matmul(v, self.v_bias)
        h_term = torch.sum(
            nn.functional.softplus(torch.matmul(v, self.W) + self.h_bias), dim=1
        )
        return -(v_term + h_term)


class RBMPUFModel(BaseEstimator, TransformerMixin):
    def __init__(
        self,
        n_hidden: int = 64,
        k: int = 1,
        learning_rate: float = 0.01,
        epochs: int = 100,
        batch_size: int = 32,
        device: str = "cpu",
    ):
        """Sklearn-wrapper for the PyTorch RBM.

        Args:
            n_hidden: Number of hidden units (latent spins).
            k: Number of Gibbs sampling steps for CD-k.
            learning_rate: Optimizer learning rate.
            epochs: Training epochs.
            batch_size: Batch size for DataLoader.
            device: 'cpu' or 'cuda'.
        """
        self.n_hidden = n_hidden
        self.k = k
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.batch_size = batch_size
        self.device = torch.device(device)
        self.model = None

    def fit(self, X: np.ndarray, y=None):
        """
        Trains the RBM using Contrastive Divergence (CD-k).
        X should be binary data {0, 1} of shape (n_samples, n_visible).
        """
        n_samples, n_visible = X.shape
        self.n_visible_ = n_visible
        self.model = PyTorchRBM(n_visible, self.n_hidden).to(self.device)

        optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=self.epochs
        )

        tensor_x = torch.tensor(X, dtype=torch.float32)
        dataset = TensorDataset(tensor_x)
        loader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True)

        self.model.train()
        pbar = tqdm(range(self.epochs), desc="Fitting RBM")
        for epoch in pbar:
            epoch_loss = 0.0

            for batch in loader:
                v0 = batch[0].to(self.device)

                # Gibbs Sampling Phase (CD-k)
                vk = v0.clone()
                for _ in range(self.k):
                    _, hk = self.model.sample_h(vk)
                    _, vk = self.model.sample_v(hk)

                # Stop gradients from the sampling phase
                vk = vk.detach()

                # The Pseudo-Loss trick: Minimizing this minimizes CD-k
                loss = torch.mean(self.model.free_energy(v0)) - torch.mean(
                    self.model.free_energy(vk)
                )

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                epoch_loss += loss.item()
            scheduler.step()

            pseudo_loss = epoch_loss / len(loader)
            pbar.set_postfix({"pseudo-loss": pseudo_loss})

        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        """
        Maps the visible PUF responses to the latent hidden space.
        Returns the probabilities P(h=1|v).
        """
        self.model.eval()
        with torch.no_grad():
            v = torch.tensor(X, dtype=torch.float32).to(self.device)
            h_prob, _ = self.model.sample_h(v)
        return h_prob.cpu().numpy()

    def generate(
        self,
        X: Optional[np.ndarray] = None,
        *,
        n_samples: int = 1,
        gibbs_steps: int = 1,
        from_hidden: bool = False,
        seed: Optional[int] = None,
        return_probabilities: bool = False,
    ) -> np.ndarray:
        """
        Generate binary visible samples from the learned RBM.

        If ``X`` is provided and ``from_hidden=False``, rows are visible seeds and
        the method runs a Gibbs chain from them. If ``from_hidden=True``, ``X`` is
        interpreted as hidden states/probabilities and decoded to visible samples.

        If ``X`` is omitted, hidden states are sampled from Bernoulli(0.5).
        """
        self._check_fitted()
        if seed is not None:
            torch.manual_seed(seed)

        self.model.eval()
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
                    _, h_sample = self.model.sample_h(v_sample)
                    v_prob, v_sample = self.model.sample_v(h_sample)
            elif from_hidden:
                h = torch.tensor(X, dtype=torch.float32, device=self.device)
                v_prob, v_sample = self.model.sample_v(h)
            else:
                v_sample = torch.tensor(X, dtype=torch.float32, device=self.device)
                v_prob = v_sample
                for _ in range(gibbs_steps):
                    _, h_sample = self.model.sample_h(v_sample)
                    v_prob, v_sample = self.model.sample_v(h_sample)

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
        self._check_fitted()

        return {
            "W": self.model.W.detach().cpu().numpy(),
            "v_bias": self.model.v_bias.detach().cpu().numpy(),
            "h_bias": self.model.h_bias.detach().cpu().numpy(),
        }

    def generate_precision_matrix(
        self,
        *,
        n_samples: int = 10_000,
        gibbs_steps: int = 10,
        alpha: Optional[float] = None,
        seed: Optional[int] = None,
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
            except Exception:
                estimator = LedoitWolf().fit(samples)

        self.generated_samples_ = samples
        self.generated_covariance_ = estimator.covariance_
        self.generated_precision_ = estimator.precision_
        return self.generated_precision_

    def _check_fitted(self):
        if self.model is None:
            raise ValueError("Model is not fitted yet.")
