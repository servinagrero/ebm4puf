"""Density-ratio residual RBM for PUF bit-flip vectors."""

import numpy as np
import torch
from torch import optim
from torch.nn import functional
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm

from ebm4puf.rbm import BaseRBMPUFModel


class ResidualRBMPUFModel(BaseRBMPUFModel):
    """Learn deviations from a matched-marginal Bernoulli base model."""

    def __init__(
        self,
        n_hidden: int = 64,
        learning_rate: float = 0.01,
        epochs: int = 100,
        batch_size: int = 32,
        device: str = "cpu",
        random_state: int | None = None,
        verbose: bool = True,
        marginal_smoothing: float = 0.5,
    ):
        super().__init__(
            n_hidden=n_hidden,
            learning_rate=learning_rate,
            epochs=epochs,
            batch_size=batch_size,
            device=device,
            random_state=random_state,
            verbose=verbose,
        )
        self.marginal_smoothing = marginal_smoothing

    def fit(self, X: np.ndarray, y=None):
        """Fit a logistic density ratio against fresh matched-marginal noise."""
        X = self._initialize_model(X)
        if self.marginal_smoothing <= 0:
            raise ValueError("marginal_smoothing must be positive")

        n_samples = len(X)
        model = self._check_fitted()
        self.marginals_ = (X.sum(axis=0) + self.marginal_smoothing) / (
            n_samples + 2 * self.marginal_smoothing
        )
        marginal = torch.tensor(
            self.marginals_,
            dtype=torch.float32,
            device=self.device,
        )
        with torch.no_grad():
            model.v_bias.copy_(torch.logit(marginal))
        model.v_bias.requires_grad_(False)

        optimizer = optim.Adam(
            (parameter for parameter in model.parameters() if parameter.requires_grad),
            lr=self.learning_rate,
        )
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=self.epochs,
        )
        generator = torch.Generator()
        if self.random_state is not None:
            generator.manual_seed(self.random_state)
        loader = DataLoader(
            TensorDataset(torch.tensor(X, dtype=torch.float32)),
            batch_size=min(self.batch_size, n_samples),
            shuffle=True,
            generator=generator,
        )

        model.train()
        self.training_loss_ = []
        progress = tqdm(
            range(self.epochs),
            desc="Fitting residual RBM",
            disable=not self.verbose,
        )
        for _ in progress:
            epoch_loss = 0.0
            for (batch,) in loader:
                genuine = batch.to(self.device)
                independent = torch.bernoulli(marginal.expand(len(genuine), -1))
                genuine_logits = model.interaction_score(genuine)
                independent_logits = model.interaction_score(independent)
                loss = functional.binary_cross_entropy_with_logits(
                    genuine_logits,
                    torch.ones_like(genuine_logits),
                ) + functional.binary_cross_entropy_with_logits(
                    independent_logits,
                    torch.zeros_like(independent_logits),
                )

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item()

            scheduler.step()
            mean_loss = epoch_loss / len(loader)
            self.training_loss_.append(mean_loss)
            progress.set_postfix({"logistic-loss": mean_loss})

        return self

    def base_score_samples(self, X: np.ndarray) -> np.ndarray:
        """Return log probability under the smoothed Bernoulli base model."""
        self._check_fitted()
        X = self._validate_binary_matrix(X, n_visible=self.n_visible_)
        probabilities = np.clip(self.marginals_, 1e-6, 1 - 1e-6)
        return np.sum(
            X * np.log(probabilities) + (1 - X) * np.log1p(-probabilities),
            axis=1,
        )

    def residual_score_samples(self, X: np.ndarray) -> np.ndarray:
        """Return the learned log-density-ratio score."""
        model = self._check_fitted()
        X = self._validate_binary_matrix(X, n_visible=self.n_visible_)
        model.eval()
        with torch.no_grad():
            visible = torch.tensor(X, dtype=torch.float32, device=self.device)
            scores = model.interaction_score(visible)
        return scores.cpu().numpy()

    def score_samples(self, X: np.ndarray) -> np.ndarray:
        """Return base log probability plus learned residual score."""
        return self.base_score_samples(X) + self.residual_score_samples(X)

    def free_energy(self, X: np.ndarray) -> np.ndarray:
        """Return the negative combined score."""
        return -self.score_samples(X)

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
        """Generate from the combined base-plus-residual energy."""
        if X is None and not from_hidden:
            self._check_fitted()
            X = (
                np.random.default_rng(seed)
                .binomial(
                    1,
                    self.marginals_,
                    size=(n_samples, self.n_visible_),
                )
                .astype(np.uint8)
            )
        return super().generate(
            X,
            n_samples=n_samples,
            gibbs_steps=gibbs_steps,
            from_hidden=from_hidden,
            seed=seed,
            return_probabilities=return_probabilities,
        )
