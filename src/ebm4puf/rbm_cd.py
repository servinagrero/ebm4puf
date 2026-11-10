"""Contrastive-divergence training for binary PUF RBMs."""

from typing import Literal

import numpy as np
import torch
from torch import optim
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm

from ebm4puf.rbm import BaseRBMPUFModel


class RBMPUFModel(BaseRBMPUFModel):
    """Binary RBM trained using CD-k or persistent CD."""

    def __init__(
        self,
        n_hidden: int = 64,
        k: int = 1,
        learning_rate: float = 0.01,
        epochs: int = 100,
        batch_size: int = 32,
        device: str = "cpu",
        random_state: int | None = None,
        verbose: bool = True,
        negative_phase: Literal["cd", "pcd"] = "cd",
        visible_bias_init: Literal["zeros", "marginal"] = "zeros",
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
        self.k = k
        self.negative_phase = negative_phase
        self.visible_bias_init = visible_bias_init

    def fit(self, X: np.ndarray, y=None):
        """Train the RBM using CD-k or persistent CD."""
        X = self._initialize_model(X)
        if self.k < 1:
            raise ValueError("k must be positive")
        if self.negative_phase not in {"cd", "pcd"}:
            raise ValueError("negative_phase must be 'cd' or 'pcd'")
        if self.visible_bias_init not in {"zeros", "marginal"}:
            raise ValueError("visible_bias_init must be 'zeros' or 'marginal'")

        n_samples = len(X)
        model = self._check_fitted()
        marginal = torch.tensor(
            X.mean(axis=0),
            dtype=torch.float32,
            device=self.device,
        )
        if self.visible_bias_init == "marginal":
            clipped = marginal.clamp(1e-4, 1 - 1e-4)
            with torch.no_grad():
                model.v_bias.copy_(torch.logit(clipped))

        optimizer = optim.Adam(model.parameters(), lr=self.learning_rate)
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
        persistent_visible = None
        if self.negative_phase == "pcd":
            persistent_visible = torch.bernoulli(
                marginal.expand(min(self.batch_size, n_samples), -1)
            )
        self.training_loss_ = []
        progress = tqdm(
            range(self.epochs),
            desc="Fitting RBM",
            disable=not self.verbose,
        )
        for _ in progress:
            epoch_loss = 0.0
            for (batch,) in loader:
                v0 = batch.to(self.device)
                vk = (
                    v0.clone()
                    if persistent_visible is None
                    else persistent_visible[: len(v0)].clone()
                )
                for _ in range(self.k):
                    _, hidden = model.sample_h(vk)
                    _, vk = model.sample_v(hidden)

                vk = vk.detach()
                if persistent_visible is not None:
                    persistent_visible[: len(v0)] = vk
                loss = model.free_energy(v0).mean() - model.free_energy(vk).mean()

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item()

            scheduler.step()
            mean_loss = epoch_loss / len(loader)
            self.training_loss_.append(mean_loss)
            progress.set_postfix({"pseudo-loss": mean_loss})

        return self
