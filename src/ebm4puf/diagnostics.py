"""Diagnostics for fitted PUF energy models."""

from typing import Protocol

import numpy as np


class GibbsSampler(Protocol):
    """Energy model interface required by transition diagnostics."""

    def gibbs_chain(
        self,
        values: np.ndarray,
        *,
        checkpoints: list[int],
        seed: int,
    ) -> dict[int, tuple[np.ndarray, np.ndarray]]: ...


def relaxation_metrics(
    *,
    model: GibbsSampler,
    scenarios: dict[str, np.ndarray],
    checkpoints: list[int],
    max_samples: int,
    seed: int,
    source_gibbs_steps: int,
) -> list[dict[str, float | int | str]]:
    """Measure local Gibbs transitions between states k - 1 and k."""
    checkpoints = sorted(set(checkpoints))
    if not checkpoints or checkpoints[0] < 1:
        raise ValueError("checkpoints must contain positive step counts")
    chain_checkpoints = sorted(
        {
            chain_step
            for checkpoint in checkpoints
            for chain_step in (checkpoint - 1, checkpoint)
            if chain_step > 0
        }
    )
    rng = np.random.default_rng(seed)
    rows = []
    for scenario, values in scenarios.items():
        if len(values) > max_samples:
            values = values[rng.choice(len(values), max_samples, replace=False)]
        chain = model.gibbs_chain(
            values,
            checkpoints=chain_checkpoints,
            seed=seed,
        )
        for step in checkpoints:
            previous = values if step == 1 else chain[step - 1][1]
            current = chain[step][1]
            rows.append(
                {
                    "scenario": scenario,
                    "relaxation_steps": step,
                    "source_gibbs_steps": (
                        source_gibbs_steps if scenario == "rbm" else 0
                    ),
                    "n_vectors": len(values),
                    "mean_transition_hamming_fraction": float(
                        np.mean(previous != current)
                    ),
                    "unchanged_transition_fraction": float(
                        np.mean(np.all(previous == current, axis=1))
                    ),
                }
            )
    return rows

