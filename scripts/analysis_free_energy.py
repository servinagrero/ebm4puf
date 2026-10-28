import argparse
from pathlib import Path

import numpy as np
import torch

from pufre import DATA_DIR, logger
from pufre.datasets import load_hdf5_artifact
from pufre.rbm import RBMPUFModel


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--artifact", type=Path, default=DATA_DIR / "puf_artifact.h5")
    parser.add_argument("--study-name", default="rbm")
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--n-hidden", type=int, default=64)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--learning-rate", type=float, default=0.01)
    parser.add_argument("--k", type=int, default=1)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--generated-samples", type=int, default=10_000)
    parser.add_argument("--gibbs-steps", type=int, default=10)
    parser.add_argument("--precision-alpha", type=float, default=None)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    output_dir = args.output_dir or (DATA_DIR / args.study_name)
    families = load_hdf5_artifact(args.artifact)
    if not families:
        raise SystemExit(f"No families found in {args.artifact}")

    for family in families:
        logger.info("Running RBM study %s", family.family)
        data = family.to_puf_data()
        family_dir = output_dir / family.family
        family_dir.mkdir(parents=True, exist_ok=True)

        model = RBMPUFModel(
            n_hidden=args.n_hidden,
            k=args.k,
            learning_rate=args.learning_rate,
            epochs=args.epochs,
            batch_size=args.batch_size,
            device=args.device,
        )
        model.fit(data.X_bin_flat)
        precision = model.generate_precision_matrix(
            n_samples=args.generated_samples,
            gibbs_steps=args.gibbs_steps,
            alpha=args.precision_alpha,
            seed=args.seed,
        )
        precision_offdiag = precision.copy()
        np.fill_diagonal(precision_offdiag, 0.0)
        precision_strength = np.sum(np.abs(precision_offdiag), axis=0)

        torch.save(model.model.state_dict(), family_dir / "rbm.pt")
        np.save(family_dir / "generated_precision.npy", precision)
        np.savez_compressed(
            family_dir / "rbm_graph_summary.npz",
            bit_id=data.bit_ids,
            physical_bit_id=data.metadata.get("selected_bit_indices", data.bit_ids),
            precision_strength=precision_strength,
            precision_diagonal=np.diag(precision),
            generated_mean=np.mean(model.generated_samples_, axis=0),
            generated_variance=np.var(model.generated_samples_, axis=0),
        )


if __name__ == "__main__":
    main()
