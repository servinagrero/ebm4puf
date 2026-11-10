"""Evaluate per-device RBMs on PUF bit-flip vectors.

The script deliberately keeps enrollment, calibration, and test readouts separate.
Free-energy sequence thresholds are calibrated on genuine calibration readouts and
reported on held-out genuine and synthetic/tampered scenarios.
"""

import argparse
import csv
import logging
from pathlib import Path

import h5py
import numpy as np
import torch

from ebm4puf.artifact import error_matrix, read_family, selected_family_names
from ebm4puf.authentication import (
    center_evidence,
    evaluate_authentication,
    evaluate_joint_score_tests,
    evaluate_score_tests,
    joint_score_statistics_rows,
    score_statistics_rows,
)
from ebm4puf.diagnostics import relaxation_metrics
from ebm4puf.fidelity import fidelity_metrics
from ebm4puf.probability import poisson_binomial_pmf
from ebm4puf.rbm_cd import RBMPUFModel
from ebm4puf.sampling import make_static_scenarios, select_devices, split_error_matrix

LOGGER = logging.getLogger("ebm4puf.free_energy")


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def parse_training_config(value: str) -> tuple[str, int]:
    """Parse a training configuration such as ``cd:5`` or ``pcd:1``."""
    try:
        method, steps = value.lower().split(":", maxsplit=1)
        steps = int(steps)
    except ValueError as error:
        raise argparse.ArgumentTypeError(
            "training configurations must use METHOD:K"
        ) from error
    if method not in {"cd", "pcd"} or steps < 1:
        raise argparse.ArgumentTypeError(
            "METHOD must be cd or pcd and K must be positive"
        )
    return method, steps


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifact", type=Path, default=Path("data/puf_artifact.h5"))
    parser.add_argument("--output-dir", type=Path, default=Path("data/rbm_free_energy"))
    parser.add_argument("--families", nargs="+")
    parser.add_argument("--include-non-room-temperature", action="store_true")
    parser.add_argument("--devices-per-family", type=int, default=10)
    parser.add_argument("--minimum-readouts", type=int, default=40)
    parser.add_argument("--train-fraction", type=float, default=0.6)
    parser.add_argument("--calibration-fraction", type=float, default=0.2)
    parser.add_argument("--n-hidden", type=int, default=64)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--learning-rate", type=float, default=0.01)
    parser.add_argument(
        "--training-configs",
        type=parse_training_config,
        nargs="+",
        default=[("cd", 1), ("cd", 5), ("cd", 10), ("pcd", 1)],
    )
    parser.add_argument(
        "--visible-bias-init", choices=["zeros", "marginal"], default="marginal"
    )
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--generated-samples", type=int, default=1_000)
    parser.add_argument("--gibbs-steps", type=int, nargs="+", default=[1, 10, 50, 100])
    parser.add_argument("--relaxation-steps", type=int, nargs="+", default=[1, 5, 10])
    parser.add_argument("--diagnostic-samples", type=int, default=100)
    parser.add_argument("--group-sizes", type=int, nargs="+", default=[1, 2, 4, 8])
    parser.add_argument("--score-repeats", type=int, default=1_000)
    parser.add_argument("--false-reject-rate", type=float, default=0.05)
    parser.add_argument(
        "--target-aucs",
        type=float,
        nargs="+",
        default=[0.8, 0.9, 0.95, 0.99],
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--verbose-training", action="store_true")
    parser.add_argument("--log-level", default="INFO")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    logging.basicConfig(
        level=getattr(logging, args.log_level.upper()),
        format="%(levelname)s %(name)s: %(message)s",
    )
    if not 0 < args.false_reject_rate < 1:
        raise ValueError("false-reject-rate must lie strictly between 0 and 1")
    args.target_aucs = sorted(set(args.target_aucs))
    if not args.target_aucs or args.target_aucs[0] <= 0.5 or args.target_aucs[-1] >= 1:
        raise ValueError("target-aucs must lie strictly between 0.5 and 1")
    args.training_configs = list(dict.fromkeys(args.training_configs))
    for name in ("gibbs_steps", "relaxation_steps", "group_sizes"):
        values = sorted(set(getattr(args, name)))
        if not values or values[0] < 1:
            raise ValueError(f"{name.replace('_', '-')} must contain positive values")
        setattr(args, name, values)
    if args.diagnostic_samples < 1:
        raise ValueError("diagnostic-samples must be positive")
    families = selected_family_names(
        args.artifact,
        args.families,
        include_non_room_temperature=args.include_non_room_temperature,
    )
    if not families:
        raise SystemExit("No room-temperature families selected")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    fidelity_rows: list[dict] = []
    authentication_rows: list[dict] = []
    generation_rows: list[dict] = []
    relaxation_rows: list[dict] = []
    error_count_rows: list[dict] = []
    score_test_rows: list[dict] = []
    score_statistic_rows: list[dict] = []
    joint_score_test_rows: list[dict] = []
    joint_score_statistic_rows: list[dict] = []
    required_readout_rows: list[dict] = []
    with h5py.File(args.artifact, "r") as h5:
        for family_index, family in enumerate(families):
            responses, references = read_family(h5, family)
            n_devices, n_readouts, n_bits = responses.shape
            inference_ready = n_readouts >= args.minimum_readouts
            if inference_ready:
                LOGGER.info(
                    "%s: %d devices, %d readouts, %d bits",
                    family,
                    n_devices,
                    n_readouts,
                    n_bits,
                )
            else:
                LOGGER.warning(
                    "%s has only %d readouts/device (< %d); per-device RBM "
                    "results are underpowered and exploratory",
                    family,
                    n_readouts,
                    args.minimum_readouts,
                )
            devices = select_devices(
                n_devices,
                args.devices_per_family,
                args.seed + family_index,
            )
            family_dir = args.output_dir / family
            family_dir.mkdir(parents=True, exist_ok=True)
            error_count_pmfs: dict[tuple[str, str, int, int, str], np.ndarray] = {
                ("empirical", "none", 0, 0, "none"): np.zeros(n_bits + 1, dtype=float),
                ("independent", "none", 0, 0, "none"): np.zeros(
                    n_bits + 1, dtype=float
                ),
            }
            for training_method, cd_k in args.training_configs:
                for gibbs_steps in args.gibbs_steps:
                    for initialization in ("empirical", "marginal", "uniform"):
                        error_count_pmfs[
                            ("rbm", training_method, cd_k, gibbs_steps, initialization)
                        ] = np.zeros(n_bits + 1, dtype=float)
            for device_index in devices:
                run_seed = args.seed + family_index * 100_000 + int(device_index)
                errors = error_matrix(responses[device_index], references[device_index])
                train, calibration, test = split_error_matrix(
                    errors,
                    train_fraction=args.train_fraction,
                    calibration_fraction=args.calibration_fraction,
                    seed=run_seed,
                )
                impostor_errors = None
                if n_devices > 1:
                    impostor_device = (int(device_index) + 1) % n_devices
                    impostor_errors = error_matrix(
                        responses[impostor_device], references[device_index]
                    )
                static_scenarios = make_static_scenarios(
                    train=train,
                    test=test,
                    impostor_errors=impostor_errors,
                    n_samples=args.generated_samples,
                    seed=run_seed,
                )
                empirical_counts = np.bincount(errors.sum(axis=1), minlength=n_bits + 1)
                error_count_pmfs[("empirical", "none", 0, 0, "none")] += (
                    empirical_counts / len(errors)
                )
                error_count_pmfs[("independent", "none", 0, 0, "none")] += (
                    poisson_binomial_pmf(errors.mean(axis=0))
                )
                marginal_fidelity = fidelity_metrics(
                    test, static_scenarios["marginal"], source="marginal"
                )
                marginal_fidelity.update(
                    {
                        "family": family,
                        "device": int(device_index),
                        "training_method": "none",
                        "cd_k": 0,
                        "gibbs_steps": 0,
                        "visible_bias_init": "none",
                        "initialization": "marginal",
                        "n_readouts": n_readouts,
                        "n_train": len(train),
                        "n_calibration": len(calibration),
                        "n_test": len(test),
                        "inference_ready": inference_ready,
                    }
                )
                fidelity_rows.append(marginal_fidelity)

                generation_rng = np.random.default_rng(run_seed)
                marginals = train.mean(axis=0)
                generation_seeds = {
                    "empirical": train[
                        generation_rng.integers(
                            0, len(train), size=args.generated_samples
                        )
                    ],
                    "marginal": generation_rng.binomial(
                        1, marginals, size=(args.generated_samples, n_bits)
                    ).astype(np.uint8),
                    "uniform": generation_rng.binomial(
                        1, 0.5, size=(args.generated_samples, n_bits)
                    ).astype(np.uint8),
                }
                common_metadata = {
                    "family": family,
                    "device": int(device_index),
                    "n_readouts": n_readouts,
                    "n_train": len(train),
                    "n_calibration": len(calibration),
                    "n_test": len(test),
                    "inference_ready": inference_ready,
                }
                for training_method, cd_k in args.training_configs:
                    LOGGER.info(
                        "Training %s device %d with %s-%d",
                        family,
                        device_index,
                        training_method.upper(),
                        cd_k,
                    )
                    model = RBMPUFModel(
                        n_hidden=args.n_hidden,
                        k=cd_k,
                        learning_rate=args.learning_rate,
                        epochs=args.epochs,
                        batch_size=args.batch_size,
                        device=args.device,
                        random_state=run_seed,
                        verbose=args.verbose_training,
                        negative_phase=training_method,
                        visible_bias_init=args.visible_bias_init,
                    ).fit(train)
                    static_authentication = evaluate_authentication(
                        model=model,
                        calibration=calibration,
                        test=test,
                        scenarios=static_scenarios,
                        marginals=marginals,
                        group_sizes=args.group_sizes,
                        repeats=args.score_repeats,
                        false_reject_rate=args.false_reject_rate,
                        seed=run_seed,
                    )
                    marginal_baselines = center_evidence(static_authentication)
                    for row in static_authentication:
                        row.update(common_metadata)
                        row.update(
                            {
                                "training_method": training_method,
                                "cd_k": cd_k,
                                "gibbs_steps": 0,
                                "visible_bias_init": args.visible_bias_init,
                                "initialization": "none",
                            }
                        )
                        authentication_rows.append(row)

                    model_metadata = {
                        **common_metadata,
                        "training_method": training_method,
                        "cd_k": cd_k,
                        "visible_bias_init": args.visible_bias_init,
                    }
                    static_statistics = score_statistics_rows(
                        model=model,
                        populations={
                            "training": train,
                            "calibration": calibration,
                            "genuine": test,
                            **static_scenarios,
                        },
                        marginals=marginals,
                    )
                    for row in static_statistics:
                        row.update(model_metadata)
                        row.update(
                            {
                                "gibbs_steps": 0,
                                "initialization": "none",
                            }
                        )
                        score_statistic_rows.append(row)

                    static_joint_statistics = joint_score_statistics_rows(
                        model=model,
                        populations={
                            "training": train,
                            "calibration": calibration,
                            "genuine": test,
                            **static_scenarios,
                        },
                        marginals=marginals,
                    )
                    for row in static_joint_statistics:
                        row.update(model_metadata)
                        row.update(
                            {
                                "gibbs_steps": 0,
                                "initialization": "none",
                            }
                        )
                        joint_score_statistic_rows.append(row)

                    static_tests, static_required = evaluate_score_tests(
                        model=model,
                        genuine=test,
                        scenarios=static_scenarios,
                        marginals=marginals,
                        group_sizes=args.group_sizes,
                        repeats=args.score_repeats,
                        target_aucs=args.target_aucs,
                        seed=run_seed,
                    )
                    for row in static_tests:
                        row.update(model_metadata)
                        row.update(
                            {
                                "gibbs_steps": 0,
                                "initialization": "none",
                            }
                        )
                        score_test_rows.append(row)
                    for row in static_required:
                        row.update(model_metadata)
                        row.update(
                            {
                                "gibbs_steps": 0,
                                "initialization": "none",
                            }
                        )
                        required_readout_rows.append(row)

                    static_joint_tests = evaluate_joint_score_tests(
                        model=model,
                        calibration=calibration,
                        genuine=test,
                        scenarios=static_scenarios,
                        marginals=marginals,
                        group_sizes=args.group_sizes,
                        repeats=args.score_repeats,
                        false_reject_rate=args.false_reject_rate,
                        seed=run_seed,
                    )
                    for row in static_joint_tests:
                        row.update(model_metadata)
                        row.update(
                            {
                                "gibbs_steps": 0,
                                "initialization": "none",
                            }
                        )
                        joint_score_test_rows.append(row)

                    generated_by_initialization = {}
                    for initialization, seeds in generation_seeds.items():
                        generated_chain = model.gibbs_chain(
                            seeds,
                            checkpoints=args.gibbs_steps,
                            seed=run_seed,
                        )
                        generated_by_step = {
                            step: samples.astype(np.uint8)
                            for step, (_, samples) in generated_chain.items()
                        }
                        generated_by_initialization[initialization] = generated_by_step
                        for gibbs_steps, generated in generated_by_step.items():
                            rbm_counts = np.bincount(
                                generated.sum(axis=1).astype(int),
                                minlength=n_bits + 1,
                            )
                            error_count_pmfs[
                                (
                                    "rbm",
                                    training_method,
                                    cd_k,
                                    gibbs_steps,
                                    initialization,
                                )
                            ] += rbm_counts / len(generated)
                            rbm_fidelity = fidelity_metrics(
                                test, generated, source="rbm"
                            )
                            rbm_fidelity.update(common_metadata)
                            rbm_fidelity.update(
                                {
                                    "training_method": training_method,
                                    "cd_k": cd_k,
                                    "gibbs_steps": gibbs_steps,
                                    "visible_bias_init": args.visible_bias_init,
                                    "initialization": initialization,
                                }
                            )
                            fidelity_rows.append(rbm_fidelity)

                            generated_authentication = evaluate_authentication(
                                model=model,
                                calibration=calibration,
                                test=test,
                                scenarios={"rbm": generated},
                                marginals=marginals,
                                group_sizes=args.group_sizes,
                                repeats=args.score_repeats,
                                false_reject_rate=args.false_reject_rate,
                                seed=run_seed,
                            )
                            generated_rows = [
                                row
                                for row in generated_authentication
                                if row["scenario"] == "rbm"
                            ]
                            center_evidence(generated_rows, marginal_baselines)
                            for row in generated_rows:
                                row.update(common_metadata)
                                row.update(
                                    {
                                        "training_method": training_method,
                                        "cd_k": cd_k,
                                        "gibbs_steps": gibbs_steps,
                                        "visible_bias_init": args.visible_bias_init,
                                        "initialization": initialization,
                                        "normalized_rbm_score_q25": float(
                                            row["rbm_score_q25"]
                                        )
                                        / int(row["group_size"]),
                                        "normalized_rbm_score_q50": float(
                                            row["rbm_score_q50"]
                                        )
                                        / int(row["group_size"]),
                                        "normalized_rbm_score_q75": float(
                                            row["rbm_score_q75"]
                                        )
                                        / int(row["group_size"]),
                                    }
                                )
                                for metric, value in rbm_fidelity.items():
                                    if metric not in row:
                                        row[metric] = value
                                generation_rows.append(row)

                            generated_statistics = score_statistics_rows(
                                model=model,
                                populations={"rbm": generated},
                                marginals=marginals,
                            )
                            for row in generated_statistics:
                                row.update(model_metadata)
                                row.update(
                                    {
                                        "gibbs_steps": gibbs_steps,
                                        "initialization": initialization,
                                    }
                                )
                                score_statistic_rows.append(row)

                            generated_joint_statistics = joint_score_statistics_rows(
                                model=model,
                                populations={"rbm": generated},
                                marginals=marginals,
                            )
                            for row in generated_joint_statistics:
                                row.update(model_metadata)
                                row.update(
                                    {
                                        "gibbs_steps": gibbs_steps,
                                        "initialization": initialization,
                                    }
                                )
                                joint_score_statistic_rows.append(row)

                            generated_tests, generated_required = evaluate_score_tests(
                                model=model,
                                genuine=test,
                                scenarios={"rbm": generated},
                                marginals=marginals,
                                group_sizes=args.group_sizes,
                                repeats=args.score_repeats,
                                target_aucs=args.target_aucs,
                                seed=run_seed,
                            )
                            for row in generated_tests:
                                row.update(model_metadata)
                                row.update(
                                    {
                                        "gibbs_steps": gibbs_steps,
                                        "initialization": initialization,
                                    }
                                )
                                score_test_rows.append(row)
                            for row in generated_required:
                                row.update(model_metadata)
                                row.update(
                                    {
                                        "gibbs_steps": gibbs_steps,
                                        "initialization": initialization,
                                    }
                                )
                                required_readout_rows.append(row)

                            generated_joint_tests = evaluate_joint_score_tests(
                                model=model,
                                calibration=calibration,
                                genuine=test,
                                scenarios={"rbm": generated},
                                marginals=marginals,
                                group_sizes=args.group_sizes,
                                repeats=args.score_repeats,
                                false_reject_rate=args.false_reject_rate,
                                seed=run_seed,
                            )
                            for row in generated_joint_tests:
                                row.update(model_metadata)
                                row.update(
                                    {
                                        "gibbs_steps": gibbs_steps,
                                        "initialization": initialization,
                                    }
                                )
                                joint_score_test_rows.append(row)

                    relaxation_scenarios = {
                        "genuine": test,
                        **static_scenarios,
                        "rbm": generated_by_initialization["empirical"][
                            args.gibbs_steps[-1]
                        ],
                    }
                    device_relaxation = relaxation_metrics(
                        model=model,
                        scenarios=relaxation_scenarios,
                        checkpoints=args.relaxation_steps,
                        max_samples=args.diagnostic_samples,
                        seed=run_seed,
                        source_gibbs_steps=args.gibbs_steps[-1],
                    )
                    for row in device_relaxation:
                        row.update(common_metadata)
                        row.update(
                            {
                                "training_method": training_method,
                                "cd_k": cd_k,
                                "visible_bias_init": args.visible_bias_init,
                                "initialization": (
                                    "empirical" if row["scenario"] == "rbm" else "none"
                                ),
                            }
                        )
                        relaxation_rows.append(row)
                    torch.save(
                        model.model.state_dict(),
                        family_dir
                        / f"device_{device_index}_{training_method}_k{cd_k}.pt",
                    )
            for (
                source,
                training_method,
                cd_k,
                gibbs_steps,
                initialization,
            ), probability in error_count_pmfs.items():
                probability /= len(devices)
                n_vectors = 0
                if source == "empirical":
                    n_vectors = len(devices) * n_readouts
                elif source == "rbm":
                    n_vectors = len(devices) * args.generated_samples
                error_count_rows.extend(
                    {
                        "family": family,
                        "source": source,
                        "training_method": training_method,
                        "cd_k": cd_k,
                        "gibbs_steps": gibbs_steps,
                        "visible_bias_init": (
                            args.visible_bias_init if source == "rbm" else "none"
                        ),
                        "initialization": initialization,
                        "error_count": error_count,
                        "probability": float(value),
                        "n_devices": len(devices),
                        "n_vectors": n_vectors,
                    }
                    for error_count, value in enumerate(probability)
                )

    write_csv(args.output_dir / "fidelity.csv", fidelity_rows)
    write_csv(args.output_dir / "authentication.csv", authentication_rows)
    write_csv(args.output_dir / "generation_sweep.csv", generation_rows)
    write_csv(args.output_dir / "relaxation.csv", relaxation_rows)
    write_csv(args.output_dir / "error_count_distributions.csv", error_count_rows)
    write_csv(args.output_dir / "score_tests.csv", score_test_rows)
    write_csv(args.output_dir / "score_statistics.csv", score_statistic_rows)
    write_csv(args.output_dir / "joint_score_tests.csv", joint_score_test_rows)
    write_csv(
        args.output_dir / "joint_score_statistics.csv",
        joint_score_statistic_rows,
    )
    write_csv(
        args.output_dir / "required_readouts.csv",
        required_readout_rows,
    )
    LOGGER.info("Wrote results to %s", args.output_dir)


if __name__ == "__main__":
    main()
