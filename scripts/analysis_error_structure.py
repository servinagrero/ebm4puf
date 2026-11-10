"""Analyze PUF error counts and error locations without fitting an EBM."""

import argparse
import logging
from pathlib import Path

import h5py
import numpy as np
import polars as pl

from ebm4puf.artifact import read_family, selected_family_names
from ebm4puf.error_structure import (
    analyze_device,
    leave_one_device_out_prior,
    population_error_statistics,
    reference_polarity_statistics,
    summarize_reference_polarity_family,
)

LOGGER = logging.getLogger("ebm4puf.error_structure")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifact", type=Path, default=Path("data/puf_artifact.h5"))
    parser.add_argument("--output-dir", type=Path, default=Path("data/error_structure"))
    parser.add_argument("--families", nargs="+")
    parser.add_argument("--include-non-room-temperature", action="store_true")
    parser.add_argument("--devices-per-family", type=int)
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument(
        "--shrinkage-strengths",
        type=float,
        nargs="+",
        default=[0.1, 0.3, 1, 3, 10, 30, 100],
    )
    parser.add_argument("--family-prior-strength", type=float, default=1.0)
    parser.add_argument("--bootstrap-repeats", type=int, default=1_000)
    parser.add_argument("--reference-bootstrap-repeats", type=int, default=2_000)
    parser.add_argument("--reference-permutations", type=int, default=2_000)
    parser.add_argument(
        "--target-failure-probabilities",
        type=float,
        nargs="+",
        default=[0.1, 0.05, 0.01],
    )
    parser.add_argument(
        "--target-device-coverages",
        type=float,
        nargs="+",
        default=[0.5, 0.9, 0.95],
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--log-level", default="INFO")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    logging.basicConfig(
        level=getattr(logging, args.log_level.upper()),
        format="%(levelname)s %(name)s: %(message)s",
    )
    families = selected_family_names(
        args.artifact,
        args.families,
        include_non_room_temperature=args.include_non_room_temperature,
    )
    if not families:
        raise SystemExit("No room-temperature families selected")
    if args.devices_per_family is not None and args.devices_per_family < 1:
        raise ValueError("devices-per-family must be positive")
    if args.family_prior_strength <= 0:
        raise ValueError("family-prior-strength must be positive")
    if args.bootstrap_repeats < 0:
        raise ValueError("bootstrap-repeats cannot be negative")
    if args.reference_bootstrap_repeats < 1 or args.reference_permutations < 1:
        raise ValueError("reference bootstrap and permutation counts must be positive")
    args.shrinkage_strengths = np.array(
        sorted(set(args.shrinkage_strengths)), dtype=float
    )
    if args.shrinkage_strengths.size == 0 or np.any(args.shrinkage_strengths <= 0):
        raise ValueError("shrinkage-strengths must contain positive values")
    args.target_failure_probabilities = np.array(
        sorted(set(args.target_failure_probabilities)), dtype=float
    )
    args.target_device_coverages = np.array(
        sorted(set(args.target_device_coverages)), dtype=float
    )
    for name in ("target_failure_probabilities", "target_device_coverages"):
        values = getattr(args, name)
        if values.size == 0 or np.any((values <= 0) | (values >= 1)):
            raise ValueError(f"{name.replace('_', '-')} must lie strictly in (0, 1)")

    count_rows: list[dict] = []
    conditional_rows: list[dict] = []
    marginal_rows: list[dict] = []
    metric_rows: list[dict] = []
    device_tail_rows: list[dict] = []
    population_rows: list[dict] = []
    coverage_rows: list[dict] = []
    capacity_rows: list[dict] = []
    polarity_rows: list[dict] = []
    polarity_device_rows: list[dict] = []
    polarity_family_rows: list[dict] = []
    with h5py.File(args.artifact, "r") as h5:
        for family_index, family in enumerate(families):
            responses, references = read_family(h5, family)
            family_errors = np.logical_xor(responses, references[:, None, :]).astype(
                np.uint8
            )
            n_devices = len(responses)
            if n_devices < 2:
                raise ValueError(
                    f"{family} needs at least two devices for family shrinkage"
                )
            device_limit = args.devices_per_family or n_devices
            devices = np.arange(min(device_limit, n_devices))
            LOGGER.info(
                "%s: analyzing %d/%d devices with %d folds",
                family,
                len(devices),
                n_devices,
                args.folds,
            )
            family_error_counts = family_errors.sum(axis=(0, 1))
            family_readouts = family_errors.shape[0] * family_errors.shape[1]
            family_empirical_pmfs = []
            family_independent_pmfs = []
            family_readout_counts = []
            family_polarity_devices = []
            family_weighted_residuals = []
            for device in devices:
                errors = family_errors[device]
                family_prior = leave_one_device_out_prior(
                    family_error_counts,
                    family_readouts,
                    errors,
                    prior_strength=args.family_prior_strength,
                )
                (
                    device_counts,
                    device_conditionals,
                    device_marginals,
                    device_metrics,
                ) = analyze_device(
                    errors,
                    family=family,
                    device=int(device),
                    family_prior=family_prior,
                    shrinkage_strengths=args.shrinkage_strengths,
                    n_folds=args.folds,
                    seed=args.seed + family_index * 100_000 + int(device),
                )
                count_rows.extend(device_counts)
                conditional_rows.extend(device_conditionals)
                marginal_rows.extend(device_marginals)
                metric_rows.append(device_metrics)
                (
                    device_polarity,
                    device_polarity_summary,
                    weighted_residuals,
                ) = reference_polarity_statistics(
                    errors,
                    references[device],
                    device_conditionals,
                    family=family,
                    device=int(device),
                )
                polarity_rows.extend(device_polarity)
                polarity_device_rows.append(device_polarity_summary)
                family_polarity_devices.append(device_polarity_summary)
                family_weighted_residuals.append(weighted_residuals)
                family_empirical_pmfs.append(
                    [row["empirical_probability"] for row in device_counts]
                )
                family_independent_pmfs.append(
                    [row["independent_probability"] for row in device_counts]
                )
                family_readout_counts.append(len(errors))

            polarity_family_rows.append(
                summarize_reference_polarity_family(
                    family_polarity_devices,
                    family_weighted_residuals,
                    references[devices],
                    bootstrap_repeats=args.reference_bootstrap_repeats,
                    permutation_repeats=args.reference_permutations,
                    seed=args.seed + family_index * 100_000 + 88_888,
                )
            )

            (
                family_device_tails,
                family_population,
                family_coverage,
                family_capacities,
            ) = population_error_statistics(
                np.asarray(family_empirical_pmfs),
                np.asarray(family_independent_pmfs),
                np.asarray(family_readout_counts),
                family=family,
                target_failure_probabilities=args.target_failure_probabilities,
                target_device_coverages=args.target_device_coverages,
                bootstrap_repeats=args.bootstrap_repeats,
                seed=args.seed + family_index * 100_000 + 99_999,
            )
            device_tail_rows.extend(family_device_tails)
            population_rows.extend(family_population)
            coverage_rows.extend(family_coverage)
            capacity_rows.extend(family_capacities)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    pl.DataFrame(count_rows).write_parquet(
        args.output_dir / "count_distributions.parquet"
    )
    pl.DataFrame(conditional_rows).write_parquet(
        args.output_dir / "conditional_probabilities.parquet"
    )
    pl.DataFrame(marginal_rows).write_parquet(
        args.output_dir / "marginal_estimates.parquet"
    )
    pl.DataFrame(metric_rows).write_parquet(args.output_dir / "device_metrics.parquet")
    pl.DataFrame(device_tail_rows).write_parquet(
        args.output_dir / "device_tail_risks.parquet"
    )
    pl.DataFrame(population_rows).write_parquet(
        args.output_dir / "population_survival.parquet"
    )
    pl.DataFrame(coverage_rows).write_parquet(
        args.output_dir / "device_coverage.parquet"
    )
    pl.DataFrame(capacity_rows).write_csv(args.output_dir / "predictive_capacities.csv")
    pl.DataFrame(polarity_rows).write_parquet(
        args.output_dir / "reference_polarity_conditionals.parquet"
    )
    pl.DataFrame(polarity_device_rows).write_parquet(
        args.output_dir / "reference_polarity_device_summary.parquet"
    )
    pl.DataFrame(polarity_family_rows).write_csv(
        args.output_dir / "reference_polarity_family_summary.csv"
    )
    LOGGER.info("Wrote results to %s", args.output_dir)


if __name__ == "__main__":
    main()

