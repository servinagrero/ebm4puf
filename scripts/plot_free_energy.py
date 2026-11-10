"""Plot the artifacts produced by ``analysis_free_energy.py``."""

import argparse
import csv
import logging
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import polars as pl

LOGGER = logging.getLogger("ebm4puf.plot_free_energy")
SCENARIO_LABELS = {
    "genuine": "Genuine empirical",
    "rbm": "RBM-generated",
    "marginal": "Independent, same marginals",
    "mean_ber": "Independent, same mean BER",
    "uniform": "Independent, p=0.5",
    "permuted": "Fixed bit permutation",
    "cross_device": "Cross-device impostor",
}
SCORE_LABELS = {
    "independent": "Independent marginal",
    "rbm": "Raw RBM",
    "residual": "Residual RBM",
}
SCORE_COLORS = {
    "independent": "tab:blue",
    "rbm": "tab:orange",
    "residual": "tab:green",
}
JOINT_TEST_LABELS = {
    "marginal_anomaly": "Marginal anomaly",
    "residual_anomaly": "Residual anomaly",
    "joint_mahalanobis": "Joint Mahalanobis",
}
DIRECTIONAL_TEST_LABELS = {
    "marginal_directional": "Marginal directional",
    "residual_directional": "Residual directional",
    "joint_bonferroni": "Joint Bonferroni",
}


def read_rows(path: Path) -> list[dict[str, str]]:
    if not path.is_file():
        raise FileNotFoundError(f"Missing analysis artifact: {path}")
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def values_in(rows: list[dict[str, str]], column: str) -> list:
    converter = int if column in {"cd_k", "gibbs_steps"} else str
    return sorted({converter(row[column]) for row in rows})


def training_configs(rows: list[dict[str, str]]) -> list[tuple[str, int]]:
    return sorted({(row["training_method"], int(row["cd_k"])) for row in rows})


def config_label(training_method: str, cd_k: int) -> str:
    return f"{training_method.upper()}-{cd_k}"


def matches_config(row: dict[str, str], training_method: str, cd_k: int) -> bool:
    return row["training_method"] == training_method and int(row["cd_k"]) == cd_k


def quartile_curve(
    rows: list[dict[str, str]],
    x_column: str,
    value_column: str,
    x_converter=int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    x_values = np.array(sorted({x_converter(row[x_column]) for row in rows}))
    quartiles = np.array(
        [
            np.nanquantile(
                [
                    float(row[value_column])
                    for row in rows
                    if x_converter(row[x_column]) == x
                ],
                [0.25, 0.5, 0.75],
            )
            for x in x_values
        ]
    )
    return x_values, quartiles[:, 0], quartiles[:, 1], quartiles[:, 2]


def add_curve(
    ax,
    rows,
    x_column,
    value_column,
    label,
    *,
    x_converter=int,
    linestyle="-",
):
    x, lower, median, upper = quartile_curve(
        rows,
        x_column,
        value_column,
        x_converter=x_converter,
    )
    (line,) = ax.plot(
        x,
        median,
        marker="o",
        linestyle=linestyle,
        label=label,
    )
    ax.fill_between(x, lower, upper, color=line.get_color(), alpha=0.15)


def plot_gaussian_auc(
    rows,
    family,
    training_method,
    cd_k,
    scenario,
    output_path,
):
    selected = [
        row
        for row in rows
        if row["family"] == family
        and matches_config(row, training_method, cd_k)
        and row["scenario"] == scenario
        and row["initialization"] == "none"
    ]
    fig, axes = plt.subplots(
        1,
        3,
        figsize=(15, 4.5),
        sharex=True,
        sharey=True,
    )
    for ax, score in zip(axes, SCORE_LABELS):
        score_rows = [row for row in selected if row["score"] == score]
        add_curve(
            ax,
            score_rows,
            "group_size",
            "empirical_auc",
            "Empirical bootstrap",
        )
        add_curve(
            ax,
            score_rows,
            "group_size",
            "gaussian_auc",
            "Gaussian prediction",
            linestyle="--",
        )
        ax.axhline(
            0.5,
            color="black",
            linestyle=":",
            linewidth=1,
        )
        ax.set(
            title=SCORE_LABELS[score],
            xlabel="Readouts per decision",
            ylim=(-0.05, 1.05),
        )
    axes[0].set_ylabel("Genuine-vs-alternative AUROC")
    axes[-1].legend(fontsize=8)
    fig.suptitle(
        f"{family}: {config_label(training_method, cd_k)}, {SCENARIO_LABELS[scenario]}"
    )
    fig.text(
        0.5,
        0.01,
        "Shading: interquartile range across devices",
        ha="center",
        fontsize=9,
    )
    fig.tight_layout(rect=(0, 0.04, 1, 1))
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def plot_joint_detection(
    rows,
    family,
    training_method,
    cd_k,
    metric,
    ylabel,
    output_path,
    *,
    test_labels=JOINT_TEST_LABELS,
) -> None:
    selected = [
        row
        for row in rows
        if row["family"] == family
        and matches_config(row, training_method, cd_k)
        and row["initialization"] == "none"
    ]
    fig, axes = plt.subplots(
        1,
        3,
        figsize=(15, 4.5),
        sharex=True,
        sharey=True,
    )
    for ax, (test_name, test_label) in zip(axes, test_labels.items()):
        test_rows = [row for row in selected if row["test"] == test_name]
        for scenario in sorted({row["scenario"] for row in test_rows}):
            add_curve(
                ax,
                [row for row in test_rows if row["scenario"] == scenario],
                "group_size",
                metric,
                SCENARIO_LABELS[scenario],
            )
        if metric == "empirical_auc":
            ax.axhline(0.5, color="black", linestyle=":", linewidth=1)
        ax.set(
            title=test_label,
            xlabel="Readouts per decision",
            ylim=(-0.05, 1.05),
        )
    axes[0].set_ylabel(ylabel)
    axes[-1].legend(fontsize=8, ncol=2)
    fig.suptitle(f"{family}: {config_label(training_method, cd_k)}")
    fig.text(
        0.5,
        0.01,
        "Shading: interquartile range across devices",
        ha="center",
        fontsize=9,
    )
    fig.tight_layout(rect=(0, 0.04, 1, 1))
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def plot_required_readouts(
    rows,
    family,
    training_method,
    cd_k,
    scenario,
    output_path,
):
    selected = [
        row
        for row in rows
        if row["family"] == family
        and matches_config(row, training_method, cd_k)
        and row["scenario"] == scenario
        and row["initialization"] == "none"
    ]
    fig, axes = plt.subplots(
        1,
        3,
        figsize=(15, 4.5),
        sharex=True,
        sharey=True,
    )
    for ax, (score, score_label) in zip(axes, SCORE_LABELS.items()):
        all_score_rows = [row for row in selected if row["score"] == score]
        all_devices = sorted({int(row["device"]) for row in all_score_rows})
        score_rows = [row for row in all_score_rows if row["required_readouts"].strip()]
        directional_devices = sorted({int(row["device"]) for row in score_rows})

        for device in directional_devices:
            device_rows = sorted(
                (row for row in score_rows if int(row["device"]) == device),
                key=lambda row: float(row["target_auc"]),
            )
            ax.plot(
                [float(row["target_auc"]) for row in device_rows],
                [float(row["required_readouts"]) for row in device_rows],
                color=SCORE_COLORS[score],
                alpha=0.18,
                linewidth=0.8,
            )

        if score_rows:
            x, _, median, _ = quartile_curve(
                score_rows,
                "target_auc",
                "required_readouts",
                x_converter=float,
            )
            ax.plot(
                x,
                median,
                color=SCORE_COLORS[score],
                marker="o",
                linewidth=2.5,
                label="Conditional median",
            )
            ax.set_yscale("log")
            ax.legend(fontsize=8)
        else:
            ax.text(
                0.5,
                0.5,
                "No directional devices",
                ha="center",
                va="center",
                transform=ax.transAxes,
            )

        ax.set(
            title=(
                f"{score_label}\n"
                f"{len(directional_devices)}/{len(all_devices)} "
                "devices directional"
            ),
            xlabel="Target directional AUROC",
            ylabel="Required readouts" if ax is axes[0] else "",
            xlim=(None, 1.01),
        )
    fig.suptitle(
        f"{family}: {config_label(training_method, cd_k)}, {SCENARIO_LABELS[scenario]}"
    )
    fig.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def write_required_readouts_summary(
    input_path: Path,
    output_path: Path,
) -> None:
    """Write device coverage and conditional required-N summaries."""
    rows = (
        pl.read_csv(input_path, infer_schema_length=None)
        .filter(
            (pl.col("scenario") == "marginal") & (pl.col("initialization") == "none")
        )
        .group_by(
            "family",
            "training_method",
            "cd_k",
            "score",
            "target_auc",
        )
        .agg(
            pl.col("device").n_unique().alias("n_devices"),
            pl.col("device")
            .filter(pl.col("required_readouts").is_not_null())
            .n_unique()
            .alias("n_directional"),
            pl.col("required_readouts").quantile(0.25).alias("conditional_q25"),
            pl.col("required_readouts").median().alias("conditional_median"),
            pl.col("required_readouts").quantile(0.75).alias("conditional_q75"),
            pl.col("mean_difference").median().alias("median_mean_difference"),
        )
        .with_columns(
            (pl.col("n_directional") / pl.col("n_devices")).alias(
                "directional_fraction"
            )
        )
        .sort(
            "family",
            "training_method",
            "cd_k",
            "score",
            "target_auc",
        )
    )
    rows.write_csv(output_path)


def plot_error_count_distribution(
    rows: list[dict[str, str]],
    family: str,
    training_method: str,
    cd_k: int,
    gibbs_steps: int,
    initialization: str,
    output_path: Path,
) -> None:
    selected = [
        row
        for row in rows
        if row["family"] == family
        and (
            row["source"] != "rbm"
            or (
                matches_config(row, training_method, cd_k)
                and int(row["gibbs_steps"]) == gibbs_steps
                and row["initialization"] == initialization
            )
        )
    ]
    source_rows = {
        source: sorted(
            (row for row in selected if row["source"] == source),
            key=lambda row: int(row["error_count"]),
        )
        for source in ("empirical", "independent", "rbm")
    }
    fig, ax = plt.subplots(figsize=(8, 4.5))
    for source, color, label in (
        ("empirical", "tab:blue", "Empirical"),
        ("rbm", "tab:orange", "RBM synthetic"),
    ):
        ax.bar(
            [int(row["error_count"]) for row in source_rows[source]],
            [float(row["probability"]) for row in source_rows[source]],
            width=0.9,
            alpha=0.45,
            color=color,
            label=label,
        )
    ax.plot(
        [int(row["error_count"]) for row in source_rows["independent"]],
        [float(row["probability"]) for row in source_rows["independent"]],
        color="black",
        linestyle="--",
        label="Independent Poisson-binomial",
    )
    support = [
        int(row["error_count"]) for row in selected if float(row["probability"]) > 1e-4
    ]
    if support:
        ax.set_xlim(max(0, min(support) - 1), max(support) + 1)
    ax.set(
        title=(
            f"{family}: {config_label(training_method, cd_k)}, "
            f"Gibbs {gibbs_steps}, {initialization} init"
        ),
        xlabel="Bit flips per readout",
        ylabel="Probability",
        ylim=(0, None),
    )
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def plot_static_authentication_auc(rows, family, training_method, cd_k, output_path):
    selected = [
        row
        for row in rows
        if row["family"] == family
        and matches_config(row, training_method, cd_k)
        and row["scenario"] != "genuine"
    ]
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5), sharex=True, sharey=True)
    for scenario in sorted({row["scenario"] for row in selected}):
        scenario_rows = [row for row in selected if row["scenario"] == scenario]
        for ax, metric in zip(axes, ("free_energy_auc", "evidence_auc")):
            add_curve(
                ax,
                scenario_rows,
                "group_size",
                metric,
                SCENARIO_LABELS[scenario],
            )
    for ax, title in zip(axes, ("Raw RBM score", "RBM vs marginal evidence")):
        ax.axhline(0.5, color="black", linestyle="--", linewidth=1)
        ax.set(title=title, xlabel="Readouts per decision", ylim=(-0.05, 1.05))
    axes[0].set_ylabel("AUROC")
    axes[1].legend(fontsize=8, ncol=2)
    fig.suptitle(f"{family}: {config_label(training_method, cd_k)}")
    fig.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def plot_generation_auc(rows, family, initialization, output_path):
    selected = [
        row
        for row in rows
        if row["family"] == family and row["initialization"] == initialization
    ]
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5), sharex=True, sharey=True)
    for training_method, cd_k in training_configs(selected):
        for gibbs_steps in values_in(selected, "gibbs_steps"):
            configuration = [
                row
                for row in selected
                if matches_config(row, training_method, cd_k)
                and int(row["gibbs_steps"]) == gibbs_steps
            ]
            for ax, metric in zip(axes, ("free_energy_auc", "evidence_auc")):
                add_curve(
                    ax,
                    configuration,
                    "group_size",
                    metric,
                    f"{config_label(training_method, cd_k)}, Gibbs {gibbs_steps}",
                )
    for ax, title in zip(axes, ("Raw RBM score", "RBM vs marginal evidence")):
        ax.axhline(0.5, color="black", linestyle="--", linewidth=1)
        ax.set(title=title, xlabel="Readouts per decision", ylim=(-0.05, 1.05))
    axes[0].set_ylabel("Genuine vs RBM-generated AUROC")
    axes[1].legend(fontsize=7, ncol=2)
    fig.suptitle(f"{family}: {initialization} initialization")
    fig.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def plot_generation_metric(rows, family, initialization, metric, ylabel, output_path):
    selected = [
        row
        for row in rows
        if row["family"] == family and row["initialization"] == initialization
    ]
    fig, ax = plt.subplots(figsize=(7, 4.2))
    for training_method, cd_k in training_configs(selected):
        add_curve(
            ax,
            [row for row in selected if matches_config(row, training_method, cd_k)],
            "gibbs_steps",
            metric,
            config_label(training_method, cd_k),
        )
    ax.set(
        title=f"{family}: {initialization} initialization",
        xlabel="Generation Gibbs steps",
        ylabel=ylabel,
    )
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def plot_normalized_scores(
    authentication,
    generation,
    family,
    training_method,
    cd_k,
    initialization,
    output_path,
):
    static = [
        row
        for row in authentication
        if row["family"] == family
        and matches_config(row, training_method, cd_k)
        and row["scenario"] in {"genuine", "marginal", "permuted"}
    ]
    generated = [
        row
        for row in generation
        if row["family"] == family
        and matches_config(row, training_method, cd_k)
        and row["initialization"] == initialization
    ]
    fig, ax = plt.subplots(figsize=(8, 4.5))
    for scenario in ("genuine", "marginal", "permuted"):
        scenario_rows = [row for row in static if row["scenario"] == scenario]
        for row in scenario_rows:
            for quantile in ("q25", "q50", "q75"):
                row[f"normalized_{quantile}"] = float(
                    row[f"rbm_score_{quantile}"]
                ) / int(row["group_size"])
        add_curve(
            ax,
            scenario_rows,
            "group_size",
            "normalized_q50",
            SCENARIO_LABELS[scenario],
        )
    for gibbs_steps in values_in(generated, "gibbs_steps"):
        add_curve(
            ax,
            [row for row in generated if int(row["gibbs_steps"]) == gibbs_steps],
            "group_size",
            "normalized_rbm_score_q50",
            f"RBM Gibbs {gibbs_steps}",
        )
    ax.set(
        title=(
            f"{family}: {config_label(training_method, cd_k)}, {initialization} init"
        ),
        xlabel="Readouts per decision",
        ylabel=r"Median $S_N/N$",
    )
    ax.legend(fontsize=8, ncol=2)
    fig.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def plot_centered_evidence(
    authentication,
    generation,
    family,
    training_method,
    cd_k,
    initialization,
    output_path,
):
    static = [
        row
        for row in authentication
        if row["family"] == family
        and matches_config(row, training_method, cd_k)
        and row["scenario"] in {"genuine", "marginal"}
    ]
    generated = [
        row
        for row in generation
        if row["family"] == family
        and matches_config(row, training_method, cd_k)
        and row["initialization"] == initialization
    ]
    fig, ax = plt.subplots(figsize=(8, 4.5))
    for scenario in ("genuine", "marginal"):
        add_curve(
            ax,
            [row for row in static if row["scenario"] == scenario],
            "group_size",
            "centered_evidence_q50",
            SCENARIO_LABELS[scenario],
        )
    for gibbs_steps in values_in(generated, "gibbs_steps"):
        add_curve(
            ax,
            [row for row in generated if int(row["gibbs_steps"]) == gibbs_steps],
            "group_size",
            "centered_evidence_q50",
            f"RBM Gibbs {gibbs_steps}",
        )
    ax.axhline(0, color="black", linestyle="--", linewidth=1)
    ax.set(
        title=(
            f"{family}: {config_label(training_method, cd_k)}, {initialization} init"
        ),
        xlabel="Readouts per decision",
        ylabel=r"Centered $\Delta\widetilde{\mathrm{LLR}}_N$",
    )
    ax.legend(fontsize=8, ncol=2)
    fig.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def plot_relaxation(rows, family, training_method, cd_k, output_path):
    selected = [
        row
        for row in rows
        if row["family"] == family and matches_config(row, training_method, cd_k)
    ]
    fig, ax = plt.subplots(figsize=(8, 4.5))
    for scenario in sorted({row["scenario"] for row in selected}):
        add_curve(
            ax,
            [row for row in selected if row["scenario"] == scenario],
            "relaxation_steps",
            "mean_transition_hamming_fraction",
            SCENARIO_LABELS[scenario],
        )
    ax.set(
        title=f"{family}: {config_label(training_method, cd_k)}",
        xlabel=r"Gibbs transition $k-1 \rightarrow k$",
        ylabel="Mean transition Hamming fraction",
    )
    ax.legend(fontsize=8, ncol=2)
    fig.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, default=Path("data/rbm_free_energy"))
    parser.add_argument(
        "--output-dir", type=Path, default=Path("figures/rbm_free_energy")
    )
    parser.add_argument("--log-level", default="INFO")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    logging.basicConfig(
        level=getattr(logging, args.log_level.upper()),
        format="%(levelname)s %(name)s: %(message)s",
    )
    authentication = read_rows(args.input_dir / "authentication.csv")
    generation = read_rows(args.input_dir / "generation_sweep.csv")
    relaxation = read_rows(args.input_dir / "relaxation.csv")
    error_counts = read_rows(args.input_dir / "error_count_distributions.csv")
    score_tests = read_rows(args.input_dir / "score_tests.csv")
    joint_score_tests = read_rows(args.input_dir / "joint_score_tests.csv")
    required_readouts = read_rows(args.input_dir / "required_readouts.csv")
    write_required_readouts_summary(
        args.input_dir / "required_readouts.csv",
        args.input_dir / "required_readouts_summary.csv",
    )

    directories = {
        name: args.output_dir / name
        for name in (
            "authentication_auc",
            "generation_sweep",
            "error_count_distributions",
            "normalized_scores",
            "centered_evidence",
            "relaxation",
            "gaussian_auc",
            "joint_detection",
            "required_readouts",
        )
    }
    for directory in directories.values():
        directory.mkdir(parents=True, exist_ok=True)

    families = values_in(authentication, "family")
    configurations = training_configs(authentication)
    gibbs_values = values_in(generation, "gibbs_steps")
    initializations = values_in(generation, "initialization")
    statistical_scenarios = sorted(
        {row["scenario"] for row in score_tests if row["initialization"] == "none"}
    )
    for family in families:
        for initialization in initializations:
            plot_generation_auc(
                generation,
                family,
                initialization,
                directories["generation_sweep"] / f"{family}_{initialization}_auc.png",
            )
            plot_generation_metric(
                generation,
                family,
                initialization,
                "error_count_cdf_distance",
                "Error-count CDF distance (lower is better)",
                directories["generation_sweep"]
                / f"{family}_{initialization}_error_count_distance.png",
            )
            plot_generation_metric(
                generation,
                family,
                initialization,
                "marginal_mae",
                "Marginal MAE (lower is better)",
                directories["generation_sweep"]
                / f"{family}_{initialization}_marginal_mae.png",
            )
        for training_method, cd_k in configurations:
            configuration_name = f"{training_method}_k{cd_k}"
            plot_joint_detection(
                joint_score_tests,
                family,
                training_method,
                cd_k,
                "empirical_auc",
                "Genuine-vs-alternative AUROC",
                directories["joint_detection"]
                / f"{family}_{configuration_name}_auc.png",
            )
            plot_joint_detection(
                joint_score_tests,
                family,
                training_method,
                cd_k,
                "alternative_acceptance_rate",
                "Alternative acceptance rate",
                directories["joint_detection"]
                / f"{family}_{configuration_name}_far.png",
            )
            plot_joint_detection(
                joint_score_tests,
                family,
                training_method,
                cd_k,
                "empirical_auc",
                "Genuine-vs-alternative AUROC",
                directories["joint_detection"]
                / f"{family}_{configuration_name}_directional_auc.png",
                test_labels=DIRECTIONAL_TEST_LABELS,
            )
            plot_joint_detection(
                joint_score_tests,
                family,
                training_method,
                cd_k,
                "alternative_acceptance_rate",
                "Alternative acceptance rate",
                directories["joint_detection"]
                / f"{family}_{configuration_name}_directional_far.png",
                test_labels=DIRECTIONAL_TEST_LABELS,
            )
            for scenario in statistical_scenarios:
                plot_gaussian_auc(
                    score_tests,
                    family,
                    training_method,
                    cd_k,
                    scenario,
                    directories["gaussian_auc"]
                    / f"{family}_{configuration_name}_{scenario}.png",
                )
            plot_required_readouts(
                required_readouts,
                family,
                training_method,
                cd_k,
                "marginal",
                directories["required_readouts"]
                / f"{family}_{configuration_name}_marginal.png",
            )
            plot_static_authentication_auc(
                authentication,
                family,
                training_method,
                cd_k,
                directories["authentication_auc"]
                / f"{family}_{configuration_name}.png",
            )
            for initialization in initializations:
                plot_normalized_scores(
                    authentication,
                    generation,
                    family,
                    training_method,
                    cd_k,
                    initialization,
                    directories["normalized_scores"]
                    / f"{family}_{configuration_name}_{initialization}.png",
                )
                plot_centered_evidence(
                    authentication,
                    generation,
                    family,
                    training_method,
                    cd_k,
                    initialization,
                    directories["centered_evidence"]
                    / f"{family}_{configuration_name}_{initialization}.png",
                )
            plot_relaxation(
                relaxation,
                family,
                training_method,
                cd_k,
                directories["relaxation"]
                / f"{family}_{configuration_name}_transition_hamming.png",
            )
            for gibbs_steps in gibbs_values:
                for initialization in initializations:
                    plot_error_count_distribution(
                        error_counts,
                        family,
                        training_method,
                        cd_k,
                        gibbs_steps,
                        initialization,
                        directories["error_count_distributions"]
                        / (
                            f"{family}_{configuration_name}_g{gibbs_steps}_"
                            f"{initialization}.png"
                        ),
                    )
    LOGGER.info("Wrote figures to %s", args.output_dir)


if __name__ == "__main__":
    main()
