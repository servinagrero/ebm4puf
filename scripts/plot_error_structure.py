"""Plot the artifacts produced by ``analysis_error_structure.py``."""

import argparse
import logging
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import polars as pl

LOGGER = logging.getLogger("ebm4puf.plot_error_structure")


def plot_ecc_capacity_curves(
    counts: pl.DataFrame,
    family: str,
    output_path: Path,
) -> None:
    """Plot empirical and independent ECC success probabilities."""
    selected = counts.filter(pl.col("family") == family).sort(["device", "k"])
    devices = selected["device"].unique().sort().to_list()
    n_bits = int(selected["k"].to_numpy().max())
    empirical = []
    independent = []
    maximum_observed = 0
    for device in devices:
        device_rows = selected.filter(pl.col("device") == device).sort("k")
        empirical_pmf = device_rows["empirical_probability"].to_numpy()
        independent_pmf = device_rows["independent_probability"].to_numpy()
        empirical.append(np.cumsum(empirical_pmf))
        independent.append(np.cumsum(independent_pmf))
        maximum_observed = max(
            maximum_observed,
            int(np.flatnonzero(empirical_pmf)[-1]),
        )
    empirical = np.asarray(empirical)
    independent = np.asarray(independent)
    tau_max = min(n_bits, maximum_observed + 3)
    tau = np.arange(tau_max + 1)

    fig, axes = plt.subplots(
        2,
        1,
        figsize=(8, 6.5),
        sharex=True,
        gridspec_kw={"height_ratios": [2, 1]},
    )
    for values, label, color in (
        (empirical, "Empirical", "tab:blue"),
        (independent, "Independent Poisson-binomial", "tab:orange"),
    ):
        lower, median, upper = np.quantile(
            values[:, : tau_max + 1], [0.25, 0.5, 0.75], axis=0
        )
        axes[0].plot(tau, median, marker="o", label=label, color=color)
        axes[0].fill_between(tau, lower, upper, color=color, alpha=0.18)

    difference = independent - empirical
    lower, median, upper = np.quantile(
        difference[:, : tau_max + 1], [0.25, 0.5, 0.75], axis=0
    )
    axes[1].plot(tau, median, marker="o", color="tab:purple")
    axes[1].fill_between(tau, lower, upper, color="tab:purple", alpha=0.18)
    axes[1].axhline(0, color="black", linestyle=":", linewidth=1)
    axes[0].set(
        title=f"{family}: probability of correcting a readout",
        ylabel=r"$P(K \leq \tau)$",
        ylim=(-0.03, 1.03),
    )
    axes[1].set(
        xlabel=r"Correction capacity $\tau$",
        ylabel="Independent − empirical",
    )
    axes[0].legend()
    fig.text(
        0.5,
        0.01,
        "Lines: device median; shading: device interquartile range",
        ha="center",
        fontsize=9,
    )
    fig.tight_layout(rect=(0, 0.03, 1, 1))
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def conditional_family_summary(
    conditionals: pl.DataFrame,
    family: str,
) -> pl.DataFrame:
    """Return equal-device conditional residual summaries."""
    return (
        conditionals.filter(pl.col("family") == family)
        .group_by(["k", "bit"])
        .agg(
            pl.col("residual").mean().alias("mean_residual"),
            (
                pl.col("residual").quantile(0.75) - pl.col("residual").quantile(0.25)
            ).alias("residual_iqr"),
            pl.col("device").n_unique().alias("device_support"),
        )
        .sort(["k", "bit"])
    )


def heatmap_matrix(
    summary: pl.DataFrame,
    column: str,
    *,
    n_bits: int,
    maximum_k: int,
    minimum_devices: int,
) -> np.ndarray:
    """Convert a tidy conditional summary to a masked heatmap matrix."""
    matrix = np.full((maximum_k + 1, n_bits), np.nan)
    for row in summary.iter_rows(named=True):
        if row["k"] <= maximum_k and row["device_support"] >= minimum_devices:
            matrix[row["k"], row["bit"]] = row[column]
    return matrix


def plot_conditional_residual_heatmaps(
    summary: pl.DataFrame,
    family: str,
    output_path: Path,
    *,
    minimum_devices: int,
    total_devices: int,
) -> None:
    """Plot mean conditional residuals, heterogeneity, and support."""
    supported = summary.filter(pl.col("device_support") >= minimum_devices)
    if supported.is_empty():
        LOGGER.warning("Skipping %s: no conditional rows meet support", family)
        return
    n_bits = int(summary["bit"].to_numpy().max()) + 1
    maximum_k = int(summary["k"].to_numpy().max())
    mean = heatmap_matrix(
        summary,
        "mean_residual",
        n_bits=n_bits,
        maximum_k=maximum_k,
        minimum_devices=minimum_devices,
    )
    iqr = heatmap_matrix(
        summary,
        "residual_iqr",
        n_bits=n_bits,
        maximum_k=maximum_k,
        minimum_devices=minimum_devices,
    )
    support_rows = summary.group_by("k").agg(pl.col("device_support").max()).sort("k")
    support = np.zeros(maximum_k + 1, dtype=int)
    support[support_rows["k"].to_numpy()] = support_rows["device_support"].to_numpy()
    displayed_mean = np.abs(mean[np.isfinite(mean)])
    displayed_iqr = iqr[np.isfinite(iqr)]
    residual_limit = max(float(np.quantile(displayed_mean, 0.99)), 1e-6)
    iqr_limit = max(float(np.quantile(displayed_iqr, 0.99)), 1e-6)

    fig, axes = plt.subplots(
        1,
        3,
        figsize=(16, 5),
        gridspec_kw={"width_ratios": [3, 3, 1.3]},
    )
    mean_image = axes[0].imshow(
        mean,
        aspect="auto",
        origin="lower",
        cmap="coolwarm",
        vmin=-residual_limit,
        vmax=residual_limit,
    )
    iqr_image = axes[1].imshow(
        iqr,
        aspect="auto",
        origin="lower",
        cmap="viridis",
        vmin=0,
        vmax=iqr_limit,
    )
    axes[2].stairs(
        support,
        np.arange(maximum_k + 2) - 0.5,
        orientation="horizontal",
        color="tab:blue",
    )
    axes[2].axvline(minimum_devices, color="black", linestyle=":", linewidth=1)
    axes[2].axvline(0, color="black", linestyle="--", linewidth=1)
    axes[0].set(title="Mean residual", xlabel="Bit position", ylabel="Error count k")
    # axes[0].xaxis.set_major_locator(MultipleLocator(32))
    axes[0].set_xticks(np.arange(0, 257, 32))
    axes[1].set(title="Across-device residual IQR", xlabel="Bit position")
    # axes[1].xaxis.set_major_locator(MultipleLocator(32))
    axes[1].set_xticks(np.arange(0, 257, 32))
    axes[2].set(
        title=f"Support (threshold {minimum_devices}/{total_devices})",
        xlabel=r"Devices with at least one $K=k$ readout",
        ylabel="Error count k",
    )
    axes[2].set_ylim(-0.5, maximum_k + 0.5)
    fig.colorbar(
        mean_image,
        ax=axes[0],
        label=r"$P(E_i=1\mid K=k)-P_0(E_i=1\mid K=k)$",
        # extend="both",
    )
    fig.colorbar(
        iqr_image,
        ax=axes[1],
        label="Interquartile range",
        # extend="max",
    )
    fig.suptitle(f"{family}: conditional departures from independence")
    fig.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def plot_population_survival(
    survival: pl.DataFrame,
    family: str,
    output_path: Path,
) -> None:
    """Plot random-device tail risk and upper device-risk quantiles."""
    selected = survival.filter(pl.col("family") == family).sort("tau")
    observed_max = int(selected["observed_max_k"][0])
    selected = selected.filter(pl.col("tau") <= observed_max + 3)
    tau = selected["tau"].to_numpy()
    empirical = selected["empirical_random_device_tail"].to_numpy()
    independent = selected["independent_random_device_tail"].to_numpy()
    lower = selected["empirical_bootstrap_q025"].to_numpy()
    upper = selected["empirical_bootstrap_q975"].to_numpy()

    fig, axes = plt.subplots(2, 1, figsize=(8, 7), sharex=True)
    axes[0].plot(
        tau,
        np.ma.masked_less_equal(empirical, 0),
        marker="o",
        label="Empirical random-device mean",
    )
    axes[0].plot(
        tau,
        np.ma.masked_less_equal(independent, 0),
        marker="o",
        linestyle="--",
        label="Rare-error independent mixture",
    )
    axes[0].fill_between(
        tau,
        lower,
        upper,
        where=lower > 0,
        alpha=0.18,
        label="Nested bootstrap 95% interval",
    )
    axes[0].set_yscale("log")
    axes[0].set(
        title=f"{family}: random-device error-count survival",
        ylabel=r"$P(K>\tau)$",
    )
    axes[0].legend(fontsize=8)

    for column, label in (
        ("empirical_device_q50", "Median device"),
        ("empirical_device_q90", "90th-percentile device"),
        ("empirical_device_q95", "95th-percentile device"),
    ):
        axes[1].plot(tau, selected[column].to_numpy(), marker="o", label=label)
    axes[1].set(
        xlabel=r"Correction capacity $\tau$",
        ylabel=r"Device risk $P_d(K>\tau)$",
        ylim=(-0.03, 1.03),
    )
    axes[1].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def plot_device_coverage(
    coverage: pl.DataFrame,
    family: str,
    output_path: Path,
) -> None:
    """Plot the fraction of devices meeting each failure-risk target."""
    selected = coverage.filter(pl.col("family") == family)
    tau_limits = []
    for target in selected["target_failure_probability"].unique().to_list():
        target_rows = selected.filter(
            pl.col("target_failure_probability") == target
        ).sort("tau")
        for column in (
            "empirical_device_coverage",
            "independent_device_coverage",
        ):
            complete = np.flatnonzero(target_rows[column].to_numpy() >= 1)
            if complete.size:
                tau_limits.append(int(target_rows["tau"][int(complete[0])]))
    tau_max = min(
        int(selected["tau"].to_numpy().max()),
        max(tau_limits, default=0) + 3,
    )
    selected = selected.filter(pl.col("tau") <= tau_max)
    fig, ax = plt.subplots(figsize=(8, 5.5))
    empirical_groups = (
        selected.select(
            "maximum_observed_failures",
            "n_readouts",
            "effective_empirical_probability",
        )
        .unique()
        .sort("maximum_observed_failures")
    )
    for index, group in enumerate(empirical_groups.iter_rows(named=True)):
        target_rows = selected.filter(
            pl.col("maximum_observed_failures") == group["maximum_observed_failures"]
        ).sort("tau")
        target_rows = (
            target_rows.group_by("tau")
            .agg(pl.col("empirical_device_coverage").first())
            .sort("tau")
        )
        ax.plot(
            target_rows["tau"],
            target_rows["empirical_device_coverage"],
            marker="o",
            color=f"C{index}",
            label=(
                f"Empirical: at most {group['maximum_observed_failures']}/"
                f"{group['n_readouts']} observed failures"
            ),
        )
    for index, target in enumerate(
        selected["target_failure_probability"].unique().sort().to_list()
    ):
        target_rows = selected.filter(
            pl.col("target_failure_probability") == target
        ).sort("tau")
        ax.plot(
            target_rows["tau"],
            target_rows["independent_device_coverage"],
            linestyle="--",
            color=f"C{index + len(empirical_groups)}",
            alpha=0.75,
            label=rf"Independent $P_d(K>\tau)\leq {target:g}$",
        )
    ax.set(
        title=f"{family}: device coverage by correction capacity",
        xlabel=r"Correction capacity $\tau$",
        ylabel="Fraction of devices meeting target",
        ylim=(-0.03, 1.03),
    )
    ax.legend(fontsize=8, ncol=2)
    fig.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def plot_device_heterogeneity(metrics: pl.DataFrame, output_path: Path) -> None:
    """Plot count mismatch against conditional-location mismatch."""
    fig, ax = plt.subplots(figsize=(8, 5.5))
    for family in metrics["family"].unique().sort().to_list():
        selected = metrics.filter(pl.col("family") == family)
        ax.scatter(
            selected["count_cdf_distance"],
            selected["conditional_residual_distance"],
            alpha=0.7,
            label=family,
        )
    ax.set(
        title="Device heterogeneity in count and conditional structure",
        xlabel="Empirical vs independent count CDF distance",
        ylabel="Conditional residual distance",
    )
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def plot_reference_polarity_landscape(
    polarity: pl.DataFrame,
    family: str,
    output_path: Path,
    *,
    minimum_devices: int,
) -> None:
    """Plot 0-to-1 versus 1-to-0 conditional error behavior."""
    selected = polarity.filter(pl.col("family") == family)
    support = (
        selected.select("device", "k")
        .unique()
        .group_by("k")
        .agg(pl.len().alias("device_support"))
        .sort("k")
    )
    supported_k = support.filter(pl.col("device_support") >= minimum_devices)[
        "k"
    ].to_list()
    selected = selected.filter(pl.col("k").is_in(supported_k))
    if selected.is_empty():
        LOGGER.warning("Skipping %s polarity plot: insufficient support", family)
        return
    class_summary = (
        selected.group_by(["k", "reference_value"])
        .agg(
            pl.col("empirical_flip_probability").mean(),
            pl.col("independent_flip_probability").mean(),
        )
        .sort(["reference_value", "k"])
    )
    contrast = (
        selected.filter(pl.col("reference_value") == 0)
        .group_by("k")
        .agg(
            pl.col("empirical_contrast").mean(),
            pl.col("independent_contrast").mean(),
            pl.col("residual_contrast").mean(),
            pl.col("residual_contrast").quantile(0.25).alias("residual_q25"),
            pl.col("residual_contrast").quantile(0.75).alias("residual_q75"),
        )
        .sort("k")
    )
    support = support.filter(pl.col("k").is_in(supported_k)).sort("k")

    fig, axes = plt.subplots(
        1,
        4,
        figsize=(17, 4.7),
        gridspec_kw={"width_ratios": [2.5, 2.2, 2.2, 1.2]},
    )
    for reference_value, color in ((0, "tab:blue"), (1, "tab:orange")):
        rows = class_summary.filter(pl.col("reference_value") == reference_value).sort(
            "k"
        )
        direction = "0→1" if reference_value == 0 else "1→0"
        axes[0].plot(
            rows["k"],
            rows["empirical_flip_probability"],
            marker="o",
            color=color,
            label=f"Empirical {direction}",
        )
        axes[0].plot(
            rows["k"],
            rows["independent_flip_probability"],
            linestyle="--",
            color=color,
            label=f"Independent {direction}",
        )
    axes[0].set(
        title="Reference-conditioned flip rate",
        xlabel="Error count k",
        ylabel=r"$P(E_i=1\mid K=k,R_i)$",
    )
    axes[0].legend(fontsize=7)

    axes[1].plot(
        contrast["k"],
        contrast["empirical_contrast"],
        marker="o",
        label="Empirical",
    )
    axes[1].plot(
        contrast["k"],
        contrast["independent_contrast"],
        marker="o",
        linestyle="--",
        label="Independent",
    )
    axes[1].axhline(0, color="black", linestyle=":")
    axes[1].set(
        title="Polarity contrast",
        xlabel="Error count k",
        ylabel="1→0 minus 0→1 flip rate",
    )
    axes[1].legend(fontsize=8)

    axes[2].plot(
        contrast["k"],
        contrast["residual_contrast"],
        marker="o",
        color="tab:purple",
    )
    axes[2].fill_between(
        contrast["k"],
        contrast["residual_q25"],
        contrast["residual_q75"],
        color="tab:purple",
        alpha=0.18,
    )
    axes[2].axhline(0, color="black", linestyle=":")
    axes[2].set(
        title="Residual polarity contrast",
        xlabel="Error count k",
        ylabel="Empirical − independent contrast",
    )

    axes[3].barh(
        np.arange(support.height),
        support["device_support"],
        color="tab:blue",
    )
    step = max(1, support.height // 8)
    positions = np.arange(0, support.height, step)
    axes[3].set(
        title=f"Support ≥ {minimum_devices}",
        xlabel="Devices",
        ylabel="Error count k",
        yticks=positions,
        yticklabels=support["k"].to_numpy()[positions],
    )
    fig.suptitle(f"{family}: reference-polarity conditional errors")
    fig.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def plot_reference_polarity_family_summary(
    summary: pl.DataFrame,
    output_path: Path,
) -> None:
    """Plot raw and residual polarity contrasts across families."""
    summary = summary.sort("mean_residual_transition_contrast")
    families = summary["family"].to_list()
    positions = np.arange(len(families))
    fig, axes = plt.subplots(1, 2, figsize=(13, max(5, 0.5 * len(families))))
    for ax, value, lower, upper, title in (
        (
            axes[0],
            "mean_raw_transition_contrast",
            "raw_bootstrap_q025",
            "raw_bootstrap_q975",
            "Raw transition asymmetry",
        ),
        (
            axes[1],
            "mean_residual_transition_contrast",
            "residual_bootstrap_q025",
            "residual_bootstrap_q975",
            "Residual conditional asymmetry",
        ),
    ):
        values = summary[value].to_numpy()
        errors = np.vstack(
            [values - summary[lower].to_numpy(), summary[upper].to_numpy() - values]
        )
        ax.errorbar(
            values,
            positions,
            xerr=errors,
            fmt="o",
            color="tab:blue",
            capsize=3,
        )
        ax.axvline(0, color="black", linestyle=":")
        ax.set(
            title=title,
            xlabel="1→0 minus 0→1 flip probability",
            yticks=positions,
            yticklabels=families,
        )
    fig.suptitle("Reference-polarity error asymmetry by PUF family")
    fig.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def write_family_summary(metrics: pl.DataFrame, output_path: Path) -> None:
    """Write device-level quartiles for each family."""
    metric_columns = (
        "mean_error_count",
        "independent_mean_error_count",
        "count_cdf_distance",
        "count_total_variation",
        "conditional_residual_distance",
        "conditional_residual_mae",
        "shrinkage_strength_mean",
        "family_prior_mean_error_count",
    )
    expressions = [pl.len().alias("n_devices")]
    for column in metric_columns:
        expressions.extend(
            [
                pl.col(column).quantile(0.25).alias(f"{column}_q25"),
                pl.col(column).median().alias(f"{column}_median"),
                pl.col(column).quantile(0.75).alias(f"{column}_q75"),
            ]
        )
    metrics.group_by("family").agg(expressions).sort("family").write_csv(output_path)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, default=Path("data/error_structure"))
    parser.add_argument(
        "--output-dir", type=Path, default=Path("figures/error_structure")
    )
    parser.add_argument("--minimum-devices", type=int, default=3)
    parser.add_argument("--minimum-device-fraction", type=float, default=0.10)
    parser.add_argument("--log-level", default="INFO")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    logging.basicConfig(
        level=getattr(logging, args.log_level.upper()),
        format="%(levelname)s %(name)s: %(message)s",
    )
    if args.minimum_devices < 1:
        raise ValueError("minimum-devices must be positive")
    if not 0 <= args.minimum_device_fraction <= 1:
        raise ValueError("minimum-device-fraction must lie in [0, 1]")
    counts = pl.read_parquet(args.input_dir / "count_distributions.parquet")
    conditionals = pl.read_parquet(args.input_dir / "conditional_probabilities.parquet")
    metrics = pl.read_parquet(args.input_dir / "device_metrics.parquet")
    survival = pl.read_parquet(args.input_dir / "population_survival.parquet")
    coverage = pl.read_parquet(args.input_dir / "device_coverage.parquet")
    polarity = pl.read_parquet(
        args.input_dir / "reference_polarity_conditionals.parquet"
    )
    polarity_summary = pl.read_csv(
        args.input_dir / "reference_polarity_family_summary.csv"
    )
    families = metrics["family"].unique().sort().to_list()
    summaries = {
        family: conditional_family_summary(conditionals, family) for family in families
    }
    ecc_dir = args.output_dir / "ecc_capacity"
    conditional_dir = args.output_dir / "conditional_residuals"
    survival_dir = args.output_dir / "population_survival"
    coverage_dir = args.output_dir / "device_coverage"
    polarity_dir = args.output_dir / "reference_polarity"
    ecc_dir.mkdir(parents=True, exist_ok=True)
    conditional_dir.mkdir(parents=True, exist_ok=True)
    survival_dir.mkdir(parents=True, exist_ok=True)
    coverage_dir.mkdir(parents=True, exist_ok=True)
    polarity_dir.mkdir(parents=True, exist_ok=True)
    for family in families:
        total_devices = metrics.filter(pl.col("family") == family).height
        minimum_devices = max(
            args.minimum_devices,
            int(np.ceil(args.minimum_device_fraction * total_devices)),
        )
        plot_ecc_capacity_curves(counts, family, ecc_dir / f"{family}.png")
        plot_population_survival(
            survival,
            family,
            survival_dir / f"{family}.png",
        )
        plot_device_coverage(
            coverage,
            family,
            coverage_dir / f"{family}.png",
        )
        plot_conditional_residual_heatmaps(
            summaries[family],
            family,
            conditional_dir / f"{family}.png",
            minimum_devices=minimum_devices,
            total_devices=total_devices,
        )
        plot_reference_polarity_landscape(
            polarity,
            family,
            polarity_dir / f"{family}.png",
            minimum_devices=minimum_devices,
        )
    plot_device_heterogeneity(metrics, args.output_dir / "device_heterogeneity.png")
    plot_reference_polarity_family_summary(
        polarity_summary,
        args.output_dir / "reference_polarity_family_summary.png",
    )
    write_family_summary(metrics, args.input_dir / "family_summary.csv")
    LOGGER.info("Wrote figures to %s", args.output_dir)


if __name__ == "__main__":
    main()
