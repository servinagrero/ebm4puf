import csv
import importlib.util
from pathlib import Path

SCRIPT_PATH = Path(__file__).parents[1] / "scripts" / "plot_free_energy.py"
SPEC = importlib.util.spec_from_file_location("plot_free_energy", SCRIPT_PATH)
assert SPEC is not None and SPEC.loader is not None
plotting = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(plotting)


def test_sweep_plot_functions_write_figures(tmp_path):
    family = "TEST"
    authentication = []
    for scenario, auc, score, evidence in (
        ("genuine", float("nan"), 2.0, 1.0),
        ("marginal", 0.5, 1.9, 0.0),
        ("permuted", 0.8, 1.0, -1.0),
    ):
        for group_size in (1, 2):
            authentication.append(
                {
                    "family": family,
                    "training_method": "cd",
                    "cd_k": "1",
                    "scenario": scenario,
                    "group_size": str(group_size),
                    "free_energy_auc": str(auc),
                    "evidence_auc": str(auc),
                    "rbm_score_q25": str(group_size * score - 0.1),
                    "rbm_score_q50": str(group_size * score),
                    "rbm_score_q75": str(group_size * score + 0.1),
                    "centered_evidence_q50": str(group_size * evidence),
                }
            )
    generation = [
        {
            "family": family,
            "training_method": "cd",
            "cd_k": "1",
            "gibbs_steps": str(gibbs_steps),
            "initialization": "empirical",
            "group_size": str(group_size),
            "free_energy_auc": str(0.5 + 0.1 * group_size),
            "evidence_auc": str(0.5 + 0.1 * group_size),
            "error_count_cdf_distance": str(0.2 / gibbs_steps),
            "marginal_mae": str(0.1 / gibbs_steps),
            "normalized_rbm_score_q50": "1.5",
            "centered_evidence_q50": str(0.2 * group_size),
        }
        for gibbs_steps in (1, 2)
        for group_size in (1, 2)
    ]
    error_counts = [
        {
            "family": family,
            "source": source,
            "training_method": "cd" if source == "rbm" else "none",
            "cd_k": str(cd_k),
            "gibbs_steps": str(gibbs_steps),
            "initialization": "empirical" if source == "rbm" else "none",
            "error_count": str(error_count),
            "probability": str(probability),
        }
        for source, cd_k, gibbs_steps, probabilities in (
            ("empirical", 0, 0, [0.5, 0.5]),
            ("independent", 0, 0, [0.6, 0.4]),
            ("rbm", 1, 1, [0.55, 0.45]),
        )
        for error_count, probability in enumerate(probabilities)
    ]
    relaxation = [
        {
            "family": family,
            "training_method": "cd",
            "cd_k": "1",
            "scenario": scenario,
            "relaxation_steps": str(step),
            "mean_transition_hamming_fraction": str(step * gain),
        }
        for scenario, gain in (("genuine", 0.5), ("rbm", 1.0))
        for step in (1, 2)
    ]
    score_tests = [
        {
            "family": family,
            "training_method": "cd",
            "cd_k": "1",
            "scenario": "marginal",
            "score": score,
            "group_size": str(group_size),
            "empirical_auc": str(empirical),
            "gaussian_auc": str(gaussian),
            "initialization": "none",
        }
        for score, empirical, gaussian in (
            ("independent", 0.4, 0.42),
            ("rbm", 0.3, 0.32),
            ("residual", 0.8, 0.78),
        )
        for group_size in (1, 2)
    ]
    required_readouts = [
        {
            "family": family,
            "training_method": "cd",
            "cd_k": "1",
            "scenario": "marginal",
            "score": score,
            "target_auc": str(target_auc),
            "required_readouts": "" if required is None else str(required),
            "initialization": "none",
            "device": str(device),
            "mean_difference": str(mean_difference),
        }
        for device, score, target_auc, required, mean_difference in (
            (0, "independent", 0.8, None, -0.2),
            (0, "independent", 0.9, None, -0.2),
            (1, "independent", 0.8, None, -0.1),
            (1, "independent", 0.9, None, -0.1),
            (0, "residual", 0.8, 4, 0.3),
            (0, "residual", 0.9, 9, 0.3),
            (1, "residual", 0.8, 6, 0.2),
            (1, "residual", 0.9, 12, 0.2),
        )
    ]
    joint_score_tests = [
        {
            "family": family,
            "training_method": "cd",
            "cd_k": "1",
            "scenario": scenario,
            "test": test,
            "group_size": str(group_size),
            "empirical_auc": str(auc),
            "alternative_acceptance_rate": str(1 - auc),
            "initialization": "none",
        }
        for scenario, auc in (("marginal", 0.8), ("uniform", 1.0))
        for test in (
            "marginal_anomaly",
            "residual_anomaly",
            "joint_mahalanobis",
            "marginal_directional",
            "residual_directional",
            "joint_bonferroni",
        )
        for group_size in (1, 2)
    ]
    paths = [tmp_path / f"figure_{index}.png" for index in range(13)]

    plotting.plot_error_count_distribution(
        error_counts, family, "cd", 1, 1, "empirical", paths[0]
    )
    plotting.plot_static_authentication_auc(authentication, family, "cd", 1, paths[1])
    plotting.plot_generation_auc(generation, family, "empirical", paths[2])
    plotting.plot_generation_metric(
        generation, family, "empirical", "marginal_mae", "MAE", paths[3]
    )
    plotting.plot_normalized_scores(
        authentication, generation, family, "cd", 1, "empirical", paths[4]
    )
    plotting.plot_centered_evidence(
        authentication, generation, family, "cd", 1, "empirical", paths[5]
    )
    plotting.plot_relaxation(relaxation, family, "cd", 1, paths[6])
    plotting.plot_gaussian_auc(
        score_tests,
        family,
        "cd",
        1,
        "marginal",
        paths[7],
    )
    plotting.plot_required_readouts(
        required_readouts,
        family,
        "cd",
        1,
        "marginal",
        paths[8],
    )
    plotting.plot_joint_detection(
        joint_score_tests,
        family,
        "cd",
        1,
        "empirical_auc",
        "AUROC",
        paths[9],
    )
    plotting.plot_joint_detection(
        joint_score_tests,
        family,
        "cd",
        1,
        "alternative_acceptance_rate",
        "FAR",
        paths[10],
    )
    plotting.plot_joint_detection(
        joint_score_tests,
        family,
        "cd",
        1,
        "empirical_auc",
        "AUROC",
        paths[11],
        test_labels=plotting.DIRECTIONAL_TEST_LABELS,
    )
    plotting.plot_joint_detection(
        joint_score_tests,
        family,
        "cd",
        1,
        "alternative_acceptance_rate",
        "FAR",
        paths[12],
        test_labels=plotting.DIRECTIONAL_TEST_LABELS,
    )

    assert all(path.is_file() for path in paths)

    required_path = tmp_path / "required_readouts.csv"
    summary_path = tmp_path / "required_readouts_summary.csv"
    with required_path.open("w", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=list(required_readouts[0]),
        )
        writer.writeheader()
        writer.writerows(required_readouts)

    plotting.write_required_readouts_summary(
        required_path,
        summary_path,
    )
    summary = plotting.pl.read_csv(summary_path)
    residual = summary.filter(plotting.pl.col("score") == "residual")

    assert summary_path.is_file()
    assert residual["n_devices"].to_list() == [2, 2]
    assert residual["n_directional"].to_list() == [2, 2]
    assert residual["directional_fraction"].to_list() == [1.0, 1.0]
