import numpy as np
import pytest

from ebm4puf.rbm_cd import RBMPUFModel
from ebm4puf.rbm_residual import ResidualRBMPUFModel


def test_free_energy_and_score_samples_have_opposite_signs():
    samples = np.array([[0, 0, 0], [1, 0, 1], [0, 1, 0], [1, 1, 1]], dtype=np.uint8)
    model = RBMPUFModel(
        n_hidden=2,
        epochs=1,
        batch_size=2,
        random_state=7,
        verbose=False,
    ).fit(samples)

    energy = model.free_energy(samples)

    assert energy.shape == (len(samples),)
    np.testing.assert_allclose(model.score_samples(samples), -energy)


def test_fit_rejects_nonbinary_input():
    with pytest.raises(ValueError, match="binary"):
        RBMPUFModel(epochs=1, verbose=False).fit(np.array([[0.0, 0.5]]))


def test_gibbs_chain_records_requested_checkpoints_deterministically():
    samples = np.array([[0, 0], [1, 1], [0, 1], [1, 0]], dtype=np.uint8)
    model = RBMPUFModel(n_hidden=2, epochs=1, random_state=3, verbose=False).fit(
        samples
    )

    first = model.gibbs_chain(samples, checkpoints=[3, 1], seed=9)
    second = model.gibbs_chain(samples, checkpoints=[1, 3], seed=9)

    assert list(first) == [1, 3]
    for step in first:
        np.testing.assert_array_equal(first[step][0], second[step][0])
        np.testing.assert_array_equal(first[step][1], second[step][1])


def test_pcd_with_marginal_bias_initialization_fits_binary_data():
    samples = np.array([[0, 0], [0, 1], [1, 1], [0, 1]], dtype=np.uint8)
    model = RBMPUFModel(
        n_hidden=2,
        epochs=1,
        batch_size=3,
        random_state=5,
        verbose=False,
        negative_phase="pcd",
        visible_bias_init="marginal",
    ).fit(samples)

    assert model.score_samples(samples).shape == (len(samples),)
    assert np.isfinite(model.model.v_bias.detach().cpu().numpy()).all()


def test_residual_rbm_freezes_smoothed_marginal_logits():
    samples = np.array(
        [[0, 0], [0, 1], [1, 1], [0, 1]],
        dtype=np.uint8,
    )
    model = ResidualRBMPUFModel(
        n_hidden=2,
        epochs=2,
        batch_size=2,
        random_state=5,
        verbose=False,
    ).fit(samples)

    expected_marginals = (samples.sum(axis=0) + 0.5) / (len(samples) + 1)
    expected_logits = np.log(expected_marginals / (1 - expected_marginals))
    np.testing.assert_allclose(model.marginals_, expected_marginals)
    np.testing.assert_allclose(
        model.model.v_bias.detach().cpu().numpy(),
        expected_logits,
        rtol=1e-6,
    )
    assert not model.model.v_bias.requires_grad


def test_residual_rbm_combines_base_and_density_ratio_scores():
    samples = np.array(
        [[0, 0], [0, 1], [1, 1], [0, 1]],
        dtype=np.uint8,
    )
    model = ResidualRBMPUFModel(
        n_hidden=2,
        epochs=1,
        batch_size=2,
        random_state=7,
        verbose=False,
    ).fit(samples)

    np.testing.assert_allclose(
        model.score_samples(samples),
        model.base_score_samples(samples) + model.residual_score_samples(samples),
    )
    generated = model.generate(n_samples=5, gibbs_steps=2, seed=3)
    assert generated.shape == (5, 2)
    assert np.all((generated == 0) | (generated == 1))
