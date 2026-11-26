import numpy as np
import pytest
import torch

import MouseArmTransformer
from MouseArmTransformer import checkpoint, helper, trainer
from MouseArmTransformer.model import SimpleTransformer
from MouseArmTransformer.preprocessing import DEFAULT_PCUTOFF, fill_low_likelihood

J = 32


def _track(T=40):
    """One joint moving along x = t, with a short gap, a long gap and a 0.8 point."""
    xy = np.zeros((T, 1, 2))
    xy[:, 0, 0] = np.arange(T)
    xy[:, 0, 1] = 5.0
    lik = np.full((T, 1), 0.99)
    lik[5:8], xy[5:8, 0, 0] = 0.3, 100  # gap, DLC guess far off
    lik[15:30], xy[15:30, 0, 0] = 0.3, 200
    lik[33], xy[33, 0, 0] = 0.8, 300
    return xy, lik


def test_default_threshold_is_0_4():
    assert DEFAULT_PCUTOFF == 0.4


def test_interpolates_points_below_threshold():
    xy, lik = _track()
    filled, replaced = fill_low_likelihood(xy, lik, 0.4)
    np.testing.assert_allclose(filled[5:8, 0, 0], [5, 6, 7])
    np.testing.assert_allclose(filled[15:30, 0, 0], np.arange(15, 30))
    assert filled[33, 0, 0] == 300  # 0.8 >= 0.4: kept
    np.testing.assert_array_equal(replaced[:, 0], lik[:, 0] < 0.4)
    filled_95, _ = fill_low_likelihood(xy, lik, 0.95)
    assert filled_95[33, 0, 0] == 33


def test_ends_held_and_never_confident_joint():
    xy, lik = _track()
    lik[:3] = 0.1
    xy[:3, 0, 0] = 50
    filled, _ = fill_low_likelihood(xy, lik, 0.4)
    np.testing.assert_allclose(filled[:3, 0, 0], 3)  # first confident value held
    lik_none = np.zeros_like(lik)
    filled, _ = fill_low_likelihood(xy, lik_none, 0.4)
    np.testing.assert_allclose(filled[:, 0, 0], np.median(xy[:, 0, 0]))
    assert np.isfinite(filled).all()


def test_none_keeps_points_and_range_is_checked():
    xy, lik = _track()
    filled, replaced = fill_low_likelihood(xy, lik, None)
    np.testing.assert_array_equal(filled, xy)
    assert not replaced.any()
    for bad in (-0.1, 1.5):
        with pytest.raises(ValueError):
            fill_low_likelihood(xy, lik, bad)


def _data(T=30, seed=0):
    rng = np.random.default_rng(seed)
    data = {k: rng.uniform(0, 300, (T, 2)) for k in helper.mausspaun_keys}
    lik = {k: rng.uniform(0, 1, T) for k in helper.mausspaun_keys}
    return data, lik


def _filled_dict(data, lik, pcutoff):
    xy = np.stack([data[k] for k in helper.mausspaun_keys], 1)
    l = np.stack([lik[k] for k in helper.mausspaun_keys], 1)
    filled, _ = fill_low_likelihood(xy, l, pcutoff)
    return {k: filled[:, i] for i, k in enumerate(helper.mausspaun_keys)}


def _stack(out):
    return np.stack([out[k] for k in helper.mausspaun_keys], 1)


def test_run_inference_fills_with_published_weights():
    data, lik = _data()
    out = MouseArmTransformer.run_inference(data, likelihood_dict=lik)  # default 0.4
    ref = MouseArmTransformer.run_inference(_filled_dict(data, lik, 0.4))
    np.testing.assert_allclose(_stack(out), _stack(ref), atol=1e-5)
    unfilled = MouseArmTransformer.run_inference(data, likelihood_dict=lik, pcutoff=None)
    np.testing.assert_allclose(_stack(unfilled), _stack(MouseArmTransformer.run_inference(data)),
                               atol=1e-5)


def test_run_inference_uses_the_checkpoint_threshold(tmp_path):
    torch.manual_seed(0)
    model = SimpleTransformer(num_joints=J).eval()
    path = tmp_path / "m.pt"
    checkpoint.save_checkpoint(path, model, seq_length=5, pcutoff=0.7)
    assert checkpoint.load_model(path)[1]["pcutoff"] == 0.7
    data, lik = _data()
    out = MouseArmTransformer.run_inference(data, model_weights=str(path), likelihood_dict=lik)
    ref = MouseArmTransformer.run_inference(_filled_dict(data, lik, 0.7), model_weights=str(path))
    np.testing.assert_allclose(_stack(out), _stack(ref), atol=1e-5)
    with pytest.warns(UserWarning, match="likelihood"):
        MouseArmTransformer.run_inference(data, model_weights=str(path))


def test_fit_fills_when_given_likelihoods():
    rng = np.random.default_rng(0)
    w = rng.normal(0, 0.05, (2, 3))

    def session(seed):
        r = np.random.default_rng(seed)
        x = r.uniform(50, 250, (200, J, 2))
        lik = r.uniform(0, 1, (200, J))
        return x.astype(np.float32), (x @ w).astype(np.float32), lik

    train, val = [session(1), session(2)], [session(3)]
    kw = dict(seq_length=3, steps=5, eval_every=5, batch_size=16, device="cpu", verbose=False)
    _, history = trainer.fit(train, val, **kw)
    assert history[-1]["pcutoff"] == 0.4
    _, history = trainer.fit(train, val, pcutoff=None, **kw)
    assert history[-1]["pcutoff"] is None
    _, history = trainer.fit([t[:2] for t in train], [v[:2] for v in val], **kw)
    assert history[-1]["pcutoff"] is None
    with pytest.raises(ValueError):
        trainer.fit(train, [v[:2] for v in val], **kw)
