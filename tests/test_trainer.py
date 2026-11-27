import numpy as np
import pytest
import torch

from MouseArmTransformer import trainer, windows

J = 32


def _sessions(n_sessions, frames, seed, nan_share=0.2):
    """Random 2D poses in pixels (independent frames); 3D = a fixed linear map of the 2D."""
    rng = np.random.default_rng(seed)
    w = np.random.default_rng(123).normal(0, 0.05, (2, 3))
    out = []
    for _ in range(n_sessions):
        x = rng.uniform(50, 250, (frames, J, 2))
        y = x @ w + 5.0
        y[rng.random((frames, J)) < nan_share] = np.nan
        out.append((x.astype(np.float32), y.astype(np.float32)))
    return out


def _fit(seed=0, steps=600, **kw):
    return trainer.fit(_sessions(3, 400, 0), _sessions(1, 300, 1), seq_length=5, steps=steps,
                       batch_size=64, warmup=20, eval_every=100, patience=10,
                       image_size=(300, 300), seed=seed, device="cpu", verbose=False, **kw)


def test_fit_learns_and_outputs_target_units():
    train, val = _sessions(3, 400, 0), _sessions(1, 300, 1)
    model, history = _fit()
    x, y = val[0]
    pred = windows.predict(model, x, 5, "last")  # outputs must be in target units
    err = np.nanmean(np.linalg.norm(pred - y, axis=-1))
    train_mean = np.nanmean(np.concatenate([t[1] for t in train]), 0)
    base = np.nanmean(np.linalg.norm(train_mean[None] - y, axis=-1))
    assert err < 0.7 * base
    assert history[-1]["best_step"] > 0


def test_fit_is_reproducible():
    a, _ = _fit(seed=3, steps=100)
    b, _ = _fit(seed=3, steps=100)
    for (k, va), vb in zip(a.state_dict().items(), b.state_dict().values()):
        torch.testing.assert_close(va, vb, msg=k)


def test_fit_with_options_and_centre_target():
    model, _ = _fit(target="center", steps=100,
                    model_kwargs={"input_proj": True, "d_model": 32, "pos_encoding": True})
    assert model.model_config()["input_proj"] and model.model_config()["pos_encoding"]


def test_fit_rejects_nan_inputs():
    train = _sessions(1, 100, 0)
    train[0][0][10, 0, 0] = np.nan
    with pytest.raises(ValueError, match="NaN"):
        trainer.fit(train, _sessions(1, 100, 1), steps=1, device="cpu", verbose=False)
