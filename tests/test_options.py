from pathlib import Path

import numpy as np
import pytest
import torch

import MouseArmTransformer
from MouseArmTransformer import checkpoint, helper, windows
from MouseArmTransformer.model import SimpleTransformer

WEIGHTS_DIR = Path(MouseArmTransformer.__file__).parent / "weights"
PUBLISHED = sorted(p.name for p in WEIGHTS_DIR.glob("*.pt"))
J = 32


def _x(b=4, t=7, seed=0):
    g = torch.Generator().manual_seed(seed)
    return torch.rand(b, t, J, 2, generator=g) * 300


@pytest.mark.parametrize("weights", PUBLISHED)
def test_default_model_is_the_published_architecture(weights):
    """Defaults add no parameters/buffers and no extra computation."""
    state = torch.load(WEIGHTS_DIR / weights, map_location="cpu", weights_only=True)
    model = SimpleTransformer(num_joints=J)
    assert set(model.state_dict()) == set(state)
    model.load_state_dict(state)
    model.eval()
    x = _x()
    with torch.no_grad():
        manual = model.fc(model.transformer(x.reshape(4, 7, J * 2))).reshape(4, 7, J, 3)
        torch.testing.assert_close(model(x), manual, rtol=0, atol=0)


@pytest.mark.parametrize(
    "kwargs",
    [
        {},
        {"input_proj": True, "d_model": 128},
        {"input_proj": True, "d_model": 128, "pos_encoding": True, "num_layers": 4,
         "dim_feedforward": 512},
        {"image_size": (300, 250)},
        {"output_norm": True, "output_scale": 10.0},
    ],
)
def test_options_shapes(kwargs):
    model = SimpleTransformer(num_joints=J, **kwargs).eval()
    with torch.no_grad():
        assert model(_x()).shape == (4, 7, J, 3)


def test_d_model_needs_projection():
    with pytest.raises(ValueError):
        SimpleTransformer(num_joints=J, d_model=128)


def test_image_scaling():
    model = SimpleTransformer(num_joints=J, image_size=(300, 250))
    pts = torch.tensor([[150.0, 125.0], [250.0, 125.0], [150.0, 225.0]])
    expected = torch.tensor([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]])
    torch.testing.assert_close(model.scale_input(pts), expected)


def test_output_normalisation():
    torch.manual_seed(0)
    plain = SimpleTransformer(num_joints=J).eval()
    normed = SimpleTransformer(num_joints=J, output_norm=True, output_scale=10.0).eval()
    state = plain.state_dict()
    mean = torch.randn(J, 3)
    normed.load_state_dict({**state, "output_mean": mean})
    x = _x()
    with torch.no_grad():
        torch.testing.assert_close(normed(x), plain(x) * 10.0 + mean)


def test_window_index():
    last = windows.window_index(5, 3, "last")
    np.testing.assert_array_equal(last[0], [0, 0, 0])
    np.testing.assert_array_equal(last[4], [2, 3, 4])
    centre = windows.window_index(5, 3, "center")
    np.testing.assert_array_equal(centre[0], [0, 0, 1])
    np.testing.assert_array_equal(centre[2], [1, 2, 3])
    np.testing.assert_array_equal(centre[4], [3, 4, 4])
    with pytest.raises(ValueError):
        windows.window_index(5, 3, "middle")


def test_predict_independent_of_batch_size():
    model = SimpleTransformer(num_joints=J).eval()
    x = np.random.default_rng(0).uniform(0, 300, (50, J, 2)).astype(np.float32)
    a = windows.predict(model, x, 7, "center", batch_size=7)
    b = windows.predict(model, x, 7, "center", batch_size=4096)
    np.testing.assert_allclose(a, b, atol=1e-5)
    assert a.shape == (50, J, 3)


def test_predict_rejects_nan():
    model = SimpleTransformer(num_joints=J)
    x = np.zeros((10, J, 2), np.float32)
    x[3, 0, 0] = np.nan
    with pytest.raises(ValueError):
        windows.predict(model, x, 7)


def _option_model():
    torch.manual_seed(1)
    model = SimpleTransformer(num_joints=J, input_proj=True, d_model=64, pos_encoding=True,
                              image_size=(300, 250), output_norm=True, output_scale=10.0)
    model.output_mean.copy_(torch.randn(J, 3))
    return model.eval()


def test_checkpoint_round_trip(tmp_path):
    model = _option_model()
    path = tmp_path / "m.pt"
    checkpoint.save_checkpoint(path, model, seq_length=9, target="center", extra={"note": "x"})
    loaded, meta = checkpoint.load_model(path)
    assert meta == {"seq_length": 9, "target": "center", "extra": {"note": "x"}, "legacy": False}
    x = _x(t=9)
    with torch.no_grad():
        torch.testing.assert_close(loaded(x), model(x))


@pytest.mark.parametrize("weights", PUBLISHED)
def test_load_model_reads_published_weights(weights):
    model, meta = checkpoint.load_model(weights)
    assert meta["legacy"] and meta["target"] == "last" and meta["seq_length"] is None
    assert model.num_joints == J


def test_run_inference_with_checkpoint(tmp_path):
    model = _option_model()
    path = tmp_path / "m.pt"
    checkpoint.save_checkpoint(path, model, seq_length=9, target="center")
    rng = np.random.default_rng(2)
    data = {k: rng.uniform(0, 300, (40, 2)) for k in helper.mausspaun_keys}
    out = MouseArmTransformer.run_inference(data, model_weights=str(path))
    assert set(out) == set(data)
    assert all(v.shape == (40, 3) for v in out.values())
    x = np.stack([data[k] for k in helper.mausspaun_keys], 1)
    expected = windows.predict(model, x, 9, "center")
    np.testing.assert_allclose(np.stack([out[k] for k in helper.mausspaun_keys], 1), expected,
                               atol=1e-4)
    with pytest.raises(ValueError):
        MouseArmTransformer.run_inference(data, model_weights=str(path), seq_length=7)
