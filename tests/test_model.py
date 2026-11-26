from pathlib import Path

import pytest
import torch

import MouseArmTransformer
from MouseArmTransformer.model import SimpleTransformer

WEIGHTS_DIR = Path(MouseArmTransformer.__file__).parent / "weights"
NUM_JOINTS = 32


def _model(seed=0):
    torch.manual_seed(seed)
    return SimpleTransformer(num_joints=NUM_JOINTS).eval()


def _windows(batch, seq_length, seed=1):
    generator = torch.Generator().manual_seed(seed)
    return torch.randn(batch, seq_length, NUM_JOINTS, 2, generator=generator)


def test_window_frames_affect_output():
    """Attention runs over the frames of a window: changing an earlier frame changes the
    prediction for the last frame."""
    model = _model()
    x = _windows(1, 7)
    x_changed = x.clone()
    x_changed[:, 0] += 1.0
    with torch.no_grad():
        last = model(x)[:, -1]
        last_changed = model(x_changed)[:, -1]
    assert not torch.allclose(last, last_changed, atol=1e-5)


def test_output_independent_of_batch():
    """Each window is lifted on its own: the result does not depend on the other windows
    in the batch."""
    model = _model()
    x = _windows(8, 7)
    with torch.no_grad():
        batched = model(x)
        for i in range(len(x)):
            alone = model(x[i:i + 1])
            torch.testing.assert_close(batched[i:i + 1], alone, atol=1e-5, rtol=1e-5)


@pytest.mark.parametrize("weights", sorted(p.name for p in WEIGHTS_DIR.glob("*.pt")))
def test_published_weights_load(weights):
    model = SimpleTransformer(num_joints=NUM_JOINTS)
    state = torch.load(WEIGHTS_DIR / weights, map_location="cpu", weights_only=True)
    model.load_state_dict(state)  # strict: parameter names and shapes unchanged
    model.eval()
    with torch.no_grad():
        out = model(_windows(4, 7) * 50 + 200)
    assert out.shape == (4, 7, NUM_JOINTS, 3)
    assert torch.isfinite(out).all()
