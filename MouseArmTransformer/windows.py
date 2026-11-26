"""Windows of consecutive frames, and whole-recording prediction for one camera."""

import numpy as np
import torch

TARGETS = ("last", "center")


def target_offset(seq_length, target):
    """Position, inside a window, of the frame the window predicts.

    ``"last"`` (causal: only past frames are used) or ``"center"`` (``seq_length // 2`` past
    and as many future frames; the prediction for a frame is available ``seq_length // 2``
    frames later).
    """
    if target not in TARGETS:
        raise ValueError(f"target must be one of {TARGETS}, got {target!r}")
    return seq_length - 1 if target == "last" else seq_length // 2


def window_index(n_frames, seq_length, target="last"):
    """(n_frames, seq_length) frame indices of the window that predicts each frame.

    Windows that would reach past the start or end of the recording repeat the first or last
    frame.
    """
    rel = target_offset(seq_length, target)
    idx = np.arange(n_frames)[:, None] - rel + np.arange(seq_length)[None, :]
    return np.clip(idx, 0, n_frames - 1)


@torch.no_grad()
def predict(model, x, seq_length, target="last", batch_size=4096, device=None):
    """3D for every frame of one recording.

    Args:
        model: A ``SimpleTransformer``.
        x: (frames, joints, 2) 2D keypoints in pixels, no NaN.
        seq_length: Window length the model was trained with.
        target: ``"last"`` or ``"center"``, as the model was trained.
        batch_size: Windows per forward pass (the result does not depend on it).

    Returns:
        (frames, joints, 3) numpy array, in the model's output units.
    """
    x = np.asarray(x, dtype=np.float32)
    if not np.isfinite(x).all():
        raise ValueError("x contains NaN or inf; fill or drop missing keypoints first.")
    device = device or next(model.parameters()).device
    model.eval()
    idx = window_index(len(x), seq_length, target)
    rel = target_offset(seq_length, target)
    out = np.empty((len(x), x.shape[1], 3), np.float32)
    for i in range(0, len(x), batch_size):
        batch = torch.from_numpy(x[idx[i:i + batch_size]]).to(device)
        out[i:i + batch_size] = model(batch)[:, rel].cpu().numpy()
    return out
