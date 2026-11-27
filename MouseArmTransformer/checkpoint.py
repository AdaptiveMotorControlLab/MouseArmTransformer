"""Self-describing model files: weights + the settings needed to use them.

The published weights are bare ``state_dict`` files of the default ``SimpleTransformer``;
``load_model`` reads those too.
"""

from pathlib import Path

import torch

from MouseArmTransformer.model import SimpleTransformer
from MouseArmTransformer.preprocessing import check_pcutoff

FORMAT = "MouseArmTransformer-checkpoint-1"
WEIGHTS_DIR = Path(__file__).parent / "weights"


_UNRECORDED = object()


def save_checkpoint(path, model, seq_length, target="last", pcutoff=_UNRECORDED, extra=None):
    """Save a model with everything needed to run it.

    Args:
        path: Output file (``.pt``).
        model: A ``SimpleTransformer``.
        seq_length: Window length used in training.
        target: ``"last"`` or ``"center"`` (frame of the window that was predicted).
        pcutoff: Likelihood threshold used to fill the training inputs
            (``preprocessing.fill_low_likelihood``; None = not filled). ``run_inference``
            applies the same one when given likelihoods. Leave unset if unknown.
        extra: Optional JSON-like dict (e.g. training settings, data notes).
    """
    blob = {
        "format": FORMAT,
        "model_config": model.model_config(),
        "seq_length": int(seq_length),
        "target": target,
        "state_dict": model.state_dict(),
        "extra": extra or {},
    }
    if pcutoff is not _UNRECORDED:
        blob["pcutoff"] = check_pcutoff(pcutoff)
    torch.save(blob, path)


def resolve(path_or_name):
    """A file path, or the name of a file shipped in ``MouseArmTransformer/weights``."""
    path = Path(path_or_name)
    if path.exists():
        return path
    packaged = WEIGHTS_DIR / str(path_or_name)
    if packaged.exists():
        return packaged
    raise FileNotFoundError(f"No model file {path_or_name!r} (also looked in {WEIGHTS_DIR}).")


def load_model(path_or_name, device="cpu"):
    """Load a model saved with ``save_checkpoint`` or a published (legacy) weights file.

    Returns:
        (model, meta): the model in eval mode on ``device``; ``meta`` has ``seq_length``
        (None for legacy files: they do not record it), ``target``, ``extra``, ``legacy``
        (True for bare state dicts) and, if the file records it, ``pcutoff``.
    """
    blob = torch.load(resolve(path_or_name), map_location="cpu", weights_only=True)
    if isinstance(blob, dict) and blob.get("format") == FORMAT:
        model = SimpleTransformer(**blob["model_config"])
        model.load_state_dict(blob["state_dict"])
        meta = {
            "seq_length": blob["seq_length"],
            "target": blob["target"],
            "extra": blob.get("extra", {}),
            "legacy": False,
        }
        if "pcutoff" in blob:
            meta["pcutoff"] = blob["pcutoff"]
    else:
        num_joints = blob["fc.weight"].shape[0] // 3
        model = SimpleTransformer(num_joints=num_joints)
        model.load_state_dict(blob)
        meta = {"seq_length": None, "target": "last", "extra": {}, "legacy": True}
    return model.to(device).eval(), meta
