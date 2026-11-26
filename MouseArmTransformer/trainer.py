"""Train a ``SimpleTransformer`` on your own recordings (2D of one camera -> 3D targets).

Self-contained: it needs only 2D keypoints of the camera to lift and 3D targets (e.g. from
stereo triangulation) for the same frames. ``training.py`` (the original training code) needs
the original data and the ``mausspaun`` package.

Example::

    from MouseArmTransformer import trainer, checkpoint

    # one (x2d, y3d, likelihood) per recording: x2d (frames, 32, 2) DLC pixels, y3d
    # (frames, 32, 3) with NaN where no 3D is known, likelihood (frames, 32); joints in
    # helper.mausspaun_keys order. Points below pcutoff (default 0.4) are filled.
    model, history = trainer.fit(train_sessions, val_sessions, image_size=(640, 480))
    checkpoint.save_checkpoint("my_model.pt", model, seq_length=7, target="last",
                               pcutoff=history[-1]["pcutoff"])
"""

import copy
import math
import time
import warnings

import numpy as np
import torch

from MouseArmTransformer.model import SimpleTransformer
from MouseArmTransformer.preprocessing import DEFAULT_PCUTOFF, fill_low_likelihood
from MouseArmTransformer.windows import target_offset


class _Windows:
    """Fixed-length windows inside recordings (never across two)."""

    def __init__(self, sessions, seq_length, target):
        self.seq = seq_length
        self.rel = target_offset(seq_length, target)
        self.x = np.concatenate([np.asarray(x, np.float32) for x, _ in sessions])
        self.y = np.concatenate([np.asarray(y, np.float32) for _, y in sessions])
        lengths = np.array([len(x) for x, _ in sessions])
        self.start = np.concatenate([[0], np.cumsum(lengths)[:-1]])
        self.end = self.start + lengths
        self.session_of = np.repeat(np.arange(len(sessions)), lengths)

    def targets(self, min_valid, joints=None):
        """Frames that can be predicted: the window fits in its recording and at least
        ``min_valid`` of ``joints`` have a 3D target."""
        y = self.y if joints is None else self.y[:, joints]
        ok = np.isfinite(y).all(-1).sum(-1) >= min_valid
        t = np.arange(len(self.x))
        s = self.session_of
        fits = (t - self.rel >= self.start[s]) & (t - self.rel + self.seq <= self.end[s])
        return np.flatnonzero(ok & fits)

    def index(self, t):
        return np.asarray(t)[:, None] - self.rel + np.arange(self.seq)[None, :]


def _prepare(sessions, name, pcutoff):
    """(x2d, y3d[, likelihood]) per recording -> (x2d, y3d), filled if likelihoods are given.

    Returns the sessions and whether they were filled.
    """
    if not sessions:
        raise ValueError(f"{name} is empty.")
    sizes = {len(s) for s in sessions}
    if sizes not in ({2}, {3}):
        raise ValueError(f"{name}: give (x2d, y3d) for every recording, or (x2d, y3d, "
                         f"likelihood) for every recording.")
    if sizes == {2}:
        return list(sessions), False
    return [(fill_low_likelihood(x, lik, pcutoff)[0], y) for x, y, lik in sessions], True


def _check(sessions, name):
    if not sessions:
        raise ValueError(f"{name} is empty.")
    for i, (x, y) in enumerate(sessions):
        x, y = np.asarray(x), np.asarray(y)
        if x.ndim != 3 or x.shape[-1] != 2 or y.shape != x.shape[:2] + (3,):
            raise ValueError(f"{name}[{i}]: expected x (frames, joints, 2) and y (frames, "
                             f"joints, 3), got {x.shape} and {y.shape}.")
        if not np.isfinite(x).all():
            raise ValueError(f"{name}[{i}]: x contains NaN or inf. Every input keypoint must be "
                             f"a number: fill or replace low-likelihood points first.")


def fit(train,
        val,
        seq_length=7,
        target="last",
        image_size=None,
        model_kwargs=None,
        target_scale=None,
        lr=3e-4,
        weight_decay=1e-4,
        batch_size=512,
        steps=30000,
        warmup=500,
        eval_every=1000,
        patience=6,
        keep_best=True,
        eval_joints=None,
        pcutoff=DEFAULT_PCUTOFF,
        min_valid_joints=6,
        max_val_windows=60000,
        seed=0,
        device=None,
        verbose=True):
    """Train a ``SimpleTransformer`` from scratch.

    Each step draws ``batch_size`` random windows from the training recordings. The loss is
    the mean squared error to the 3D targets (missing targets ignored), after centring on the
    training mean and dividing by ``target_scale``. Learning rate: linear warm-up, then cosine
    decay to 0 over ``steps``. Training stops early when the validation error has not improved
    for ``patience`` evaluations.

    Args:
        train, val: Lists of ``(x2d, y3d, likelihood)`` per recording: x2d (frames,
            joints, 2) DLC keypoints in pixels; y3d (frames, joints, 3), NaN where unknown;
            likelihood (frames, joints). Points below ``pcutoff`` are replaced by
            interpolation (``preprocessing.fill_low_likelihood``). Or ``(x2d, y3d)`` to use
            x2d as given (it must then contain no NaN).
        seq_length, target: Window length and predicted frame (``"last"`` or ``"center"``).
        image_size: ``(width, height)`` to scale inputs inside the model (recommended);
            None = raw pixels, as the published model.
        model_kwargs: Other ``SimpleTransformer`` arguments (e.g. ``input_proj``,
            ``pos_encoding``, ``d_model``, ``num_layers``, ``dim_feedforward``).
        target_scale: Divisor of the centred targets. Default: their standard deviation.
        eval_joints: Joint indices for the validation error (default: all).
        pcutoff: Likelihood threshold for filling (None = no filling); used only when
            likelihoods are given.
        min_valid_joints: A frame is used only if at least this many joints have a target.
        max_val_windows: Validation uses at most this many frames, spread evenly.

    Returns:
        (model, history): the model (best validation weights if ``keep_best``) in eval mode,
        outputs in target units; ``history`` is a list of
        ``{"step", "train_loss", "val_mpjpe"}`` plus a final ``{"best_step",
        "best_val_mpjpe", "seconds", "pcutoff"}`` (``pcutoff`` None if inputs were not
        filled), ready for ``checkpoint.save_checkpoint(..., pcutoff=...)``.
    """
    train, filled = _prepare(train, "train", pcutoff)
    val, val_filled = _prepare(val, "val", pcutoff)
    if filled != val_filled:
        raise ValueError("Give likelihoods for both train and val, or for neither.")
    _check(train, "train")
    _check(val, "val")
    device = torch.device(device) if device else torch.device(
        "cuda" if torch.cuda.is_available() else "cpu")
    num_joints = np.asarray(train[0][0]).shape[1]

    win = _Windows(train, seq_length, target)
    vwin = _Windows(val, seq_length, target)
    train_t = win.targets(min_valid_joints)
    eval_joints = np.arange(num_joints) if eval_joints is None else np.asarray(eval_joints)
    val_t = vwin.targets(min_valid_joints, joints=eval_joints)
    if len(train_t) == 0 or len(val_t) == 0:
        raise ValueError("No frame with enough 3D targets in train or val.")
    if len(val_t) > max_val_windows:
        val_t = val_t[np.linspace(0, len(val_t) - 1, max_val_windows).astype(int)]

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        mean = np.nan_to_num(np.nanmean(win.y, 0)).astype(np.float32)  # (joints, 3)
        if target_scale is None:
            target_scale = float(np.nanstd(win.y - mean))

    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    kwargs = dict(model_kwargs or {})
    kwargs.setdefault("max_seq_length", max(64, seq_length))
    model = SimpleTransformer(num_joints=num_joints,
                              image_size=image_size,
                              output_norm=True,
                              output_scale=target_scale,
                              **kwargs)
    model.output_mean.copy_(torch.from_numpy(mean))
    model.to(device)

    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)

    def lr_at(step):
        if step < warmup:
            return lr * (step + 1) / warmup
        frac = (step - warmup) / max(1, steps - warmup)
        return lr * 0.5 * (1 + math.cos(math.pi * min(frac, 1.0)))

    @torch.no_grad()
    def val_mpjpe():
        model.eval()
        pred = np.empty((len(val_t), num_joints, 3), np.float32)
        for i in range(0, len(val_t), 4096):
            xb = torch.from_numpy(vwin.x[vwin.index(val_t[i:i + 4096])]).to(device)
            pred[i:i + 4096] = model(xb)[:, vwin.rel].cpu().numpy()
        err = np.linalg.norm(pred[:, eval_joints] - vwin.y[val_t][:, eval_joints], axis=-1)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            return float(np.nanmean(err))

    best = (np.inf, copy.deepcopy(model.state_dict()), 0)
    history, running, n_run, bad = [], 0.0, 0, 0
    t0 = time.time()
    for step in range(steps):
        model.train()
        for g in opt.param_groups:
            g["lr"] = lr_at(step)
        idx = win.index(rng.choice(train_t, batch_size))
        xb = torch.from_numpy(win.x[idx]).to(device)
        yb = torch.from_numpy(win.y[idx]).to(device)
        out = model(xb)
        mask = torch.isfinite(yb)
        diff = torch.where(mask, (out - torch.nan_to_num(yb)) / target_scale, torch.zeros_like(out))
        loss = (diff**2).sum() / mask.sum().clamp(min=1)
        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        running += float(loss)
        n_run += 1
        if (step + 1) % eval_every == 0 or step + 1 == steps:
            v = val_mpjpe()
            history.append({"step": step + 1, "train_loss": running / n_run, "val_mpjpe": v})
            running, n_run = 0.0, 0
            if verbose:
                print(f"  step {step + 1:6d}  loss {history[-1]['train_loss']:.4f}  "
                      f"val {v:.4f}  ({time.time() - t0:.0f} s)", flush=True)
            if v < best[0]:
                best, bad = (v, copy.deepcopy(model.state_dict()), step + 1), 0
            else:
                bad += 1
                if bad >= patience:
                    break
    if keep_best:
        model.load_state_dict(best[1])
    model.eval()
    history.append({
        "best_step": best[2],
        "best_val_mpjpe": best[0],
        "seconds": time.time() - t0,
        "pcutoff": pcutoff if filled else None,
    })
    return model, history
