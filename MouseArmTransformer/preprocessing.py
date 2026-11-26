"""Preparing DLC keypoints for lifting."""

import numpy as np

#: Default likelihood threshold below which keypoints are replaced (see ``fill_low_likelihood``).
DEFAULT_PCUTOFF = 0.4


def check_pcutoff(pcutoff):
    """``pcutoff`` must be None (no filling) or a number in [0, 1]."""
    if pcutoff is None:
        return None
    pcutoff = float(pcutoff)
    if not 0.0 <= pcutoff <= 1.0:
        raise ValueError(f"pcutoff must be None or between 0 and 1, got {pcutoff}.")
    return pcutoff


def fill_low_likelihood(xy, likelihood, pcutoff=DEFAULT_PCUTOFF):
    """Replace keypoints whose DLC likelihood is below ``pcutoff`` by linear interpolation in time.

    For each joint and coordinate, a point below the threshold gets the value on the straight
    line between the nearest confident frames before and after it. Before the first (after
    the last) confident frame the first (last) confident value is held. A joint with fewer than
    two confident frames gets the median of its positions. Interpolation uses future frames, so
    this is for recorded videos, not for live use.

    Args:
        xy: (frames, joints, 2) keypoints in pixels (DLC's best guess for every point).
        likelihood: (frames, joints) DLC likelihoods.
        pcutoff: Threshold in [0, 1], or None to keep every point as given.

    Returns:
        (filled, replaced): (frames, joints, 2) float array and the (frames, joints) bool mask
        of replaced points.
    """
    xy = np.array(xy, dtype=float)
    likelihood = np.asarray(likelihood, dtype=float)
    if xy.ndim != 3 or xy.shape[-1] != 2 or likelihood.shape != xy.shape[:2]:
        raise ValueError(f"expected xy (frames, joints, 2) and likelihood (frames, joints), got "
                         f"{xy.shape} and {likelihood.shape}.")
    pcutoff = check_pcutoff(pcutoff)
    if pcutoff is None:
        return xy, np.zeros(likelihood.shape, bool)
    bad = ~(likelihood >= pcutoff)  # NaN likelihood counts as not confident
    t = np.arange(len(xy))
    for j in range(xy.shape[1]):
        good = ~bad[:, j]
        for c in range(2):
            if good.sum() >= 2:
                xy[:, j, c] = np.interp(t, t[good], xy[good, j, c])
            else:
                xy[:, j, c] = np.nanmedian(xy[:, j, c])
    return xy, bad
