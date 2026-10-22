"""OP-TEMPO pooled pipeline, ported from op_tempo_unified_pipeline_v2.ipynb and
explore_pooling_methods_performance.py, extended to human, AI and human-AI blocks.

Faithful parts (same rules as the dissertation code):
  * quartile symbols per member and sensor (assign_percentile_symbols, spacing 25)
  * seven pooling rules across members within a sensor (pooled_sensor_value)
  * pooled sensor values concatenated into one team-state label per layer
  * Shannon entropy (bits) in 30-s windows, 200-s moving average, drop first 159 s
  * mutual information MI = H(X) + H(Y) - H(X,Y) in bits

Extensions used for the human-AI demo:
  * the AI is pooled the same way, with attention heads as its "members"
  * block-to-block MI between pooled human and pooled AI states ("integration")
  * leave-one-out role-to-system MI (focal person removed from the pool)
  * circular-shift surrogate baseline (keeps each block's autocorrelation,
    breaks only human-AI simultaneity), as in the RQA smoke test
  * entropy and MI windows both end at t (aligned)
"""
from __future__ import annotations

import math
from collections import Counter

import numpy as np
from scipy.ndimage import uniform_filter1d

WINDOW = 30
SMOOTH = 200
REMOVE_INITIAL = 159

METHODS = [
    "high_count_3level", "high_count_exact", "any_high_binary", "majority_high_binary",
    "mean_bin_3level", "median_bin_4level", "mode_bin_4level",
]
METHOD_LABELS = {
    "high_count_3level": "High-count, 3-level",
    "high_count_exact": "Exact high count",
    "any_high_binary": "Any-high binary",
    "majority_high_binary": "Majority-high binary",
    "mean_bin_3level": "Mean-bin, 3-level",
    "median_bin_4level": "Median-bin, 4-level",
    "mode_bin_4level": "Mode-bin, 4-level",
}


# ----------------------------------------------------------------- symbols
def quartile_bins(x, spacing: int = 25) -> np.ndarray:
    """Same rule as assign_percentile_symbols: first j with x < P[j], else last."""
    x = np.nan_to_num(np.asarray(x, dtype=float), nan=0.0, posinf=0.0, neginf=0.0)
    thr = np.percentile(x, np.arange(spacing, 101, spacing))
    b = np.searchsorted(thr, x, side="right")
    return np.minimum(b, len(thr) - 1).astype(np.int8)


# ----------------------------------------------------------------- pooling
def pool(bins: np.ndarray, method: str) -> np.ndarray:
    """bins: [members, T] ints 0..3  ->  pooled value per time point."""
    bins = np.asarray(bins)
    n = bins.shape[0]
    high = (bins >= 2).sum(0)
    if method == "high_count_3level":
        r = high / max(n, 1)
        return np.where(r <= 1 / 3, 0, np.where(r < 2 / 3, 1, 2)).astype(np.int8)
    if method == "high_count_exact":
        return high.astype(np.int8)
    if method == "any_high_binary":
        return (high > 0).astype(np.int8)
    if method == "majority_high_binary":
        return (high > n / 2).astype(np.int8)
    if method == "mean_bin_3level":
        m = bins.mean(0)
        return np.where(m < 1, 0, np.where(m < 2, 1, 2)).astype(np.int8)
    if method == "median_bin_4level":
        return np.clip(np.floor(np.median(bins, 0) + 0.5), 0, 3).astype(np.int8)
    if method == "mode_bin_4level":
        out = np.empty(bins.shape[1], dtype=np.int8)
        for t in range(bins.shape[1]):
            vals = bins[:, t].tolist()
            c = Counter(vals)
            mx = max(c.values())
            modes = [v for v, k in c.items() if k == mx]
            med = float(np.median(vals))
            out[t] = min(modes, key=lambda v: (abs(v - med), v))
        return out
    raise ValueError(method)


def encode(columns: list[np.ndarray]) -> np.ndarray:
    """Concatenate per-sensor symbols into one state label (dense integer id)."""
    M = np.stack([np.asarray(c, dtype=np.int64) for c in columns], 1)
    _, ids = np.unique(M, axis=0, return_inverse=True)
    return ids.reshape(-1).astype(np.int64)


def tuples(columns: list[np.ndarray]) -> np.ndarray:
    return np.stack([np.asarray(c, dtype=np.int64) for c in columns], 1)


# ----------------------------------------------------------------- measures
def _H(counts) -> float:
    c = np.fromiter(counts, dtype=float)
    p = c / c.sum()
    return float(-(p * np.log2(p)).sum())


def entropy(seq) -> float:
    return _H(Counter(seq.tolist() if hasattr(seq, "tolist") else seq).values())


def mi(x, y) -> float:
    x = np.asarray(x, dtype=np.int64); y = np.asarray(y, dtype=np.int64)
    j = x * (int(y.max()) + 1) + y
    return entropy(x) + entropy(y) - entropy(j)


def windowed_entropy(seq, w: int = WINDOW) -> np.ndarray:
    """OP-TEMPO windowed_entropy_forward: value at t uses seq[t-w:t]; w leading zeros."""
    seq = np.asarray(seq)
    n = len(seq)
    out = np.zeros(n)
    for i in range(n - w):
        out[i + w] = _H(Counter(seq[i:i + w].tolist()).values())
    return out


def windowed_mi(x, y, w: int = WINDOW) -> np.ndarray:
    """MI over the same trailing windows as windowed_entropy (w+1 samples ending at t)."""
    x = np.asarray(x, dtype=np.int64); y = np.asarray(y, dtype=np.int64)
    j = x * (int(y.max()) + 1) + y
    n = len(x)
    out = np.zeros(n)
    for t in range(w, n):
        a, b, c = x[t - w:t + 1], y[t - w:t + 1], j[t - w:t + 1]
        out[t] = (_H(Counter(a.tolist()).values()) + _H(Counter(b.tolist()).values())
                  - _H(Counter(c.tolist()).values()))
    return out


def finish(series, smooth: int = SMOOTH, remove_initial: int = REMOVE_INITIAL) -> np.ndarray:
    """OP-TEMPO smooth(): uniform filter, nearest edges; then drop the first samples."""
    s = uniform_filter1d(np.asarray(series, dtype=float), size=smooth, mode="nearest")
    return s[remove_initial:]


def circular_surrogate_mi(x, y, w: int = WINDOW, n_surr: int = 19, min_shift: int = 300, seed: int = 0):
    """Windowed MI after circularly shifting y. Returns [n_surr, n] raw (unsmoothed)."""
    rng = np.random.default_rng(seed)
    n = len(x)
    shifts = rng.integers(min_shift, n - min_shift, n_surr)
    return np.stack([windowed_mi(x, np.roll(y, int(s)), w) for s in shifts]), shifts


def lagged_mi(x, y, lags) -> np.ndarray:
    """MI(x[t], y[t+lag]) over the whole segment (lag > 0: y follows x)."""
    out = []
    for L in lags:
        if L > 0:
            out.append(mi(x[:-L], y[L:]))
        elif L < 0:
            out.append(mi(x[-L:], y[:L]))
        else:
            out.append(mi(x, y))
    return np.array(out)
