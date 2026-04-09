"""Dynamic Time Warping for temporal alignment of feature sequences."""

from __future__ import annotations

from typing import Callable, List, Tuple

import numpy as np


def dtw_align(
    ref: List[FrameFeatures],
    user: List[FrameFeatures],
    local_cost: Callable[[int, int], float],
) -> Tuple[np.ndarray, float]:
    """
    Sakoe–Chiba–style DTW with unit step (i-1,j), (i,j-1), (i-1,j-1).

    ``local_cost(ref_index, user_index)`` should return the cost of pairing
    those frames (pose mismatch plus any small time-sync prior).

    Returns ``(path, total_cost)`` where ``path`` is ``(K, 2)`` int64 pairs
    ``(ref_index, user_index)`` from start to end, and ``total_cost`` is the
    sum of ``local_cost`` along the path.
    """
    n = len(ref)
    m = len(user)
    if n == 0 or m == 0:
        return np.zeros((0, 2), dtype=np.int64), 0.0

    inf = 1e100
    acc = np.full((n, m), inf, dtype=np.float64)
    back = np.zeros((n, m, 2), dtype=np.int16)

    c00 = float(local_cost(0, 0))
    acc[0, 0] = c00

    for i in range(1, n):
        c = float(local_cost(i, 0))
        acc[i, 0] = acc[i - 1, 0] + c
        back[i, 0] = (i - 1, 0)
    for j in range(1, m):
        c = float(local_cost(0, j))
        acc[0, j] = acc[0, j - 1] + c
        back[0, j] = (0, j - 1)

    for i in range(1, n):
        for j in range(1, m):
            c = float(local_cost(i, j))
            opts = [
                (acc[i - 1, j], (i - 1, j)),
                (acc[i, j - 1], (i, j - 1)),
                (acc[i - 1, j - 1], (i - 1, j - 1)),
            ]
            best_val, best_prev = min(opts, key=lambda t: t[0])
            acc[i, j] = best_val + c
            back[i, j] = best_prev

    path_rev: List[Tuple[int, int]] = []
    i, j = n - 1, m - 1
    while i >= 0 and j >= 0:
        path_rev.append((i, j))
        if i == 0 and j == 0:
            break
        pi, pj = int(back[i, j, 0]), int(back[i, j, 1])
        if pi == i and pj == j:
            break
        i, j = pi, pj

    path = np.array(list(reversed(path_rev)), dtype=np.int64)
    total = float(acc[n - 1, m - 1]) if path.size else 0.0
    return path, total
