# -*- coding: utf-8 -*-
"""Matrix Embedding Module

This module contains functions for embedding PV power time series data into
a matrix

"""

import numpy as np
import pandas as pd

from solardatatools._time_grid import (
    calendar_grid,
    resolve_cadence,
    samples_per_day,
    wall_clock_index,
)


def find_start_end(arr):
    n = len(arr)
    for i in range(n):
        if not arr[i]:
            break
    for j in range(-1, -(n + 1), -1):
        if not arr[j]:
            break
    j += 1
    if j == 0:
        j = None
    return i, j


def make_2d(
    df,
    key="dc_power",
    trim_start=False,
    trim_end=False,
    return_day_axis=False,
    *,
    sampling_interval=None,
):
    """Embed aligned observations in dated columns on a fixed daily grid.

    Missing slots and partial days are padded with NaN. Use
    `standardize_time_axis` first if the timestamps need snapping. An explicit
    fixed `sampling_interval` can identify the grid when observations are sparse.
    The trim flags drop the first or last day. Fully missing edge days are
    removed for the selected column.
    """
    if df is None:
        return None
    if not isinstance(df.index, pd.DatetimeIndex):
        raise ValueError("make_2d requires a DatetimeIndex.")
    index = wall_clock_index(df.index)
    if (
        index.empty
        or index.hasnans
        or not index.is_unique
        or not index.is_monotonic_increasing
    ):
        raise ValueError(
            "make_2d requires increasing, distinct timestamps. Use standardize_time_axis first."
        )
    interval = resolve_cadence(index, sampling_interval)
    if np.any(index.astype("datetime64[ns]").asi8 % interval.value):
        raise ValueError(
            "make_2d requires midnight-aligned timestamps. Use standardize_time_axis first."
        )

    grid = calendar_grid(index[0], index[-1], interval)
    rows = samples_per_day(interval)
    day_axis = grid[::rows]
    matrix = (
        df[key].set_axis(index).reindex(grid).to_numpy().reshape(rows, -1, order="F")
    )
    trim = slice(int(trim_start), -1 if trim_end else None)
    matrix = matrix[:, trim]
    day_axis = day_axis[trim]
    empty_days = pd.isna(matrix).all(axis=0)
    first, last = find_start_end(empty_days) if np.any(~empty_days) else (0, 0)
    matrix = matrix[:, first:last].copy()
    if return_day_axis:
        return matrix, day_axis[first:last]
    return matrix
