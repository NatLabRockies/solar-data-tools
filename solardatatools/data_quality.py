# -*- coding: utf-8 -*-
"""Data Quality Checking Module

This module contains functions for identifying corrupt or bad quality data.

"""

import numpy as np
from scipy.stats import mode
from solardatatools._time_grid import fixed_duration
from solardatatools.signal_decompositions import tl1_l2d2p365


def make_quality_flags(
    density_scores,
    linearity_scores,
    density_lower_threshold=0.6,
    density_upper_threshold=1.05,
    linearity_threshold=0.1,
):
    density_flags = np.logical_and(
        density_scores > density_lower_threshold,
        density_scores < density_upper_threshold,
    )
    linearity_flags = linearity_scores < linearity_threshold
    return density_flags, linearity_flags


def make_density_scores(
    data_matrix,
    threshold=0.2,
    return_density_signal=False,
    return_fit=False,
    solver=None,
):
    nans = np.isnan(data_matrix)
    capacity_est = np.quantile(data_matrix[~nans], 0.95)
    data_copy = np.copy(data_matrix)
    data_copy[nans] = 0.0
    foo = data_copy > 0.02 * capacity_est
    density_signal = np.sum(foo, axis=0) / data_matrix.shape[0]
    use_days = np.logical_and(density_signal > threshold, density_signal < 0.8)
    fit_signal = tl1_l2d2p365(density_signal, use_ixs=use_days, tau=0.85, solver=solver)
    scores = density_signal / fit_signal
    out = [scores]
    if return_density_signal:
        out.append(density_signal)
    if return_fit:
        out.append(fit_signal)
    if len(out) == 1:
        out = out[0]
    else:
        out = tuple(out)
    return out


def make_linearity_scores(
    data_matrix,
    capacity,
    density_baseline=None,
    *,
    sampling_interval=None,
    daylight_mask=None,
):
    """Return the flagged fraction of daylight and the sample-level infill mask.

    Each column of the filled data matrix is a day on a fixed-step grid.
    Supply a sampling interval with units, such as "1s", and a Boolean
    daylight mask of the same shape. Candidate ramps stay in daylight at or
    above 0.5% of capacity. They must have nonzero slope, span at least 30
    seconds, and contain at least two interior samples. Straightness is checked
    against the line through the endpoints using at least double precision.
    Interpolation between equal anchors produces a flat segment and receives
    no flags.

    The sample mask marks only ramp interiors, regardless of the daily
    rejection threshold. The denominator includes all estimated daylight
    slots. Days without daylight, and matrices without a positive finite
    capacity estimate, have unavailable (NaN) scores.

    Calls with only the original three arguments retain the historical modal
    difference score and mask. The density baseline is used for those calls.
    """
    if (
        sampling_interval is None
        and daylight_mask is None
        and density_baseline is not None
    ):
        return _legacy_linearity_scores(data_matrix, capacity, density_baseline)
    if sampling_interval is None or daylight_mask is None:
        raise ValueError("Supply both sampling_interval and daylight_mask.")

    data = np.asarray(data_matrix)
    if data.ndim != 2 or data.dtype.kind not in "fiu":
        raise ValueError("data_matrix must be a two-dimensional real array.")
    daylight = np.asarray(daylight_mask)
    if daylight.dtype != bool or daylight.shape != data.shape:
        raise ValueError("daylight_mask must be Boolean with the shape of data_matrix.")
    interval_ns = fixed_duration(sampling_interval, "sampling_interval").value
    if interval_ns <= 0:
        raise ValueError("sampling_interval must be positive.")

    infill = np.zeros_like(daylight)
    scores = np.full(data.shape[1], np.nan)
    if not np.isfinite(capacity) or capacity <= 0:
        return scores, infill

    span_ns = 30 * 10**9
    # Two anchors and two interior samples span at least three intervals.
    minimum_steps = max(3, (span_ns + interval_ns - 1) // interval_ns)
    if len(data) > minimum_steps:
        for trace, day, flags in zip(data.T, daylight.T, infill.T):
            _flag_straight_ramps(trace, day, capacity, minimum_steps, flags)

    daylight_count = daylight.sum(axis=0)
    np.divide(infill.sum(axis=0), daylight_count, out=scores, where=daylight_count > 0)
    return scores, infill


def _flag_straight_ramps(trace, daylight, capacity, minimum_steps, flags):
    trace = trace.astype(np.result_type(trace.dtype, np.float64), copy=False)
    capacity = trace.dtype.type(capacity)
    eligible = daylight & np.isfinite(trace) & (trace >= 0.005 * capacity)
    values = np.where(eligible, trace, np.nan)
    changes = np.diff(values)
    tolerance = 64 * np.finfo(values.dtype).eps * capacity
    continuous = np.abs(np.diff(changes)) <= tolerance
    edges = np.flatnonzero(np.r_[True, ~continuous, True])
    long_enough = np.diff(edges) >= minimum_steps
    for first, last in zip(edges[:-1][long_enough], edges[1:][long_enough]):
        if abs(values[last] - values[first]) <= tolerance:
            continue
        expected = np.linspace(values[first], values[last], last - first + 1)
        if np.max(np.abs(values[first : last + 1] - expected)) <= tolerance:
            flags[first + 1 : last] = True


def _legacy_linearity_scores(data_matrix, capacity, density_baseline):
    temp_mat = np.copy(data_matrix)
    temp_mat[temp_mat < 0.005 * capacity] = np.nan
    difference_mat = np.round(temp_mat[1:] - temp_mat[:-1], 4)
    modes, counts = mode(difference_mat, axis=0, nan_policy="omit", keepdims=True)
    n = data_matrix.shape[0] - 1
    linearity_scores = np.asarray(counts.data).squeeze() / (n * density_baseline)
    # Label detected infill points with a boolean mask
    infill = np.zeros_like(data_matrix, dtype=bool)
    slct = linearity_scores >= 0.1
    reference_diffs = np.tile(modes[0][slct], (data_matrix.shape[0], 1))
    found_infill = np.logical_or(
        np.isclose(
            np.r_[np.zeros(data_matrix.shape[1]).reshape((1, -1)), difference_mat][
                :, slct
            ],
            reference_diffs,
        ),
        np.isclose(
            np.r_[difference_mat, np.zeros(data_matrix.shape[1]).reshape((1, -1))][
                :, slct
            ],
            reference_diffs,
        ),
    )
    infill[:, slct] = found_infill
    infill_mask = infill
    return linearity_scores, infill_mask


def daily_missing_data_simple(data_matrix, threshold=0.2, return_density_signal=False):
    """
    This function takes a PV power data matrix and returns a boolean array,
    identifying good days. The good days are the ones that are not missing a
    significant amount of data. This assessment is made based on the fraction
    of non-zero and non-NaN values each day. In a typical "good" data set,
    around 40-60% of the measured values each day will be non-zero. The default
    threshold for this function is 20%.

    :param data_matrix: numpy.array, a matrix containing PV power signals
    :param threshold: float, the threshold to identify good days
    :return: a boolean array, with a True if the day passes the test and a
        False if the day fails
    """
    nans = np.isnan(data_matrix)
    capacity_est = np.quantile(data_matrix[~nans], 0.95)
    data_copy = np.copy(data_matrix)
    data_copy[nans] = 0.0
    foo = data_copy > 0.005 * capacity_est
    bar = np.sum(foo, axis=0) / data_matrix.shape[0]
    good_days = bar > threshold
    if return_density_signal:
        return good_days, bar
    else:
        return good_days


def dataset_quality_score(
    data_matrix, threshold=0.2, good_days=None, use_advanced=True
):
    """
    This function scores a complete data set. The score is the fraction of days
    in the data set that pass the missing data test. A score of 1 means all the
    days in the data set pass the test and are not missing data.

    :param data_matrix: numpy.array, a matrix containing PV power signals
    :param threshold: float, the threshold to identify good days
    :return: the score, a float between 0 and 1
    """
    if good_days is None:
        if use_advanced:
            good_days = make_density_scores(data_matrix, threshold=threshold)
        else:
            good_days = daily_missing_data_simple(data_matrix, threshold=threshold)
    score = np.sum(good_days) / data_matrix.shape[1]
    return score
