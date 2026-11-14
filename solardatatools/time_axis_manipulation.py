# -*- coding: utf-8 -*-
"""Data Transforms Module

This module contains functions for transforming PV power data, including time-axis standardization and
2D-array generation

"""

import warnings
import numpy as np
import pandas as pd
from typing import Optional

from solardatatools._time_grid import (
    calendar_grid,
    fixed_duration,
    midnight_distances,
    resolve_cadence,
    wall_clock_index,
)

TZ_LOOKUP = {
    "America/Anchorage": 9,
    "America/Chicago": 6,
    "America/Denver": 7,
    "America/Los_Angeles": 8,
    "America/New_York": 5,
    "America/Phoenix": 7,
    "Pacific/Honolulu": 10,
    "Canada/Central": 6,
}


def make_time_series(
    df,
    return_keys=True,
    localize_time=-8,
    timestamp_key="ts",
    value_key="meas_val_f",
    name_key="meas_name",
    groupby_keys=["site", "sensor"],
    filter_length=200,
):
    """
    Accepts a Pandas data frame extracted from a relational or Cassandra database.
    These queries often result in data with repeated timestamps, as you might
    have multiple columns stacked into rows in the database. Defaults are
    intended to work with GISMo's VADER Cassandra database implementation.

    Returns a data frame with a single timestamp
    index and the data from different systems split into columns.

    :param df: A Pandas data from generated from a query the VADER Cassandra database
    :param return_keys: If true, return the mapping from data column names to site and system ID
    :param localize_time: If non-zero, localize the time stamps. Default is PST or UTC-8
    :param filter_length: The number of non-null data values a single system must have to be included in the output
    :return: A time-series data frame
    """
    # Check to see if there is a reasonable amount of data
    if np.sum(df[value_key].values >= 0) < 24:
        raise ValueError(
            "Insufficient data to run pipeline. Please check your data frame."
        )
    # Make sure that the timestamps are monotonically increasing. There may be
    # missing or repeated time stamps
    df.sort_values(timestamp_key, inplace=True)
    time_index = pd.to_datetime(df[timestamp_key].sort_values())
    time_index = time_index[~time_index.duplicated(keep="first")]
    output = pd.DataFrame(index=time_index)
    site_keys = []
    site_keys_a = site_keys.append
    grouped = df.groupby(groupby_keys)
    keys = grouped.groups.keys()
    counter = 1
    for key in keys:
        df_view = df.loc[grouped.groups[key]]
        ############## data cleaning ####################################
        # df_view = df_view[pd.notnull(df_view[value_key])]               # Drop records with nulls
        df_view.set_index(
            timestamp_key, inplace=True
        )  # Make the timestamp column the index
        df_view.index = pd.to_datetime(df_view.index)
        df_view.sort_index(inplace=True)  # Sort on time
        df_view = df_view[
            ~df_view.index.duplicated(keep="first")
        ]  # Drop duplicate times
        df_view = df_view.reindex(
            index=time_index, method=None
        )  # Match the master index, interp missing
        #################################################################
        meas_name = str(df_view[name_key].iloc[0])
        col_name = meas_name + "_{:02}".format(counter)
        output[col_name] = df_view[value_key]
        if (
            output[col_name].count() > filter_length
        ):  # final filter on low data count relative to time index
            site_keys_a((key, col_name))
            counter += 1
        else:
            del output[col_name]
    if localize_time:
        output.index = output.index + pd.Timedelta(hours=localize_time)  # Localize time

    if return_keys:
        return output, site_keys
    else:
        return output


def standardize_time_axis(
    df,
    timeindex=True,
    power_col=None,
    datetimekey=None,
    correct_tz=None,
    verbose=True,
    *,
    sampling_interval=None,
    snap_tolerance=None,
    return_diagnostics=False,
):
    """
    This function takes in a pandas data frame containing tabular time series
    data, likely generated with a call to pandas.read_csv(). It is assumed that
    each row of the data frame corresponds to a unique date-time, though not
    necessarily on standard intervals. This function will attempt to convert a
    user-specified column containing time stamps to python datetime objects,
    assign this column to the index of the data frame, and then standardize the
    index over time. By standardize, we mean reconstruct the index to be at
    regular intervals, starting at midnight of the first day of the data set.
    This solves a couple common data errors when working with raw data.
    (1) Missing data points from skipped scans in the data acquisition system.
    (2) Time stamps that are at irregular exact times, including fractional
    seconds.

    :param df: A pandas data frame containing the tabular time series data
    :param datetimekey: An optional key corresponding to the name of the column that contains the time stamps
    :param power_col: Retained for compatibility with callers of the former timezone check.
    :param correct_tz: Deprecated. This option no longer applies a timezone correction.
    :param sampling_interval: Exact fixed duration, such as ``"500ms"``. If
                              omitted, resolve the interval from the timestamps.
    :param snap_tolerance: Maximum displacement, inclusive. Defaults to half
                           the sampling interval.
    :param return_diagnostics: Append cadence, rejection and collision counts,
                               and source-row records to the return tuple.
    :return: ``(frame, 0)`` for compatibility, with a third result when
             ``return_diagnostics=True``.
    """
    if correct_tz is not None:
        warnings.warn(
            "correct_tz is deprecated and no longer adjusts timestamps. Omit it.",
            DeprecationWarning,
            stacklevel=2,
        )
    # convert index to timeseries
    df = df.copy()
    if not timeindex:
        try:
            df[datetimekey] = pd.to_datetime(df[datetimekey])
            df.set_index(datetimekey, inplace=True)
        except KeyError:
            time_cols = [
                col for col in df.columns if np.logical_or("Time" in col, "time" in col)
            ]
            key = time_cols[0]
            df[datetimekey] = pd.to_datetime(df[key])
            df.set_index(datetimekey, inplace=True)
    index = wall_clock_index(pd.DatetimeIndex(pd.to_datetime(df.index)))
    valid = index.notna()
    source_rows = np.flatnonzero(valid)
    if not source_rows.size:
        raise ValueError("No valid timestamps to standardize.")
    interval = resolve_cadence(index, sampling_interval)
    if snap_tolerance is None:
        limit = interval.value // 2
    else:
        limit = fixed_duration(snap_tolerance, "snap_tolerance").value
        if limit < 0:
            raise ValueError("snap_tolerance must be nonnegative.")

    targets, distances = midnight_distances(
        index[valid].astype("datetime64[ns]").asi8, interval.value
    )
    accepted = distances <= limit
    if not np.any(accepted):
        raise ValueError("No timestamps within snap_tolerance of the requested grid.")
    candidates = source_rows[accepted]
    targets = targets[accepted]
    # Sort by target, then distance, then original row position.
    order = np.lexsort((candidates, distances[accepted], targets))
    target_times, first, counts = np.unique(
        targets[order], return_index=True, return_counts=True
    )
    winners = candidates[order[first]]
    snapped = pd.to_datetime(target_times)
    start = min(index.min(), snapped[0])
    end = max(index.max(), snapped[-1])
    grid = calendar_grid(start, end, interval)
    frame = df.iloc[winners].set_axis(snapped).reindex(grid)
    rejection_count = len(index) - len(winners)
    collision_count = int(np.count_nonzero(counts > 1))
    if verbose and rejection_count:
        print(
            f"Timestamp snapping rejected {rejection_count} rows. Collision slots: {collision_count}."
        )
    if not return_diagnostics:
        return frame, 0

    loses = np.ones(len(candidates), dtype=bool)
    loses[first] = False
    discarded = candidates[order[loses]]
    collision_records = pd.DataFrame(
        {
            "grid_timestamp": snapped.repeat(counts)[loses],
            "retained_row": np.repeat(winners, counts)[loses],
            "discarded_row": discarded,
        }
    )
    reasons = np.full(len(index), "", dtype=object)
    reasons[~valid] = "missing timestamp"
    reasons[source_rows[~accepted]] = "outside tolerance"
    reasons[discarded] = "collision"
    rejected = np.flatnonzero(reasons != "")
    diagnostics = {
        "sampling_interval": interval,
        "rejection_count": rejection_count,
        "collision_count": collision_count,
        "rejected_rows": pd.DataFrame(
            {
                "source_row": rejected,
                "timestamp": index[rejected],
                "reason": reasons[rejected],
            }
        ),
        "collisions": collision_records,
    }
    return frame, 0, diagnostics


def fix_daylight_savings_with_known_tz(df, tz="America/Los_Angeles", inplace=False):
    index = (
        df.index.tz_localize(tz, nonexistent="NaT", ambiguous="NaT")
        .tz_convert("Etc/GMT+{}".format(TZ_LOOKUP[tz]))
        .tz_localize(None)
    )
    if inplace:
        df.index = index
        return
    else:
        df_out = df.copy()
        df_out.index = index
        return df_out


def remove_index_timezone(df):
    """
    Removes the timezone information from the index of a pandas DataFrame, if
    it is timezone aware. This function was written with ChatGPT and checked by
    Bennet Meyers

    Parameters:
    df (pandas.DataFrame): The DataFrame to modify.

    Returns:
    pandas.DataFrame: The modified DataFrame.
    """
    if isinstance(df.index, pd.DatetimeIndex) and df.index.tzinfo is not None:
        new_index = df.index.tz_localize(None)
        return df.set_index(new_index)
    else:
        return df


def get_index_timezone(df: pd.DataFrame) -> Optional[str]:
    """
    Return the timezone name or offset label of an aware index.
    """
    if isinstance(df.index, pd.DatetimeIndex):
        tzinfo = df.index.tzinfo
        if tzinfo is not None:
            return tzinfo.tzname(None) or str(tzinfo)
    return None
