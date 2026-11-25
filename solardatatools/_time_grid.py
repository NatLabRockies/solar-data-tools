"""Fixed-step local standard-time grids shared by timestamp consumers."""

from itertools import product

import numpy as np
import pandas as pd


DAY = pd.Timedelta(days=1)
# One day in nanoseconds is 2**16 * 3**3 * 5**11.
_TILING_STEPS = np.sort(
    [2**a * 3**b * 5**c for a, b, c in product(range(17), range(4), range(12))]
)


def wall_clock_index(index):
    """Keep local labels and reject an aware index spanning an offset change."""
    if index.tz is None:
        return index
    wall = index.tz_localize(None)
    valid = index.dropna()
    if len(valid):
        utc = valid.tz_convert("UTC")
        dates = pd.date_range(utc.min().ceil("D"), utc.max(), freq="D")
        probes = valid.append(dates.tz_convert(index.tz))
        offsets = probes.tz_localize(None) - probes.tz_convert(None)
        if offsets.min() != offsets.max():
            raise ValueError(
                "Normalize timestamps to local standard time before processing an offset change."
            )
    return pd.DatetimeIndex(wall, freq=index.freq)


def fixed_duration(value, name):
    """Parse an explicitly supplied physical duration."""
    if isinstance(value, (int, float, np.integer, np.floating)) and not isinstance(
        value, np.timedelta64
    ):
        raise ValueError(f"{name} needs a duration with units, such as '500ms'.")
    try:
        interval = pd.Timedelta(value)
        if pd.isna(interval):
            raise ValueError
        interval.value  # Require exact representation at nanosecond precision.
    except (TypeError, ValueError, OverflowError):
        raise ValueError(
            f"{name} must be a fixed duration representable in nanoseconds."
        ) from None
    return interval


def samples_per_day(interval):
    if interval <= pd.Timedelta(0) or DAY.value % interval.value:
        raise ValueError(
            "sampling_interval must be positive and divide a 24-hour day exactly."
        )
    return DAY.value // interval.value


def midnight_distances(timestamps, step):
    """Return nearest ticks and their distances on the arithmetic grid."""
    remainder = timestamps % step
    upper_distance = step - remainder
    displacement = np.where(remainder > upper_distance, upper_distance, -remainder)
    bounds = np.iinfo(np.int64)
    if np.any(timestamps > bounds.max - np.maximum(displacement, 0)) or np.any(
        timestamps <= bounds.min - np.minimum(displacement, 0)
    ):
        raise ValueError(
            "A snapped timestamp lies outside the supported nanosecond range."
        )
    nearest = timestamps + displacement
    return nearest, np.minimum(remainder, upper_distance)


def resolve_cadence(index, sampling_interval=None):
    """Resolve an explicit, declared, or unambiguous inferred cadence."""
    declared = sampling_interval if sampling_interval is not None else index.freq
    if declared is not None:
        interval = fixed_duration(declared, "sampling_interval")
        samples_per_day(interval)
        return interval

    timestamps = index.dropna().unique().sort_values().astype("datetime64[ns]").asi8
    gaps = np.diff(timestamps)
    if gaps.size < 3:
        raise ValueError(
            "Too few distinct timestamps to infer cadence. Specify sampling_interval."
        )
    if np.all(gaps == gaps[0]):
        interval = pd.Timedelta(int(gaps[0]), unit="ns")
        samples_per_day(interval)
        return interval

    median = np.median(gaps)
    candidates = _TILING_STEPS[np.abs(_TILING_STEPS - median) <= 0.02 * median]
    supported = []
    for step in candidates:
        ordinary = np.abs(gaps - step) <= 0.02 * step
        ordinary_count = np.count_nonzero(ordinary)
        if ordinary_count < 3 or 2 * ordinary_count <= gaps.size:
            continue
        phase = timestamps % step
        distance = np.minimum(phase, step - phase)
        if np.median(distance) <= step / 4:
            supported.append((step, ordinary))
    if len(supported) == 1:
        step, ordinary = supported[0]
        # A sustained run of similar nonnominal gaps can be another cadence or
        # systematic missing scans. Either case needs the caller's interval.
        runs = np.lib.stride_tricks.sliding_window_view(gaps, 3)
        minimum = runs.min(axis=1)
        competing = ~(ordinary[:-2] | ordinary[1:-1] | ordinary[2:])
        if not np.any(competing & (runs.max(axis=1) - minimum <= 0.04 * minimum)):
            return pd.Timedelta(int(step), unit="ns")
    raise ValueError(
        "Ambiguous or mixed timestamp cadence. Specify sampling_interval or split the record."
    )


def calendar_grid(first, last, interval):
    """Build all slots from the first through the last calendar date."""
    first_day = first.value // DAY.value
    last_day = last.value // DAY.value
    start = first_day * DAY.value
    end = (last_day + 1) * DAY.value - interval.value
    if start < pd.Timestamp.min.value or end > pd.Timestamp.max.value:
        raise ValueError(
            "The calendar grid lies outside the supported nanosecond range."
        )
    periods = (last_day - first_day + 1) * samples_per_day(interval)
    return pd.date_range(pd.Timestamp(start, unit="ns"), periods=periods, freq=interval)
