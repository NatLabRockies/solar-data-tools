"""Omega, the hour angle is estimated as defined on p. 13 in:
Duffie, John A., and William A. Beckman. Solar engineering of thermal
processes. New York: Wiley, 1991."""

import numpy as np
from pvsystemprofiler.utilities.time_convert import clock_to_solar


def calculate_omega(doy, data_sampling, lon, gmt_offset, *, samples_per_day=None):
    """
    :param data_sampling: sampling interval in minutes.
    :param samples_per_day: row count of the handler's daily grid, when available.
    :param lon: longitude in Degrees (float)
    :param doy: day of year (float or array)
    :param gmt_offset: local timezone offset in hours from UTC/GMT (float or int)
    :return: hour angle omega (float or array)
    """
    num_days = len(doy)
    if not np.isfinite(data_sampling) or data_sampling <= 0:
        raise ValueError("data_sampling must be a positive interval in minutes.")
    rows = round(1440 / data_sampling) if samples_per_day is None else samples_per_day
    if not np.isclose(rows * data_sampling, 1440, rtol=1e-12, atol=0):
        raise ValueError("data_sampling and the row count must cover a 24-hour day.")
    minutes_day = 1440 * np.arange(rows) / rows
    minutes_doy = np.tile(minutes_day.reshape(-1, 1), (1, num_days))
    hours_doy_solar = clock_to_solar(minutes_doy, lon, doy, gmt_offset, eot="duffie")
    hours_doy_solar /= 60
    omega = 15 * (hours_doy_solar - 12)
    return omega
