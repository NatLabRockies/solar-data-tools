from itertools import product

import numpy as np
import pandas as pd
from sklearn.model_selection import TimeSeriesSplit

from solardatatools import DataHandler, sensor_identification


def test_identify_splits_common_observations_in_timestamp_order(monkeypatch):
    samples_per_day = 4
    num_days = 6
    timestamps = pd.date_range(
        "2025-01-01", periods=samples_per_day * num_days, freq="6h"
    )
    sample_numbers = np.arange(len(timestamps), dtype=float)
    sensor_offsets = {"east": 0, "west": len(timestamps)}
    sensors = {name: sample_numbers + offset for name, offset in sensor_offsets.items()}
    sensors["east"][[2, 9]] = np.nan
    sensors["west"][[6, 18, 20]] = np.nan
    available = np.isfinite(sensors["east"]) & np.isfinite(sensors["west"])

    handler = DataHandler()
    shape = (samples_per_day, num_days)
    handler.filled_data_matrix = (4 * sample_numbers + 11).reshape(shape, order="F")
    handler.extra_matrices = {
        name: values.reshape(shape, order="F") for name, values in sensors.items()
    }
    flags = handler.daily_flags
    flags.no_errors = np.array([True, True, True, False, True, True])
    flags.flag_clear_cloudy(np.array([True, False, False, False, True, False]))

    observed_splits = []

    class RecordingTimeSeriesSplit(TimeSeriesSplit):
        def split(self, X, y=None, groups=None):
            folds = list(super().split(X, y, groups))
            observed_splits.append((X[:, 0], folds))
            return iter(folds)

    monkeypatch.setattr(
        sensor_identification, "TimeSeriesSplit", RecordingTimeSeriesSplit
    )
    identifier = sensor_identification.SensorIdentification(handler)
    identifier.identify(n_splits=2)

    day_filters = (flags.no_errors, flags.clear, flags.cloudy)
    within_day_split = False
    for (values, folds), (days, offset) in zip(
        observed_splits, product(day_filters, sensor_offsets.values()), strict=True
    ):
        observed_times = timestamps[(values - offset).astype(int)]
        assert len(folds) == 2
        for train_ix, validation_ix in folds:
            train_end = observed_times[train_ix].max()
            validation_start = observed_times[validation_ix].min()
            assert train_end < validation_start
            within_day_split |= train_end.normalize() == validation_start.normalize()
        retained = np.repeat(days, samples_per_day) & available
        assert observed_times.equals(timestamps[retained])

    assert within_day_split
    np.testing.assert_allclose(identifier.results_table["cv-rmse"], 0, atol=1e-10)
