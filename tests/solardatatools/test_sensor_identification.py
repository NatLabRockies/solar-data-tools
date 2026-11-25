import numpy as np
import pandas as pd
import pytest
from sklearn.linear_model import LinearRegression
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

    class RecordingRegression(LinearRegression):
        def fit(self, X, y):
            self.training_samples = X[:, 0]
            return super().fit(X, y)

        def predict(self, X):
            observed_splits.append((self.training_samples, X[:, 0]))
            return super().predict(X)

    monkeypatch.setattr(sensor_identification, "LinearRegression", RecordingRegression)
    identifier = sensor_identification.SensorIdentification(handler)
    identifier.identify(n_splits=2)

    observed = iter(observed_splits)
    within_day_split = False
    for days in (flags.no_errors, flags.clear, flags.cloudy):
        retained = np.repeat(days, samples_per_day) & available
        expected_times = timestamps[retained]
        folds = list(TimeSeriesSplit(n_splits=2).split(expected_times))
        for offset in sensor_offsets.values():
            for train_ix, validation_ix in folds:
                train, validation = next(observed)
                train_times = timestamps[(train - offset).astype(int)]
                validation_times = timestamps[(validation - offset).astype(int)]
                assert train_times.max() < validation_times.min()
                within_day_split |= (
                    train_times[-1].normalize() == validation_times[0].normalize()
                )
                assert train_times.equals(expected_times[train_ix])
                assert validation_times.equals(expected_times[validation_ix])

    assert next(observed, None) is None
    assert within_day_split
    np.testing.assert_allclose(identifier.results_table["cv-rmse"], 0, atol=1e-10)


def _make_handler(clear_days):
    samples = np.arange(4 * len(clear_days), dtype=float).reshape(4, -1, order="F")
    handler = DataHandler()
    handler.filled_data_matrix = 4 * samples + 11
    handler.extra_matrices = {
        "east": samples + 0.1 * np.sin(samples),
        "west": samples + 100,
    }
    handler.daily_flags.no_errors = np.ones(len(clear_days), dtype=bool)
    handler.daily_flags.flag_clear_cloudy(np.array(clear_days))
    return handler


@pytest.mark.parametrize("available", ["clear", "cloudy"])
def test_identify_keeps_rankings_when_a_day_subset_is_empty(available):
    handler = _make_handler([available == "clear"] * 3)
    identifier = sensor_identification.SensorIdentification(handler)
    identifier.identify(n_splits=2)

    missing = "cloudy" if available == "clear" else "clear"
    assert set(identifier.results_table["filter"]) == {"no_errors", available}
    assert identifier.chosen_sensor == {"no_errors": "west", available: "west"}
    assert identifier.consistent_answer is None
    assert identifier.coverage_complete is False
    coverage = identifier.validation_coverage[missing]
    assert coverage["status"] == "unavailable"
    assert coverage["n_samples"] == 0
    assert coverage["planned_folds"] == 0
    assert "At least 3" in coverage["reason"]


def test_identify_cannot_split_two_common_observations():
    handler = _make_handler([True])
    handler.extra_matrices["east"][2:, :] = np.nan
    identifier = sensor_identification.SensorIdentification(handler)
    identifier.identify(n_splits=2)

    assert identifier.results_table.empty
    assert identifier.chosen_sensor is None
    assert identifier.consistent_answer is None
    assert identifier.coverage_complete is False
    assert identifier.validation_coverage["no_errors"]["n_samples"] == 2
    assert identifier.validation_coverage["no_errors"]["status"] == "unavailable"
    assert identifier.validation_coverage["clear"]["planned_folds"] == 0


@pytest.mark.parametrize("failure", ["fit", "predict", "nonfinite"])
def test_identify_requires_each_fold_and_selects_sole_eligible_sensor(
    monkeypatch, failure
):
    class SometimesFails(LinearRegression):
        def fit(self, X, y):
            self.first_clear_fold_for_east = X[0, 0] == 0 and len(X) == 4
            if self.first_clear_fold_for_east and failure == "fit":
                raise RuntimeError("Forced fit failure")
            return super().fit(X, y)

        def predict(self, X):
            if self.first_clear_fold_for_east and failure == "predict":
                raise RuntimeError("Forced prediction failure")
            if self.first_clear_fold_for_east and failure == "nonfinite":
                return np.full(len(X), np.inf)
            return super().predict(X)

    monkeypatch.setattr(sensor_identification, "LinearRegression", SometimesFails)
    handler = _make_handler([True] * 3 + [False] * 3)
    identifier = sensor_identification.SensorIdentification(handler)
    identifier.identify(n_splits=2)

    assert identifier.chosen_sensor == dict.fromkeys(
        ("no_errors", "clear", "cloudy"), "west"
    )
    assert identifier.consistent_answer is None
    assert identifier.coverage_complete is False
    clear = identifier.results_table.query("filter == 'clear'")
    assert clear["sensor"].tolist() == ["west"]
    assert clear["cv-rmse"].iloc[0] == pytest.approx(0)
    coverage = identifier.validation_coverage["clear"]
    assert coverage["status"] == "partial"
    assert coverage["planned_folds"] == 2
    assert coverage["planned_validation_samples"] == 8
    east, west = (coverage["sensors"][name] for name in ("east", "west"))
    assert east["completed_folds"] == 1
    assert east["validation_samples"] == 4
    assert set(east["failed_folds"]) == {1}
    assert failure in east["failed_folds"][1].lower()
    assert west["completed_folds"] == 2
    assert west["validation_samples"] == 8
    assert not west["failed_folds"]
    assert identifier.validation_coverage["cloudy"]["status"] == "complete"
    assert identifier.validation_coverage["no_errors"]["status"] == "complete"


def test_identify_reports_disagreement_when_one_subset_has_no_eligible_sensor(
    monkeypatch,
):
    class FailsOnFirstClearFold(LinearRegression):
        def fit(self, X, y):
            if y[0] == 11 and len(X) == 4:
                raise RuntimeError("Forced clear fold failure")
            return super().fit(X, y)

    monkeypatch.setattr(
        sensor_identification, "LinearRegression", FailsOnFirstClearFold
    )
    handler = _make_handler([True] * 3 + [False] * 3)
    handler.extra_matrices["west"][:, :3] += np.array([[1], [-1], [1], [-1]])
    identifier = sensor_identification.SensorIdentification(handler)
    identifier.identify(n_splits=2)

    assert identifier.chosen_sensor == {"no_errors": "east", "cloudy": "west"}
    assert identifier.consistent_answer is False
    assert identifier.coverage_complete is False
    assert set(identifier.results_table["filter"]) == {"no_errors", "cloudy"}
    assert identifier.validation_coverage["clear"]["status"] == "unavailable"
    assert set(identifier.validation_coverage["clear"]["sensors"]) == {"east", "west"}


def test_identify_resets_selection_when_all_fits_fail(monkeypatch):
    handler = _make_handler([True] * 3 + [False] * 3)
    identifier = sensor_identification.SensorIdentification(handler)
    identifier.identify(n_splits=2)
    assert identifier.chosen_sensor == "west"
    assert identifier.consistent_answer is True
    assert identifier.coverage_complete is True

    class AlwaysFails(LinearRegression):
        def fit(self, X, y):
            raise RuntimeError("Forced fit failure")

    monkeypatch.setattr(sensor_identification, "LinearRegression", AlwaysFails)
    identifier.identify(n_splits=2)

    assert identifier.results_table.empty
    assert identifier.chosen_sensor is None
    assert identifier.consistent_answer is None
    assert identifier.coverage_complete is False
    assert all(
        subset["status"] == "unavailable"
        for subset in identifier.validation_coverage.values()
    )


def test_identify_refreshes_common_observations_after_adding_a_sensor():
    handler = _make_handler([True] * 3 + [False] * 3)
    west = handler.extra_matrices.pop("west").ravel(order="F").copy()
    west[[1, 13]] = np.nan
    handler.raw_data_matrix = handler.filled_data_matrix.copy()
    handler.data_frame = pd.DataFrame(
        {"west": west}, index=pd.date_range("2025-01-01", periods=24, freq="6h")
    )
    identifier = sensor_identification.SensorIdentification(handler)
    identifier.identify(n_splits=2)
    assert identifier.chosen_sensor == "east"
    assert identifier.validation_coverage["no_errors"]["n_samples"] == 24

    handler.generate_extra_matrix("west")
    identifier.identify(n_splits=2)
    fresh = sensor_identification.SensorIdentification(handler)
    fresh.identify(n_splits=2)

    assert identifier.validation_coverage["no_errors"]["n_samples"] == 22
    assert identifier.validation_coverage == fresh.validation_coverage
    pd.testing.assert_frame_equal(identifier.results_table, fresh.results_table)
    np.testing.assert_array_equal(identifier.sensor_keys, ["east", "west"])
    assert identifier.chosen_sensor == fresh.chosen_sensor == "west"
    assert identifier.coverage_complete is fresh.coverage_complete is True
