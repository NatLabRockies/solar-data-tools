"""Sensor Identification Module

This module contains a class for choosing which irradiance sensor best
describes a PV power or current data set. We assume a linear model between
irradiance and power/current, and we use k-fold cross validation to assess
which irradiance sensor provides the best predictive power.

Generally speaking, we can try to assess a sensor's distance from an array and
its plane-of-array mismatch. Hopefully, there exists a sensor that is both
close by the array and well aligned; however, this is not always the case.
We use clear sky data to assess POA mismatch and cloudy sky data to assess
distance from array. If there is a discrepancy in which sensor is "best" under
these two data filtering schemes, the algorithm alerts the user.

"""

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression, HuberRegressor
from sklearn.model_selection import TimeSeriesSplit


def rmse(residuals):
    return np.sqrt(np.average(np.power(residuals, 2)))


def _score_folds(estimator, data, response, folds):
    residuals = []
    failed_folds = {}
    for fold, (train_ix, test_ix) in enumerate(folds, start=1):
        try:
            fit = estimator.fit(data[train_ix], response[train_ix])
            prediction = fit.predict(data[test_ix])
        except Exception as exc:
            failed_folds[fold] = f"{type(exc).__name__}: {exc}"
            continue
        with np.errstate(over="ignore", invalid="ignore"):
            error = response[test_ix] - prediction
        if np.isfinite(error).all():
            residuals.append(error)
        else:
            failed_folds[fold] = "Nonfinite validation residuals."

    coverage = {
        "completed_folds": len(residuals),
        "validation_samples": sum(error.size for error in residuals),
        "failed_folds": failed_folds,
        "reason": None,
    }
    if failed_folds:
        return None, coverage

    errors = np.concatenate(residuals)
    with np.errstate(over="ignore", invalid="ignore"):
        scores = rmse(errors), np.mean(errors)
    if not np.isfinite(scores).all():
        coverage["reason"] = "Nonfinite aggregate validation scores."
        return None, coverage
    return scores, coverage


class SensorIdentification:
    def __init__(self, data_handler_obj):
        """
        This class is always instantiated with a DataHandler class instance.
        This instance should have the pipeline run, with extra columns included
        that identify the candidate sensors.

        :param data_handler_obj: (required) a DataHandler class instance
        """
        self.data_handler = data_handler_obj
        candidates = self.data_handler.extra_matrices
        self._refresh_candidates(candidates)
        if not candidates:
            print("Please add sensor columns to DataHandler pipeline.")
        # These attributes are set when running the identify method
        self.results_table = None
        self.chosen_sensor = None
        self.consistent_answer = None
        self.coverage_complete = None
        self.validation_coverage = None

    def _refresh_candidates(self, candidates):
        self.sensor_keys = np.array(list(candidates))
        self.coverage_scores = self.data_handler.extra_quality_scores
        self.compare_mask = (
            np.all([~np.isnan(matrix) for matrix in candidates.values()], axis=0)
            if candidates
            else None
        )

    def identify(
        self,
        n_splits=20,
        model="least-squares",
        epsilon=1.35,
        max_iter=100,
        alpha=0.0001,
    ):
        """Rank sensors by cross-validation RMSE within each day subset.

        Each eligible sensor completes the same planned folds within a subset.
        Inspect ``validation_coverage`` for unavailable subsets and failed
        folds, and ``coverage_complete`` for overall coverage. Incomplete
        coverage yields ``consistent_answer=None`` unless the available subset
        selections disagree, in which case it is ``False``.
        """
        self.results_table = None
        self.chosen_sensor = None
        self.consistent_answer = None
        self.coverage_complete = None
        self.validation_coverage = None
        # least squares is about 10x faster than huber, but less robust
        if model == "least-squares":
            lr = LinearRegression()
        elif model == "huber":
            lr = HuberRegressor(epsilon=epsilon, max_iter=max_iter, alpha=alpha)
        else:
            raise ValueError(f"Unknown sensor identification model: {model!r}.")
        splitter = TimeSeriesSplit(n_splits=n_splits)
        candidates = self.data_handler.extra_matrices.copy()
        self._refresh_candidates(candidates)
        daily_flags = self.data_handler.daily_flags
        filters = {
            name: getattr(daily_flags, name, None)
            for name in ("no_errors", "clear", "cloudy")
        }
        power = self.data_handler.filled_data_matrix
        rows = []
        validation_coverage = {}
        for filter_key, days in filters.items():
            subset_coverage = {
                "status": "unavailable",
                "reason": None,
                "n_samples": None,
                "planned_folds": 0,
                "planned_validation_samples": 0,
                "sensors": {},
            }
            validation_coverage[filter_key] = subset_coverage
            if not candidates:
                subset_coverage["reason"] = "No candidate sensors available."
                continue
            if days is None or power is None:
                subset_coverage["reason"] = "Day flags or filled response unavailable."
                continue

            mask = np.zeros_like(self.compare_mask, dtype=bool)
            mask[:, days] = self.compare_mask[:, days]
            # Read each day in order before splitting by sample.
            chronological_mask = mask.ravel(order="F")
            y = power.ravel(order="F")[chronological_mask]
            subset_coverage["n_samples"] = y.size
            if y.size <= splitter.n_splits:
                subset_coverage["reason"] = (
                    f"{y.size} common observations for {splitter.n_splits} folds. "
                    f"At least {splitter.n_splits + 1} observations are required."
                )
                continue

            folds = list(splitter.split(y))
            subset_coverage["planned_folds"] = len(folds)
            subset_coverage["planned_validation_samples"] = sum(
                test_ix.size for _, test_ix in folds
            )
            eligible = 0
            for key, matrix in candidates.items():
                data = matrix.ravel(order="F")[chronological_mask].reshape(-1, 1)
                scores, sensor_coverage = _score_folds(lr, data, y, folds)
                subset_coverage["sensors"][key] = sensor_coverage
                if scores is None:
                    continue
                with np.errstate(divide="ignore", invalid="ignore"):
                    corr = np.corrcoef(data[:, 0], y)[0, 1]
                rows.append((key, filter_key, corr, *scores))
                eligible += 1

            if eligible == len(candidates):
                subset_coverage["status"] = "complete"
            elif eligible:
                subset_coverage["status"] = "partial"
                subset_coverage["reason"] = (
                    "Some candidates lack valid scores across all planned folds."
                )
            else:
                subset_coverage["reason"] = (
                    "No candidates have valid scores across all planned folds."
                )

        self.validation_coverage = validation_coverage
        self.coverage_complete = all(
            subset["status"] == "complete" for subset in validation_coverage.values()
        )
        results = pd.DataFrame(
            rows, columns=["sensor", "filter", "corr", "cv-rmse", "cv-mbe"]
        )
        self.results_table = results
        if results.empty:
            return

        winner_indices = results.groupby("filter")["cv-rmse"].idxmin()
        winners = results.loc[winner_indices].set_index("filter")["sensor"].to_dict()
        self.chosen_sensor = winners
        if len(set(winners.values())) > 1:
            self.consistent_answer = False
        elif self.coverage_complete:
            self.chosen_sensor = next(iter(winners.values()))
            self.consistent_answer = True
