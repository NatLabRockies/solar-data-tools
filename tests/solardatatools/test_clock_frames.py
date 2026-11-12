import numpy as np
import pandas as pd
import pytest

from solardatatools import DataHandler
from solardatatools.algorithms import SunriseSunset, TimeShift


@pytest.mark.parametrize("periodic_detector", [False, True])
def test_pipeline_keeps_clock_frames_aligned(monkeypatch, periodic_detector):
    rolls = np.tile([0, 2, -1], 4)
    num_days = len(rolls)
    corrected = np.zeros((24, num_days))
    daylight_power = np.array([100, 250, 450, 700, 900, 1000, 875, 650, 350, 150])
    corrected[7:17] = daylight_power[:, None]
    corrected[11, 1] = corrected[13, 2] = np.nan
    observed = np.column_stack(
        [np.roll(day, -roll) for day, roll in zip(corrected.T, rolls)]
    )
    timestamps = pd.date_range("2024-01-01", periods=observed.size, freq="h")
    power = observed.ravel(order="F")
    dh = DataHandler(
        pd.DataFrame({"power": power, "sensor": 2 * power}, index=timestamps)
    )

    def estimate_daylight(analysis, data, **kwargs):
        producing = data > 0
        analysis.sunrise_estimates = producing.argmax(axis=0)
        analysis.sunset_estimates = 24 - producing[::-1].argmax(axis=0)
        hours = np.arange(24)[:, None]
        analysis.sunup_mask_estimated = (hours >= analysis.sunrise_estimates) & (
            hours < analysis.sunset_estimates
        )

    monkeypatch.setattr(SunriseSunset, "run_optimizer", estimate_daylight)
    monkeypatch.setattr(SunriseSunset, "calculate_times", estimate_daylight)
    monkeypatch.setattr(DataHandler, "clipping_check", lambda self, **kwargs: None)
    monkeypatch.setattr(DataHandler, "capacity_clustering", lambda self, **kwargs: None)
    attempts = []

    def fit_shift(analysis, data, *, periodic_detector, **kwargs):
        attempts.append(periodic_detector)
        analysis.roll_by_index = rolls
        analysis.index_set = np.flatnonzero(np.diff(rolls))
        analysis.corrected_data = analysis.apply_corrections(data)
        # This would have triggered the old automatic retry.
        analysis.s2 = np.ones(num_days)

    monkeypatch.setattr(TimeShift, "run", fit_shift)

    recurrence = {"periodic_detector": True} if periodic_detector else {}
    dh.run_pipeline(
        fix_shifts=True,
        correct_tz=False,
        extra_cols="sensor",
        verbose=False,
        **recurrence,
    )

    assert attempts == [periodic_detector]
    assert dh.time_shifts is True
    np.testing.assert_array_equal(dh.raw_data_matrix, observed)
    np.testing.assert_array_equal(dh.boolean_masks.missing_values, np.isnan(observed))
    np.testing.assert_array_equal(dh.corrected_observed_matrix, corrected)
    np.testing.assert_array_equal(dh.extra_matrices["sensor"], 2 * corrected)
    np.testing.assert_array_equal(
        dh.daytime_analysis.sunrise_estimates, np.full(num_days, 7)
    )
    np.testing.assert_array_equal(
        dh.daytime_analysis.sunset_estimates, np.full(num_days, 17)
    )
    daylight = np.zeros_like(corrected, dtype=bool)
    daylight[7:17] = True
    np.testing.assert_array_equal(dh.boolean_masks.daytime, daylight)
    assert np.isfinite(dh.filled_data_matrix).all()
    np.testing.assert_array_equal(
        dh.filled_data_matrix[~np.isnan(corrected)], corrected[~np.isnan(corrected)]
    )

    def fit_quantiles(estimator, signal):
        daily_signal = signal[1:].reshape(11, num_days, order="F")
        complete_days = ~np.isnan(corrected).any(axis=0)
        np.testing.assert_array_equal(
            daily_signal[:-1, complete_days], corrected[7:17, complete_days]
        )
        assert np.isnan(daily_signal[4, 1])
        assert np.isnan(daily_signal[6, 2])
        # Supply a known fit to verify undilation independently of model fitting.
        daily_fit = np.r_[np.full(10, 500.0), 0.0]
        estimator.fit_quantiles = np.r_[0, np.tile(daily_fit, num_days)][:, None]

    monkeypatch.setattr(
        "solardatatools.algorithms.quantile_estimation.SmoothPeriodicQuantiles.fit",
        fit_quantiles,
    )
    dh.estimate_quantiles(nvals_dil=11, quantile_levels=[0.5])
    quantiles = dh.quantile_object
    np.testing.assert_array_equal(quantiles.sig_original, corrected.ravel(order="F"))
    expected_fit = np.zeros_like(corrected)
    expected_fit[7:17] = 500
    np.testing.assert_array_equal(
        quantiles.quantiles_original[0.5], expected_fit.ravel(order="F")
    )

    dh.apply_time_dilation()
    np.testing.assert_array_equal(
        dh.dilation_object.signal_ori, corrected.ravel(order="F")
    )
    monkeypatch.setattr(
        "solardatatools.data_handler.plot_bundt_cake", lambda data, **kwargs: data
    )
    bundt_data = dh.plot_bundt(aggregate=False, units="W")
    np.testing.assert_array_equal(
        bundt_data[:num_days], dh.sig_dilated[1:].reshape(num_days, 101)
    )

    dh.augment_data_frame(np.isnan(corrected), "corrected_gaps")
    dh.augment_data_frame(
        dh.boolean_masks.missing_values, "observed_gaps", input_clock="observation"
    )
    gap_timestamps = timestamps[np.isnan(power)]
    for key in ("corrected_gaps", "observed_gaps"):
        pd.testing.assert_index_equal(
            dh.data_frame_raw.index[dh.data_frame_raw[key]], gap_timestamps
        )


def test_time_shift_refit_uses_same_input_and_keeps_previous_fit_on_failure(
    monkeypatch,
):
    filled = np.arange(12.0).reshape(6, 2)
    dh = DataHandler()
    dh.filled_data_matrix = filled
    dh.data_clearness_score = 0.0
    dh.daily_flags.no_errors = np.ones(2, dtype=bool)
    inputs = []

    def fit_shift(analysis, data, *, periodic_detector, **kwargs):
        inputs.append(data.copy())
        analysis.roll_by_index = np.array([0, -1 if periodic_detector else 1])

    monkeypatch.setattr(TimeShift, "run", fit_shift)
    dh.auto_fix_time_shifts()
    dh.auto_fix_time_shifts(periodic_detector=True)

    np.testing.assert_array_equal(inputs[0], inputs[1])
    expected = np.column_stack((filled[:, 0], np.roll(filled[:, 1], -1)))
    np.testing.assert_array_equal(dh.filled_data_matrix, expected)
    accepted = dh.time_shift_analysis

    def fail_shift(*args, **kwargs):
        raise RuntimeError("time-shift fit failed")

    monkeypatch.setattr(TimeShift, "run", fail_shift)
    with pytest.raises(RuntimeError, match="time-shift fit failed"):
        dh.auto_fix_time_shifts()
    assert dh.time_shift_analysis is accepted
    assert dh.time_shifts is True
    np.testing.assert_array_equal(dh.filled_data_matrix, expected)
