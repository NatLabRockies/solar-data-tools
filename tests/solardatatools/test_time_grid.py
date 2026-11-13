from datetime import timezone, timedelta
import warnings

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest

from pvsystemprofiler.utilities.hour_angle_equation import calculate_omega
from solardatatools import DataHandler, make_2d, standardize_time_axis
from solardatatools.algorithms import SunriseSunset, TimeShift


@pytest.mark.parametrize("step", ["500ms", "5s", "15s", "90s"])
def test_infers_exact_regular_cadence_without_frequency_metadata(step):
    index = pd.DatetimeIndex(
        pd.date_range("2024-01-01", periods=8, freq=step), freq=None
    )
    data = pd.DataFrame({"power": np.arange(8.0)}, index=index)
    grid, offset, diagnostics = standardize_time_axis(data, return_diagnostics=True)
    assert diagnostics["sampling_interval"] == pd.Timedelta(step)
    assert offset == 0
    assert diagnostics["rejection_count"] == diagnostics["collision_count"] == 0
    np.testing.assert_array_equal(grid.loc[index, "power"], data["power"])
    assert grid["power"].notna().sum() == len(index)


def test_jittered_cadence_and_missing_scan():
    regular = pd.date_range("2024-01-01", periods=12, freq="500ms")
    jitter = pd.to_timedelta([0, 1, -1, 2, 1, 0, -2, 0, 1, -1, 2, 1], unit="ms")
    data = pd.DataFrame({"power": np.arange(12.0)}, index=regular + jitter).drop(
        regular[5]
    )
    grid, _ = standardize_time_axis(data)
    assert grid.index.freq == pd.Timedelta("500ms")
    expected = np.arange(12.0)
    expected[5] = np.nan
    np.testing.assert_array_equal(grid.loc[regular, "power"], expected)


def test_declared_cadence_precedence_and_ambiguous_segments():
    index = pd.date_range("2024-01-01", periods=4, freq="10s")
    data = pd.DataFrame({"power": np.arange(4)}, index=index)
    _, _, diagnostics = standardize_time_axis(data.iloc[:1], return_diagnostics=True)
    assert diagnostics["sampling_interval"] == pd.Timedelta("10s")
    grid, _ = standardize_time_axis(data, sampling_interval="5s")
    assert np.isnan(grid.iloc[1, 0])

    seconds = np.r_[np.arange(0, 40, 5), [45, 55, 65, 75]]
    index = pd.Timestamp("2024-01-01") + pd.to_timedelta(seconds, unit="s")
    mixed = pd.DataFrame({"power": np.arange(len(index))}, index=index)
    with pytest.raises(ValueError, match="Ambiguous or mixed"):
        standardize_time_axis(mixed)
    with pytest.raises(ValueError, match="Too few distinct"):
        standardize_time_axis(mixed.iloc[:3])
    grid, _ = standardize_time_axis(mixed, sampling_interval="5s")
    np.testing.assert_array_equal(grid.loc[index, "power"], mixed["power"])
    with pytest.raises(ValueError, match="divide a 24-hour day"):
        standardize_time_axis(mixed, sampling_interval="7min")
    with pytest.raises(ValueError, match="nonnegative"):
        standardize_time_axis(mixed, sampling_interval="5s", snap_tolerance="-1s")


def test_whole_row_winners_ties_and_final_midnight_extent():
    base = pd.Timestamp("2024-01-01")
    index = base + pd.to_timedelta([550, 250, 450, 1000, 1000, 86_399_900], unit="ms")
    index = index.insert(len(index), pd.NaT)
    source = pd.DataFrame(
        {"power": np.arange(7.0), "sensor": [np.nan, 10, 20, 30, 40, 50, 60]},
        index=index,
    )
    grid, _, diagnostics = standardize_time_axis(
        source, sampling_interval="500ms", return_diagnostics=True, verbose=False
    )
    assert grid.index[-1] == base + pd.Timedelta(days=2) - pd.Timedelta("500ms")
    assert grid.loc[base, "power"] == 1  # Exactly halfway, assigned earlier.
    assert grid.loc[base + pd.Timedelta("500ms"), "power"] == 0
    assert np.isnan(grid.loc[base + pd.Timedelta("500ms"), "sensor"])
    assert grid.loc[base + pd.Timedelta("1s"), "power"] == 3
    assert grid.loc[base + pd.Timedelta(days=1), "power"] == 5
    assert grid["power"].notna().sum() == 4
    assert diagnostics["rejection_count"] == 3
    assert diagnostics["collision_count"] == 2
    assert diagnostics["rejected_rows"]["reason"].tolist() == [
        "collision",
        "collision",
        "missing timestamp",
    ]
    assert diagnostics["collisions"][
        ["retained_row", "discarded_row"]
    ].values.tolist() == [[0, 2], [3, 4]]
    strict, _, diagnostics = standardize_time_axis(
        source,
        sampling_interval="500ms",
        snap_tolerance="50ms",
        return_diagnostics=True,
        verbose=False,
    )
    assert np.isnan(strict.loc[base, "power"])
    assert strict.loc[base + pd.Timedelta("500ms"), "power"] == 0
    assert diagnostics["rejection_count"] == 5
    assert diagnostics["collision_count"] == 2


def test_fixed_offset_labels_and_dst_validation():
    fixed = pd.date_range(
        "2024-01-01 07:00", periods=3, freq="h", tz=timezone(timedelta(hours=-5))
    )
    source = pd.DataFrame({"power": [1, 2, 3]}, index=fixed)
    grid, _ = standardize_time_axis(source)
    assert grid.index.tz is None
    np.testing.assert_array_equal(grid.loc[fixed.tz_localize(None), "power"], [1, 2, 3])
    pd.testing.assert_index_equal(
        DataHandler(source).data_frame_raw.index, fixed.tz_localize(None), exact=False
    )
    dst = pd.date_range("2024-03-10", periods=6, freq="h", tz="America/Los_Angeles")
    source = pd.DataFrame({"power": np.arange(6.0)}, index=dst)
    for operation in (standardize_time_axis, DataHandler, make_2d):
        with pytest.raises(ValueError, match="Normalize timestamps"):
            operation(source)
    winters = pd.DatetimeIndex(["2024-01-01", "2024-12-01"], tz="America/Los_Angeles")
    with pytest.raises(ValueError, match="Normalize timestamps"):
        standardize_time_axis(
            pd.DataFrame({"power": [1, 2]}, index=winters), sampling_interval="1h"
        )


def test_make_2d_pads_partial_days_and_preserves_trimming():
    index = pd.date_range("2024-01-01 06:00", periods=24, freq="h")
    source = pd.DataFrame({"power": np.arange(24.0)}, index=index)
    matrix, dates = make_2d(source, "power", return_day_axis=True)
    assert matrix.shape == (24, 2)
    assert dates.tolist() == [pd.Timestamp("2024-01-01"), pd.Timestamp("2024-01-02")]
    assert np.isnan(matrix[:6, 0]).all()
    assert np.isnan(matrix[6:, 1]).all()
    np.testing.assert_array_equal(matrix[~np.isnan(matrix)].shape, (24,))
    np.testing.assert_array_equal(matrix.ravel(order="F")[6:30], np.arange(24.0))
    np.testing.assert_array_equal(
        make_2d(source, "power", trim_start=True), matrix[:, 1:]
    )
    np.testing.assert_array_equal(
        make_2d(source, "power", trim_end=True), matrix[:, :1]
    )
    empty, no_days = make_2d(
        source, "power", trim_start=True, trim_end=True, return_day_axis=True
    )
    assert empty.shape == (24, 0) and no_days.empty
    off_grid = source.set_axis(index + pd.Timedelta("1s"))
    with pytest.raises(ValueError, match="midnight-aligned"):
        make_2d(off_grid, "power")


def test_make_2d_keeps_an_interior_empty_day():
    index = pd.date_range("2024-01-01", periods=120, freq="h")
    values = np.full(120, np.nan)
    values[[28, 85]] = [20, 30]
    matrix, dates = make_2d(
        pd.DataFrame({"power": values}, index=index), "power", return_day_axis=True
    )
    assert dates.tolist() == index[24:96:24].tolist()
    assert matrix[4, 0] == 20 and matrix[13, 2] == 30
    assert np.isnan(matrix[:, 1]).all()


def test_sunrise_mask_uses_exact_nonintegral_minute_rows(monkeypatch):
    rows = 960  # 90-second samples.
    matrix = np.zeros((rows, 4))
    matrix[240:720] = 1000
    monkeypatch.setattr(
        "solardatatools.algorithms.sunrise_sunset_estimation.rise_set_smoothed",
        lambda measured, **kwargs: {
            "sunrises": np.full(4, 6),
            "sunsets": np.full(4, 18),
        },
    )
    analysis = SunriseSunset()
    analysis.calculate_times(matrix, threshold=0.01)
    np.testing.assert_array_equal(analysis.sunup_mask_estimated, matrix > 0)


@pytest.mark.parametrize("strict", [False, True])
def test_fractional_grid_through_pipeline_and_profiler(monkeypatch, strict):
    interval = pd.Timedelta("500ms")
    day_index = pd.date_range("2024-01-01", periods=4, freq="D")
    samples = []
    for date in day_index:
        midday = pd.date_range(date + pd.Timedelta(hours=12), periods=10, freq=interval)
        samples.append(pd.Series(1000.0, index=midday.delete(2)))
    for midnight in day_index[1:]:
        before = pd.date_range(
            midnight - pd.Timedelta(minutes=30), periods=3600, freq=interval
        )
        after = pd.date_range(midnight, periods=3600, freq=interval)
        samples.extend((pd.Series(0.0, index=before[:-1]), pd.Series(0.0, index=after)))
        tie = midnight - pd.Timedelta("250ms")
        samples.append(pd.Series(800.0, index=[tie]))
    competing = day_index[1] + pd.Timedelta("100ms")
    samples.append(pd.Series(9000.0, index=[competing]))
    power = pd.concat(samples).rename("power")
    source = power.to_frame().assign(sensor=power * 2)

    def daylight(analysis, matrix, **kwargs):
        hours = np.arange(matrix.shape[0])[:, None] * 24 / matrix.shape[0]
        analysis.sunup_mask_estimated = np.broadcast_to(
            (hours >= 6) & (hours < 18), matrix.shape
        )
        analysis.sunrise_estimates = np.full(matrix.shape[1], 6)
        analysis.sunset_estimates = np.full(matrix.shape[1], 18)

    def set_scores(handler, **kwargs):
        handler.daily_scores.density = np.ones(4)
        handler.daily_scores.linearity = np.zeros(4)

    def set_flags(handler, **kwargs):
        handler.daily_flags.no_errors = np.ones(4, dtype=bool)
        handler.daily_flags.clear = np.zeros(4, dtype=bool)
        handler.daily_flags.inverter_clipped = np.zeros(4, dtype=bool)

    def apply_fit(analysis, matrix, **kwargs):
        analysis.roll_by_index = np.array([0, 2, 0, -1])

    monkeypatch.setattr(SunriseSunset, "run_optimizer", daylight)
    monkeypatch.setattr(SunriseSunset, "calculate_times", daylight)
    monkeypatch.setattr(TimeShift, "run", apply_fit)
    monkeypatch.setattr(DataHandler, "get_daily_flags", set_flags)
    monkeypatch.setattr(DataHandler, "get_daily_scores", set_scores)
    for method in (
        "detect_clear_days",
        "clipping_check",
        "capacity_clustering",
    ):
        monkeypatch.setattr(DataHandler, method, lambda self, **kwargs: None)

    handler = DataHandler(source)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        handler.run_pipeline(
            sampling_interval=interval,
            snap_tolerance="200ms" if strict else None,
            interp_day=False,
            fix_shifts=True,
            extra_cols="sensor",
            verbose=False,
        )
    rows = 172800
    assert handler.sampling_interval == interval
    assert handler.raw_data_matrix.shape == (rows, 4)
    assert handler.data_sampling == pytest.approx(0.5 / 60)
    assert handler.day_index.tolist() == day_index.tolist()
    assert handler.possible_timezone_misalignment is (None if strict else True)
    assert any("production near midnight" in str(w.message) for w in caught) is (
        not strict
    )
    assert np.isnan(handler.raw_data_matrix[0, 0])
    assert np.isnan(handler.raw_data_matrix[rows // 2 + 2, 0])
    assert handler.raw_data_matrix[0, 1] == 0
    np.testing.assert_array_equal(
        handler.extra_matrices["sensor"], 2 * handler.corrected_observed_matrix
    )

    report = handler.report(verbose=False, return_values=True)
    assert report["timestamp rejection count"] == (4 if strict else 1)
    assert report["timestamp collision count"] == 1
    mask = np.zeros_like(handler.raw_data_matrix, dtype=bool)
    mask[2, 1] = True  # The corrected location of the winning midnight row.
    handler.augment_data_frame(mask, "selected")
    selected = handler.data_frame_raw["selected"]
    assert selected.loc[day_index[1]]
    assert pd.isna(selected.loc[competing])
    pd.testing.assert_index_equal(selected.index, source.index)

    angles = calculate_omega(
        day_index.dayofyear, handler.data_sampling, 0, 0, samples_per_day=rows
    )
    assert angles.shape == handler.raw_data_matrix.shape
    assert angles[-1, 0] - angles[0, 0] == pytest.approx(15 * (24 - 0.5 / 3600))
    figure = handler.plot_daily_signals(
        start_day=0,
        num_days=2,
        filled=False,
        boolean_index=np.array([False, True, False, True]),
    )
    try:
        x = figure.axes[0].lines[0].get_xdata()
        assert pd.Timestamp(x[1]) - pd.Timestamp(x[0]) == interval
        assert pd.Timestamp(x[rows - 1]) == day_index[2] - interval
        assert pd.Timestamp(x[rows]) == day_index[3]
    finally:
        plt.close(figure)
