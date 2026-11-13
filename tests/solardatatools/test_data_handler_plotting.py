from types import SimpleNamespace

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest

from solardatatools import DataHandler
from solardatatools.algorithms import CapacityChange, TimeShift


@pytest.fixture
def prepared_handler():
    raw = np.arange(72.0).reshape(24, 3)
    raw[10, 0] = np.nan
    handler = DataHandler(raw_data_matrix=raw)
    handler.day_index = pd.date_range("2024-02-01", periods=3, freq="D")
    handler.power_units = "W"
    handler.boolean_masks.daytime = (np.arange(24)[:, None] >= 7) & (
        np.arange(24)[:, None] < 17
    )
    handler.boolean_masks.daytime = np.tile(handler.boolean_masks.daytime, (1, 3))
    handler.make_filled_data_matrix(zero_night=False, interp_day=False)
    handler.extra_matrices["sensor"] = np.full((24, 3), 23.0)
    handler.daily_flags.no_errors = np.array([True, False, True])
    return handler


@pytest.mark.parametrize(
    "matrix, flag, kwargs, units, divisor",
    [
        ("raw", None, {}, "kW", 1000),
        ("filled", "good", {"scale_to_kw": False}, "W", 1),
        ("sensor", "bad", {"units": "°C"}, "°C", 1),
    ],
)
def test_handler_heatmap_selection(
    prepared_handler, matrix, flag, kwargs, units, divisor
):
    handler = prepared_handler
    matrices = {
        "raw": handler.raw_data_matrix,
        "filled": handler.filled_data_matrix,
        **handler.extra_matrices,
    }
    source = matrices[matrix]
    before = source.copy()
    fig = handler.plot_heatmap(matrix=matrix, flag=flag, year_lines=False, **kwargs)
    try:
        ax, colorbar = fig.axes
        np.testing.assert_equal(
            ax.images[0].get_array().filled(np.nan), before / divisor
        )
        np.testing.assert_equal(source, before)
        assert colorbar.get_ylabel() == units
        np.testing.assert_equal(ax.get_xticks(), [0])
        if flag:
            assert ax.get_title() == f"Measured power, {flag} days flagged"
            expected_days = [0, 2] if flag == "good" else [1]
            np.testing.assert_equal(
                ax.collections[0].get_offsets()[:, 0], expected_days
            )
        else:
            assert ax.get_title() == "Measured power"
            assert len(ax.collections) == 0
    finally:
        plt.close(fig)


def test_handler_daily_energy_preserves_flag_dates_and_units(prepared_handler):
    handler = prepared_handler
    energy = handler.daily_signals.energy.copy()
    fig = handler.plot_daily_energy(flag="bad", units="kWh")
    try:
        ax = fig.axes[0]
        all_days, bad_days = ax.lines
        np.testing.assert_allclose(all_days.get_ydata(), energy / 1000)
        np.testing.assert_allclose(bad_days.get_ydata(), energy[[1]] / 1000)
        pd.testing.assert_index_equal(
            pd.DatetimeIndex(bad_days.get_xdata()), handler.day_index[[1]]
        )
        assert ax.get_ylabel() == "Energy (kWh)"
        np.testing.assert_equal(handler.daily_signals.energy, energy)
    finally:
        plt.close(fig)


@pytest.mark.parametrize("show_filter", [False, True])
def test_handler_time_shift_plot_uses_fitted_results(prepared_handler, show_filter):
    handler = prepared_handler
    metric = np.array([12.5, 11.5, 12.1])
    shift = np.array([0.5, -0.5, 0.0])
    seasonal = np.array([12.0, 12.0, 12.1])
    analysis = TimeShift()
    analysis.metric = metric
    analysis.s1 = shift
    analysis.s2 = seasonal
    analysis.use_ixs = handler.daily_flags.no_errors
    handler.time_shift_analysis = analysis
    fig = handler.plot_time_shift_analysis_results(
        show_filter=show_filter, figsize=(5, 3)
    )
    try:
        lines = {line.get_label(): line for line in fig.axes[0].lines}
        assert tuple(fig.get_size_inches()) == (5, 3)
        np.testing.assert_equal(lines["daily solar noon"].get_ydata(), metric)
        np.testing.assert_equal(lines["shift detector"].get_ydata(), shift)
        np.testing.assert_equal(lines["signal model"].get_ydata(), shift + seasonal)
        assert ("filtered days" in lines) == show_filter
        if show_filter:
            pd.testing.assert_index_equal(
                pd.DatetimeIndex(lines["filtered days"].get_xdata()),
                handler.day_index[[0, 2]],
            )
        np.testing.assert_equal(analysis.metric, [12.5, 11.5, 12.1])
        np.testing.assert_equal(analysis.s1, [0.5, -0.5, 0.0])
        np.testing.assert_equal(analysis.s2, [12.0, 12.0, 12.1])
    finally:
        plt.close(fig)


def test_standalone_time_shift_plot_uses_day_numbers():
    analysis = TimeShift()
    assert analysis.plot_analysis() is None
    analysis.metric = np.array([12.5, 11.5, 12.1])
    analysis.s1 = np.array([0.5, -0.5, 0.0])
    analysis.s2 = np.array([12.0, 12.0, 12.1])

    fig = analysis.plot_analysis(figsize=(6, 4))
    try:
        ax = fig.axes[0]
        assert ax.get_xlabel() == "day number"
        assert ax.get_ylabel() == "solar noon [hours]"
        assert tuple(fig.get_size_inches()) == (6, 4)
        assert [line.get_label() for line in ax.lines] == [
            "metric",
            "shift detector",
            "SD model",
        ]
        for line in ax.lines:
            np.testing.assert_equal(line.get_xdata(), np.arange(3))
        np.testing.assert_equal(ax.lines[0].get_ydata(), [12.5, 11.5, 12.1])
        np.testing.assert_equal(ax.lines[1].get_ydata(), [0.5, -0.5, 0.0])
        np.testing.assert_equal(ax.lines[2].get_ydata(), [12.5, 11.5, 12.1])
        assert ax.lines[2].get_linestyle() == "--"
        np.testing.assert_equal(analysis.metric, [12.5, 11.5, 12.1])
        np.testing.assert_equal(analysis.s1, [0.5, -0.5, 0.0])
        np.testing.assert_equal(analysis.s2, [12.0, 12.0, 12.1])
    finally:
        plt.close(fig)


@pytest.mark.parametrize("detected", [False, True])
def test_handler_clipping_scores_use_stored_results(prepared_handler, detected):
    handler = prepared_handler
    assert handler.plot_clipping() is None
    handler.daily_scores.clipping_1 = np.array([0.75, 0.8, 1.0])
    handler.daily_scores.clipping_2 = np.array([0.1, 0.2, 0.4])
    handler.daily_flags.inverter_clipped = np.array([False, False, True])
    handler.inverter_clipping = detected
    expected = ([0.75, 0.8, 1.0], [0.1, 0.2, 0.4])

    fig = handler.plot_clipping(figsize=(6, 4))
    try:
        assert tuple(fig.get_size_inches()) == (6, 4)
        for ax, scores in zip(fig.axes, expected, strict=True):
            assert len(ax.lines) == (2 if detected else 1)
            np.testing.assert_equal(ax.lines[0].get_ydata(), scores)
            if detected:
                np.testing.assert_equal(ax.lines[1].get_ydata(), scores[-1:])
                pd.testing.assert_index_equal(
                    pd.DatetimeIndex(ax.lines[1].get_xdata()), handler.day_index[-1:]
                )
        np.testing.assert_equal(handler.daily_scores.clipping_1, expected[0])
        np.testing.assert_equal(handler.daily_scores.clipping_2, expected[1])
    finally:
        plt.close(fig)


@pytest.mark.parametrize(
    "configured, show_fit", [(False, True), (True, True), (True, False)]
)
def test_density_plot_uses_quality_thresholds(prepared_handler, configured, show_fit):
    handler = prepared_handler
    handler.daily_signals.density = np.array([0.65, 0.7, 0.75])
    fit = np.array([0.7, 0.75, 0.8])
    handler.daily_signals.seasonal_density_fit = fit
    thresholds = (0.4, 0.9) if configured else (0.6, 1.05)
    if configured:
        handler.daily_scores.density = np.array([0.65, 0.7, 0.75])
        handler.daily_scores.linearity = np.array([0.01, 0.02, 0.03])
        handler.get_daily_flags(
            density_lower_threshold=thresholds[0], density_upper_threshold=thresholds[1]
        )
    fig = handler.plot_density_signal(flag="good", show_fit=show_fit)
    try:
        lines = fig.axes[0].lines
        assert len(lines) == (5 if show_fit else 2)
        if show_fit:
            np.testing.assert_equal(lines[2].get_ydata(), fit)
            for line, threshold in zip(lines[3:], thresholds, strict=True):
                np.testing.assert_allclose(line.get_ydata(), threshold * fit)
    finally:
        plt.close(fig)


@pytest.mark.parametrize("show_clusters", [False, True])
def test_handler_capacity_plot_converts_fitted_components_to_power(
    prepared_handler, show_clusters
):
    handler = prepared_handler
    analysis = CapacityChange()
    analysis.metric = np.log([10.0, 16.0, 24.0])
    analysis.s1 = np.log([10.0, 10.0, 20.0])
    analysis.s2 = np.log([1.0, 1.5, 1.2])
    analysis.s3 = np.log([2.0, 2.0, 1.0])
    analysis.labels = np.array([0, 0, 1])
    handler.capacity_analysis = analysis

    fig = handler.plot_capacity_change_analysis(show_clusters=show_clusters)
    try:
        assert len(fig.axes) == (2 if show_clusters else 1)
        lines = {line.get_label(): line for line in fig.axes[0].lines}
        np.testing.assert_allclose(lines["measured signal"].get_ydata(), [10, 24])
        pd.testing.assert_index_equal(
            pd.DatetimeIndex(lines["measured signal"].get_xdata()),
            handler.day_index[[0, 2]],
        )
        np.testing.assert_allclose(
            lines["capacity change detector"].get_ydata(), [10, 10, 20]
        )
        np.testing.assert_allclose(lines["signal model"].get_ydata(), [20, 30, 24])
        if show_clusters:
            np.testing.assert_equal(fig.axes[1].lines[0].get_ydata(), analysis.labels)
        assert handler.capacity_changes is True
    finally:
        plt.close(fig)


def test_handler_bundt_renders_with_daylight_bounds(prepared_handler):
    handler = prepared_handler
    handler.daytime_analysis = SimpleNamespace(
        sunrise_estimates=np.full(3, 7), sunset_estimates=np.full(3, 17)
    )
    before = handler.raw_data_matrix.copy()
    fig = handler.plot_bundt(aggregate=False, figsize=(5, 4), units="W", cmap="viridis")
    try:
        ax = fig.axes[0]
        assert ax.name == "3d"
        assert ax.get_zlabel() == "W"
        assert ax.collections[0].get_cmap().name == "viridis"
        fig.canvas.draw()
        np.testing.assert_equal(handler.raw_data_matrix, before)
    finally:
        plt.close(fig)
