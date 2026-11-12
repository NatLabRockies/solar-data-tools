from types import SimpleNamespace

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest

from solardatatools import DataHandler
from solardatatools.algorithms import LossFactorAnalysis, PVQuantiles, SunriseSunset


@pytest.fixture
def run_pipeline(monkeypatch):
    def estimate_daylight(analysis, data, **kwargs):
        analysis.sunup_mask_estimated = data > 0
        analysis.sunrise_estimates = np.full(data.shape[1], 7)
        analysis.sunset_estimates = np.full(data.shape[1], 16)

    def detect_clipping(handler, **kwargs):
        handler.inverter_clipping = False
        handler.daily_flags.inverter_clipped = np.zeros(handler.num_days, dtype=bool)

    def detect_capacity_changes(handler, **kwargs):
        handler.capacity_changes = False
        handler.capacity_analysis = SimpleNamespace(
            labels=np.zeros(handler.num_days, dtype=int)
        )

    monkeypatch.setattr(SunriseSunset, "run_optimizer", estimate_daylight)
    monkeypatch.setattr(SunriseSunset, "calculate_times", estimate_daylight)
    monkeypatch.setattr(DataHandler, "clipping_check", detect_clipping)
    monkeypatch.setattr(DataHandler, "capacity_clustering", detect_capacity_changes)

    def run(peak, **kwargs):
        profile = np.zeros(24)
        profile[7:16] = peak * np.array([0.1, 0.4, 0.7, 0.9, 1, 0.9, 0.7, 0.4, 0.1])
        power = np.tile(profile, 12)
        index = pd.date_range("2024-01-01", periods=len(power), freq="h")
        dh = DataHandler(pd.DataFrame({"power": power}, index=index))
        dh.run_pipeline(linearity_threshold=0.5, verbose=False, **kwargs)
        return dh, profile

    return run


def assert_daily_plot(handler, expected_energy, expected_label, **kwargs):
    fig = handler.plot_daily_energy(**kwargs)
    try:
        np.testing.assert_allclose(fig.axes[0].lines[0].get_ydata(), expected_energy)
        assert fig.axes[0].get_ylabel() == expected_label
    finally:
        plt.close(fig)


@pytest.mark.parametrize("peak_w", [100, 3000])
def test_pipeline_and_plots_use_declared_power_units(run_pipeline, monkeypatch, peak_w):
    watts, profile_w = run_pipeline(peak_w)
    kilowatts, _ = run_pipeline(peak_w / 1000, units="kW")
    assert watts.power_units == "W"
    assert kilowatts.power_units == "kW"
    energy_wh = np.full(watts.num_days, profile_w.sum())
    np.testing.assert_allclose(watts.daily_signals.energy, energy_wh)
    np.testing.assert_allclose(kilowatts.daily_signals.energy, energy_wh / 1000)
    assert watts.capacity_estimate == pytest.approx(np.quantile(profile_w, 0.95))
    assert kilowatts.capacity_estimate == pytest.approx(watts.capacity_estimate / 1000)
    for handler in (watts, kilowatts):
        assert handler.report(verbose=False, return_values=True)[
            "capacity"
        ] == pytest.approx(watts.capacity_estimate / 1000)

    assert_daily_plot(watts, energy_wh, "Energy (Wh)")
    assert_daily_plot(watts, energy_wh / 1000, "Energy (kWh)", units="kWh")
    assert_daily_plot(kilowatts, energy_wh / 1000, "Energy (kWh)")
    assert_daily_plot(kilowatts, energy_wh, "Energy (Wh)", units="Wh")
    assert watts.energy_units == "Wh"
    assert kilowatts.energy_units == "kWh"
    np.testing.assert_allclose(watts.daily_signals.energy, energy_wh)
    np.testing.assert_allclose(kilowatts.daily_signals.energy, energy_wh / 1000)

    monkeypatch.setattr(
        "solardatatools.data_handler.plot_bundt_cake",
        lambda data, *, units, **kwargs: (data, units),
    )
    plot_kw, label = watts.plot_bundt(aggregate=False)
    assert label == "kW"
    assert np.max(plot_kw) == pytest.approx(peak_w / 1000)
    plot_w, label = kilowatts.plot_bundt(aggregate=False, units="W")
    assert label == "W"
    np.testing.assert_allclose(plot_w, plot_kw * 1000)
    np.testing.assert_allclose(
        watts.raw_data_matrix, np.tile(profile_w[:, None], (1, watts.num_days))
    )

    fig = watts.plot_heatmap(units="custom label")
    try:
        assert fig.axes[1].get_ylabel() == "custom label"
        np.testing.assert_array_equal(
            fig.axes[0].images[0].get_array(), watts.raw_data_matrix
        )
    finally:
        plt.close(fig)


def test_custom_signal_units_are_preserved(run_pipeline, monkeypatch, capsys):
    handler, profile = run_pipeline(300, units="W/m²")
    assert handler.power_units == "W/m²"
    assert handler.report(return_values=True)["capacity"] is None
    assert "W/m²" in capsys.readouterr().out
    assert_daily_plot(
        handler, np.full(handler.num_days, profile.sum()), "Integral ((W/m²) h)"
    )
    with pytest.raises(ValueError, match="Cannot convert"):
        handler.plot_daily_energy(units="kWh")
    with pytest.raises(ValueError, match="requires power units"):
        handler.run_loss_factor_analysis()

    monkeypatch.setattr(
        "solardatatools.data_handler.plot_bundt_cake",
        lambda data, *, units, **kwargs: (data, units),
    )
    data, label = handler.plot_bundt(aggregate=False)
    assert label == "W/m²"
    assert np.max(data) == pytest.approx(300)


def test_quantile_bundt_displays_fitted_power_on_requested_scale(
    run_pipeline, monkeypatch
):
    watts, _ = run_pipeline(300, units="W")
    quantiles = PVQuantiles(watts, nvals_dil=11)
    fit_w = np.full(365 * 11 + 1, 300.0)
    quantiles.quantiles_dilated[0.5] = fit_w.copy()
    outputs = []

    def plot(data, *, units):
        outputs.append((data, units))

    monkeypatch.setattr(
        "solardatatools.algorithms.quantile_estimation.plot_bundt_cake", plot
    )
    try:
        quantiles.plot_quantile_bundt(0.5)
        quantiles.plot_quantile_bundt(0.5, units="W")
    finally:
        plt.close(plt.gcf())
    for (data, units), (expected, expected_units) in zip(
        outputs, [(0.3, "kW"), (300, "W")], strict=True
    ):
        assert units == expected_units
        np.testing.assert_allclose(data, expected)
    np.testing.assert_array_equal(quantiles.quantiles_dilated[0.5], fit_w)


def test_loss_factor_units_are_propagated_and_reports_remain_in_kwh(
    run_pipeline, monkeypatch, capsys
):
    watts, _ = run_pipeline(300, units="W")
    kilowatts, _ = run_pipeline(0.3, units="kW")

    def estimate_losses(analysis):
        energy = analysis.energy_data
        analysis.energy_model = np.atleast_2d(2 * energy)
        loss = -np.sum(energy)
        analysis.total_energy_loss = loss
        analysis.degradation_energy_loss = 0.4 * loss
        analysis.soiling_energy_loss = 0.3 * loss
        analysis.capacity_change_loss = 0.15 * loss
        analysis.weather_energy_loss = 0.1 * loss
        analysis.outage_energy_loss = 0.05 * loss
        analysis.degradation_rate = -1.0

    monkeypatch.setattr(LossFactorAnalysis, "estimate_losses", estimate_losses)
    reports = []
    for handler in (watts, kilowatts):
        handler.run_loss_factor_analysis(deg_rate=-1.0, include_soiling=False)
        analysis = handler.loss_analysis
        assert analysis.energy_units == handler.energy_units
        assert analysis.use_ixs.all()
        np.testing.assert_array_equal(
            analysis.energy_data, handler.daily_signals.energy
        )
        reports.append(analysis.report())
        lines = capsys.readouterr().out.splitlines()
        total_line = next(line for line in lines if "total energy loss [kWh]" in line)
        total_kwh = -np.sum(kilowatts.daily_signals.energy)
        assert float(total_line.rsplit(maxsplit=1)[-1]) == pytest.approx(
            round(total_kwh, 1)
        )

    assert reports[0]["total energy loss [kWh]"] == pytest.approx(total_kwh)
    for key in reports[0]:
        if key.endswith("[kWh]"):
            assert reports[0][key] == pytest.approx(reports[1][key])

    direct = LossFactorAnalysis(kilowatts.daily_signals.energy, include_soiling=False)
    assert direct.energy_units == "kWh"
    estimate_losses(direct)
    assert direct.report() == reports[1]
    with pytest.raises(ValueError, match="energy_units"):
        LossFactorAnalysis(np.ones(12), energy_units="W")

    wh_analysis = watts.loss_analysis
    for analysis, requested, expected_unit, baseline in (
        (wh_analysis, {}, "Wh", -2 * total_kwh * 1000),
        (wh_analysis, {"units": "kWh"}, "kWh", -2 * total_kwh),
        (direct, {}, "kWh", -2 * total_kwh),
        (direct, {"units": "Wh"}, "Wh", -2 * total_kwh * 1000),
    ):
        fig = analysis.plot_waterfall(**requested)
        try:
            assert fig.axes[0].get_ylabel() == f"Energy ({expected_unit})"
            assert fig.axes[0].patches[0].get_height() == pytest.approx(baseline)
        finally:
            plt.close(fig)

    assert wh_analysis.total_energy_loss == pytest.approx(total_kwh * 1000)
    direct.outage_energy_loss = None
    assert direct.report()["system outage loss [kWh]"] is None
