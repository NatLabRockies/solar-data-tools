from types import SimpleNamespace

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest

from solardatatools import DataHandler
from solardatatools.algorithms import LossFactorAnalysis, SunriseSunset, TimeShift


@pytest.fixture
def source():
    power = np.tile(np.r_[np.zeros(6), np.arange(1.0, 13), np.zeros(6)], 4)
    index = pd.date_range("2024-01-01", periods=power.size, freq="h")
    return pd.DataFrame({"power": power, "sensor": 2 * power}, index=index)


@pytest.fixture(autouse=True)
def deterministic_stages(monkeypatch):
    def daylight(analysis, data, **kwargs):
        days = data.shape[1]
        analysis.sunrise_estimates = np.full(days, 6.0)
        analysis.sunset_estimates = np.full(days, 18.0)
        analysis.sunup_mask_estimated = data > 0
        analysis.threshold = 0.01

    def daily_scores(handler, **kwargs):
        handler.daily_scores.density = np.ones(handler.num_days)
        handler.daily_scores.linearity = np.zeros(handler.num_days)

    def flags(handler, **kwargs):
        handler.daily_flags.no_errors = np.ones(handler.num_days, dtype=bool)

    def clear(handler, **kwargs):
        handler.daily_flags.clear = np.ones(handler.num_days, dtype=bool)

    def clipping(handler, **kwargs):
        handler.daily_flags.inverter_clipped = np.zeros(handler.num_days, dtype=bool)
        handler.inverter_clipping = False

    def capacity(handler, **kwargs):
        handler.capacity_analysis = SimpleNamespace(
            labels=np.zeros(handler.num_days, dtype=int)
        )
        handler.capacity_changes = False

    monkeypatch.setattr(SunriseSunset, "run_optimizer", daylight)
    monkeypatch.setattr(SunriseSunset, "calculate_times", daylight)
    monkeypatch.setattr(DataHandler, "get_daily_scores", daily_scores)
    monkeypatch.setattr(DataHandler, "get_daily_flags", flags)
    monkeypatch.setattr(DataHandler, "detect_clear_days", clear)
    monkeypatch.setattr(DataHandler, "clipping_check", clipping)
    monkeypatch.setattr(DataHandler, "capacity_clustering", capacity)


def _report(handler):
    return handler.report(verbose=False, return_values=True)


def _fail(reason):
    def fail(*args, **kwargs):
        raise RuntimeError(reason)

    return fail


def test_verbose_pipeline_keeps_attribution_before_preflight(
    source, monkeypatch, capsys
):
    handler = DataHandler(source)
    monkeypatch.setattr("cvxpy.Problem.solve", _fail("no test license"))

    handler.run_pipeline(solver="MOSEK", verbose=True)

    output = capsys.readouterr().out
    banner, outcome = output.split("Pipeline elapsed time", maxsplit=1)
    assert "Solar Data Tools Data Onboarding Pipeline" in banner
    assert "Authors: Bennet Meyers and Sara Miskovich, SLAC" in banner
    acknowledgment = (
        "This work was supported by the U.S. Department of Energy’s Office "
        "of Critical Minerals and Energy Innovation (CMEI), Integrated "
        "Energy Systems Office (IESO), under Agreement 34368."
    )
    assert acknowledgment in " ".join(banner.split())
    assert "no test license" in outcome


@pytest.mark.parametrize("reset", [True, False])
def test_failed_preflight_replaces_previous_results(source, monkeypatch, capsys, reset):
    handler = DataHandler(source)
    handler.run_pipeline(verbose=False)
    assert _report(handler)["capacity"] > 0
    assert _report(handler)["capacity change"] is False
    handler.clearsky_sig = np.ones(len(source))
    handler.loss_analysis = object()
    monkeypatch.setattr("cvxpy.Problem.solve", _fail("no test license"))

    handler.run_pipeline(verbose=False, solver="MOSEK", reset=reset)
    report = _report(handler)
    assert report["length"] is report["capacity"] is report["capacity change"] is None
    assert report["time shift correction"] is None
    assert (
        handler.raw_data_matrix is handler.loss_analysis is handler.clearsky_sig is None
    )
    assert report["stage outcomes"]["solver preflight"]["status"] == "failed"
    assert "no test license" in report["stage outcomes"]["solver preflight"]["reason"]
    assert report["stage outcomes"]["daily flags"]["status"] == "skipped"
    assert handler.total_time >= 0
    assert not capsys.readouterr().out


def test_input_error_propagates_with_a_current_report(source):
    handler = DataHandler(source)
    handler.run_pipeline(verbose=False)
    with pytest.raises(ValueError, match="divide a 24-hour day"):
        handler.run_pipeline(sampling_interval="7h", verbose=False)
    report = _report(handler)
    assert report["length"] is None
    assert report["stage outcomes"]["observation preparation"]["status"] == "failed"


def test_raw_matrix_rerun_uses_original_input():
    raw = np.tile(np.arange(24.0)[:, None], (1, 4))
    before = raw.copy()
    handler = DataHandler(raw_data_matrix=raw)
    handler.run_pipeline(verbose=False, max_val=15)
    assert np.isnan(handler.raw_data_matrix[16:]).all()
    handler.run_pipeline(verbose=False, reset=False)
    np.testing.assert_equal(handler.raw_data_matrix, before)
    np.testing.assert_equal(raw, before)


def test_explicit_dst_correction_survives_a_rerun(source, capsys):
    handler = DataHandler(source)
    handler.fix_dst()
    corrected = handler.data_frame_raw.copy()
    handler.run_pipeline(verbose=False)
    handler.run_pipeline(verbose=False, reset=False)
    handler.fix_dst()
    pd.testing.assert_frame_equal(handler.data_frame_raw, corrected)
    assert "already performed" in capsys.readouterr().out


@pytest.mark.parametrize("fit_fails", [False, True])
def test_clock_fit_falls_back_to_good_days_after_clear_day_failure(
    source, monkeypatch, fit_fails
):
    handler = DataHandler(source)
    handler.run_pipeline(verbose=False)
    monkeypatch.setattr(handler, "detect_clear_days", _fail("clear-day solve failed"))
    selections = []

    def fit(analysis, data, *, use_ixs, **kwargs):
        selections.append(use_ixs.copy())
        if fit_fails:
            raise RuntimeError("clock solve failed")
        analysis.roll_by_index = np.array([0, 1, 0, -1])

    monkeypatch.setattr(TimeShift, "run", fit)
    handler.run_pipeline(verbose=False, fix_shifts=True, reset=False)
    report = _report(handler)
    outcomes = report["stage outcomes"]
    assert len(selections) == 1 and selections[0].all()
    assert outcomes["clear days"]["status"] == "failed"
    assert report["clearness score"] is None
    assert report["quality score"] == 1.0
    assert report["capacity change"] is False
    assert outcomes["final daylight"]["status"] == "success"
    if fit_fails:
        assert outcomes["time shift"]["status"] == "failed"
        assert report["time shift correction"] is None
        np.testing.assert_equal(handler.filled_data_matrix, handler.raw_data_matrix)
    else:
        assert outcomes["time shift"]["status"] == "warned fallback"
        assert "all good days" in outcomes["time shift"]["reason"]
        assert report["time shift correction"] is True


def test_final_daylight_failure_keeps_aligned_extra_matrices(
    source, monkeypatch, capsys
):
    def fit(analysis, data, **kwargs):
        analysis.roll_by_index = np.array([0, 1, 0, -1])

    monkeypatch.setattr(TimeShift, "run", fit)
    monkeypatch.setattr(
        SunriseSunset, "calculate_times", _fail("daylight refresh failed")
    )
    handler = DataHandler(source)
    handler.run_pipeline(
        verbose=False, extra_cols=["sensor", "unknown"], fix_shifts=True
    )
    report = _report(handler)
    assert report["quality score"] == 1.0
    assert report["inverter clipping"] is False
    assert report["clipped fraction"] == 0
    assert report["time shift correction"] is True
    assert report["stage outcomes"]["final daylight"]["status"] == "failed"
    assert report["stage outcomes"]["extra quality"]["status"] == "skipped"
    assert report["stage outcomes"]["extra matrix 'unknown'"]["status"] == "failed"
    np.testing.assert_equal(
        handler.extra_matrices["sensor"], 2 * handler.corrected_observed_matrix
    )
    assert handler.extra_quality_scores == {}
    assert handler.daytime_analysis is handler.boolean_masks.daytime is None
    assert not capsys.readouterr().out
    handler.report()
    console = capsys.readouterr().out
    assert "daylight refresh failed" in console and "plot_heatmap" in console
    plt.close(handler.plot_heatmap())
    handler.detect_clear_sky(loss_correction=False)
    assert handler.plot_bundt() is None
    assert _report(handler)["stage outcomes"]["clear sky"]["status"] == "skipped"


def test_provisional_failure_skips_filling_and_preserves_observations(
    source, monkeypatch
):
    monkeypatch.setattr(
        SunriseSunset, "run_optimizer", _fail("provisional solve failed")
    )
    handler = DataHandler(source)
    handler.run_pipeline(verbose=False, zero_night=False, extra_cols="sensor")
    report = _report(handler)
    assert report["capacity"] > 0 and report["quality score"] is None
    assert handler.filled_data_matrix is None
    assert report["stage outcomes"]["matrix filling"]["status"] == "skipped"
    assert report["stage outcomes"]["extra matrices"]["status"] == "success"
    assert "sensor" in handler.extra_matrices


@pytest.mark.parametrize("solar_noon_method", ["rise_set_average", "energy_com"])
def test_location_setup_only_requires_final_bounds_for_optimized_modes(
    source, monkeypatch, solar_noon_method
):
    monkeypatch.setattr(SunriseSunset, "calculate_times", _fail("refresh failed"))
    handler = DataHandler(source, gmt_offset=-8)
    handler.run_pipeline(verbose=False)
    handler.setup_location_and_orientation_estimation(-8)
    assert handler.parameter_estimation is None
    assert handler.estimate_latitude() is None
    assert handler.estimate_longitude() is None
    assert handler.estimate_orientation() is None
    assert handler.estimate_location_and_orientation() is None
    stage = "location and orientation setup"
    outcome = _report(handler)["stage outcomes"][stage]
    assert outcome["status"] == "skipped"
    assert "final daylight" in outcome["reason"]

    handler.setup_location_and_orientation_estimation(
        -8, solar_noon_method=solar_noon_method, daylight_method="sunrise-sunset"
    )
    assert _report(handler)["stage outcomes"][stage]["status"] == "success"
    assert np.isfinite(handler.parameter_estimation.solarnoon).all()
    assert np.isfinite(handler.parameter_estimation.hours_daylight).all()


def test_loss_prerequisites_and_named_clear_sky_factors(source, monkeypatch):
    monkeypatch.setattr(
        DataHandler, "capacity_clustering", _fail("capacity solve failed")
    )
    handler = DataHandler(source)
    handler.run_pipeline(verbose=False)
    handler.run_loss_factor_analysis(verbose=False, deg_rate=-1)
    assert _report(handler)["stage outcomes"]["loss analysis"]["status"] == "skipped"

    def fit(analysis):
        assert analysis.capacity_change_labels is None
        analysis.total_energy_loss = -1.0
        analysis.energy_model = np.ones((5, handler.num_days))
        analysis.energy_model[1] = 0.5
        analysis.energy_model[2] = 0.8

    monkeypatch.setattr(LossFactorAnalysis, "estimate_losses", fit)
    with pytest.warns(UserWarning, match="one year or less"):
        handler.run_loss_factor_analysis(
            verbose=False, deg_rate=-1, use_capacity_change_labels=False
        )
    assert _report(handler)["stage outcomes"]["loss analysis"]["status"] == "success"
    assert "one year" in _report(handler)["stage outcomes"]["loss analysis"]["warning"]
    signal = handler.raw_data_matrix.ravel(order="F").copy()
    handler.quantile_object = SimpleNamespace(
        quantile_levels=[0.9],
        quantiles_original={0.9: np.ones_like(signal)},
        sig_original=signal,
    )

    class ClearSky:
        def __init__(self, corrected, quantile, **kwargs):
            np.testing.assert_allclose(corrected, signal / 0.4)

        def get_clearsky_sig(self):
            return np.ones_like(signal)

    monkeypatch.setattr("solardatatools.data_handler.ClearSkyDetection", ClearSky)
    handler.detect_clear_sky()
    assert _report(handler)["stage outcomes"]["clear sky"]["status"] == "success"
    assert handler.boolean_masks.clear_times.any()
    handler.loss_analysis.energy_model[1, 0] = 0
    handler.detect_clear_sky()
    assert _report(handler)["stage outcomes"]["clear sky"]["status"] == "skipped"
    assert handler.boolean_masks.clear_times is None
