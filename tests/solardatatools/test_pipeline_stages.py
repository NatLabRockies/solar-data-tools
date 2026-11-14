import numpy as np
import pandas as pd
import pytest

from solardatatools import DataHandler
from solardatatools.algorithms import SunriseSunset, TimeShift


@pytest.mark.parametrize("fix_shifts", [False, True])
def test_pipeline_preserves_stage_order_and_options(monkeypatch, fix_shifts):
    index = pd.date_range("2024-01-01", periods=96, freq="h")
    power = np.tile(np.r_[np.zeros(7), np.arange(1.0, 11), np.zeros(7)], 4)
    power[24 + 9] = np.nan
    source = pd.DataFrame({"sensor": np.arange(96.0), "power": power}, index=index)
    handler = DataHandler(source)
    calls = []
    filled = handler.make_filled_data_matrix
    embed_extra = handler._embed_extra_matrix
    provisional_mask = np.zeros((24, 2), dtype=bool)
    provisional_mask[7:17] = True
    corrected_mask = np.roll(provisional_mask, int(fix_shifts), axis=0)

    def estimate_daylight(analysis, data, **kwargs):
        calls.append(("provisional daylight", kwargs))
        assert handler.power_units == "kW"
        assert handler.data_sampling == 60
        assert handler.time_axis_diagnostics["rejection_count"] == 0
        assert handler.capacity_estimate == np.nanquantile(data, 0.95)
        np.testing.assert_array_equal(data, power[24:72].reshape(24, 2, order="F"))
        np.testing.assert_array_equal(
            handler.boolean_masks.missing_values, np.isnan(data)
        )
        analysis.threshold = 0.023
        analysis.sunrise_estimates = np.full(2, 7)
        analysis.sunset_estimates = np.full(2, 17)
        analysis.sunup_mask_estimated = provisional_mask

    def fill(**kwargs):
        calls.append(("fill", kwargs))
        filled(**kwargs)

    def record(name):
        def run(**kwargs):
            calls.append((name, kwargs))
            if name == "quality scores":
                handler.daily_scores.density = np.ones(2)
                handler.daily_scores.linearity = np.zeros(2)
            elif name == "quality flags":
                handler.daily_flags.no_errors = np.ones(2, dtype=bool)
            elif name == "clear days":
                handler.daily_flags.clear = np.ones(2, dtype=bool)
            elif name == "capacity":
                handler.capacity_changes = False

        return run

    def clip(**kwargs):
        calls.append(("clipping", kwargs))
        if fix_shifts:
            raise RuntimeError("test clipping failure")
        handler.inverter_clipping = False
        handler.daily_flags.inverter_clipped = np.zeros(2, dtype=bool)

    def summarize():
        calls.append(("summary", {}))
        handler.data_quality_score = 0.75

    def correct_clock(**kwargs):
        calls.append(("clock correction", kwargs))
        assert handler.data_quality_score == 0.75
        analysis = TimeShift()
        analysis.roll_by_index = np.ones(2, dtype=int)
        handler.time_shift_analysis = analysis
        handler.time_shifts = True
        handler.filled_data_matrix = analysis.apply_corrections(
            handler.filled_data_matrix
        )

    def refresh_daylight(analysis, data, **kwargs):
        calls.append(("final daylight", kwargs))
        assert analysis is handler.daytime_analysis
        assert analysis.threshold == 0.023
        expected = np.nan_to_num(handler.raw_data_matrix)
        np.testing.assert_array_equal(data, np.roll(expected, int(fix_shifts), axis=0))
        analysis.sunup_mask_estimated = corrected_mask

    def align_extra(column):
        calls.append(("extra column", {"column": column}))
        np.testing.assert_array_equal(handler.boolean_masks.daytime, corrected_mask)
        return embed_extra(column)

    monkeypatch.setattr(SunriseSunset, "run_optimizer", estimate_daylight)
    monkeypatch.setattr(SunriseSunset, "calculate_times", refresh_daylight)
    monkeypatch.setattr(handler, "make_filled_data_matrix", fill)
    monkeypatch.setattr(handler, "get_daily_scores", record("quality scores"))
    monkeypatch.setattr(handler, "get_daily_flags", record("quality flags"))
    monkeypatch.setattr(handler, "detect_clear_days", record("clear days"))
    monkeypatch.setattr(handler, "clipping_check", clip)
    monkeypatch.setattr(handler, "score_data_set", summarize)
    monkeypatch.setattr(handler, "capacity_clustering", record("capacity"))
    monkeypatch.setattr(handler, "auto_fix_time_shifts", correct_clock)
    monkeypatch.setattr(handler, "_embed_extra_matrix", align_extra)

    result = handler.run_pipeline(
        power_col="power",
        start_day_ix=1,
        end_day_ix=3,
        sampling_interval="1h",
        snap_tolerance="2min",
        units="kW",
        zero_night=False,
        interp_day=False,
        density_lower_threshold=0.4,
        density_upper_threshold=0.95,
        linearity_threshold=0.2,
        clear_day_smoothness_param=0.7,
        clear_day_energy_param=0.6,
        sunrise_sunset_random_seed=6,
        fix_shifts=fix_shifts,
        round_shifts_to_hour=False,
        time_shift_weight_change_detector=0.25,
        time_shift_weight_seasonal=0.2,
        periodic_detector=True,
        solar_noon_estimator="com",
        daytime_threshold=0.01,
        extra_cols="sensor",
        solver="SHIFT_SOLVER",
        solver_convex="CONVEX_SOLVER",
        verbose=False,
    )

    expected_calls = [
        (
            "provisional daylight",
            {"random_seed": 6, "plot": False, "solver": "CONVEX_SOLVER"},
        ),
        ("fill", {"zero_night": False, "interp_day": False}),
        ("quality scores", {"threshold": 0.2, "solver": "CONVEX_SOLVER"}),
        (
            "quality flags",
            {
                "density_lower_threshold": 0.4,
                "density_upper_threshold": 0.95,
                "linearity_threshold": 0.2,
            },
        ),
        (
            "clear days",
            {
                "smoothness_threshold": 0.7,
                "energy_threshold": 0.6,
                "solver": "CONVEX_SOLVER",
            },
        ),
        ("clipping", {"solver": "CONVEX_SOLVER"}),
        ("summary", {}),
        ("capacity", {"solver": "SHIFT_SOLVER"}),
    ]
    if fix_shifts:
        expected_calls.append(
            (
                "clock correction",
                {
                    "round_shifts_to_hour": False,
                    "w1": 0.25,
                    "w2": 0.2,
                    "estimator": "com",
                    "threshold": 0.01,
                    "periodic_detector": True,
                    "solver": "SHIFT_SOLVER",
                },
            )
        )
    expected_calls.extend(
        [
            ("final daylight", {"solver": "CONVEX_SOLVER"}),
            ("extra column", {"column": "sensor"}),
        ]
    )
    assert calls == expected_calls
    assert result is None
    assert handler._ran_pipeline
    assert handler.data_quality_score == 0.75
    assert handler.inverter_clipping is (None if fix_shifts else False)
    outcome = handler.report(verbose=False, return_values=True)["stage outcomes"][
        "clipping"
    ]
    assert outcome["status"] == ("failed" if fix_shifts else "success")
    assert outcome["reason"] == ("test clipping failure" if fix_shifts else None)
    sensor = source["sensor"].to_numpy()[24:72].reshape(24, 2, order="F")
    np.testing.assert_array_equal(
        handler.extra_matrices["sensor"], np.roll(sensor, int(fix_shifts), axis=0)
    )
    assert handler.extra_quality_scores["sensor"] == 1
