import warnings

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest

from solardatatools import DataHandler, standardize_time_axis
from solardatatools.daytime import _check_midnight_production


@pytest.mark.parametrize("correct_tz", [True, False])
def test_standardization_keeps_legacy_tuple_without_timezone_shift(correct_tz):
    # Solar noon near 18:00 used to trigger the initial timezone correction.
    power = np.zeros((24, 5))
    power[14:23] = 800
    power[16, 2] = np.nan
    timestamps = pd.date_range("2024-01-01", periods=power.size, freq="h")
    source = pd.DataFrame({"power": power.ravel(order="F")}, index=timestamps)

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        frame, offset = standardize_time_axis(
            source, power_col="power", correct_tz=correct_tz, verbose=False
        )
    pd.testing.assert_frame_equal(
        frame, source, check_index_type=False, check_freq=False
    )
    assert offset == 0
    assert len(caught) == 1
    assert caught[0].category is DeprecationWarning
    assert "correct_tz is deprecated" in str(caught[0].message)


def test_midnight_rule_requires_recurring_meaningful_production():
    data = np.zeros((96, 41))
    data[40:56] = 1000
    data[0, 0] = data[-1, -1] = 1000  # Excluded outer record boundaries.
    data[4] = data[-5] = 1000  # 01:00 and 22:45 are outside the windows.
    data[-4, :-1] = 1  # Recurring noise.
    data[1, 1:] = 50  # Exactly 5% of the reference is insufficient.
    data[0, 1] = 800  # An isolated spike is insufficient.
    assert _check_midnight_production(data) is False
    assert _check_midnight_production(data[:, :11]) is False  # One in ten.

    data[0, 2:4] = 800  # Three of 40 eligible midnights is below 10%.
    assert _check_midnight_production(data) is False

    data[-4, 3] = 800  # Exactly four of 40 eligible midnights, using the prior day.
    np.testing.assert_array_equal(data[0, 4:], 0)
    assert _check_midnight_production(data) is True
    assert _check_midnight_production(data / 1000) is True


def test_midnight_rule_requires_finite_observations_on_both_sides():
    data = np.zeros((96, 5))
    data[40:56] = 1000
    data[:4] = data[-4:] = np.nan
    data[-4:-2, :3] = 0
    data[:2, 1:4] = 200
    assert _check_midnight_production(data) is True

    # The third midnight still has power and more than half coverage in total,
    # but the hour preceding it has fewer than half its expected observations.
    data[2, 3] = 200
    data[-4, 2] = np.inf
    assert _check_midnight_production(data) is None
    assert _check_midnight_production(np.zeros((24, 5))) is None
    assert _check_midnight_production(np.ones((12, 5))) is None


def test_missing_midnight_slots_do_not_supply_advisory_coverage(monkeypatch):
    hours = np.arange(24)
    profile = 1000 * np.maximum(np.sin(np.pi * (hours - 6) / 12), 0)
    timestamps = pd.date_range("2024-01-01", periods=24 * 12, freq="h")
    source = pd.DataFrame({"power": np.tile(profile, 12)}, index=timestamps)
    source = source[~source.index.hour.isin([23, 0])]
    handler = DataHandler(source)
    monkeypatch.setattr(DataHandler, "clipping_check", lambda self, **kwargs: None)
    monkeypatch.setattr(DataHandler, "capacity_clustering", lambda self, **kwargs: None)

    handler.run_pipeline(verbose=False, linearity_threshold=0.5)

    assert np.isnan(handler.raw_data_matrix[[0, -1]]).all()
    assert handler.possible_timezone_misalignment is None


@pytest.mark.parametrize(
    "scenario, correct_tz, expected",
    [
        ("late", True, False),
        ("mixed", False, True),
        ("gaps", None, None),
        ("wrapped", None, True),
    ],
)
def test_pipeline_advises_on_observations_without_shifting_them(
    scenario, correct_tz, expected
):
    hours = np.arange(24)
    observed = np.column_stack(
        [
            (900 + day * 10) * np.maximum(np.sin(np.pi * (hours - 6 - day % 2) / 12), 0)
            for day in range(18)
        ]
    )
    if scenario == "late":
        # This offset was too small for the initial heuristic but triggered the late roll.
        observed = np.roll(observed, 3, axis=0)
    elif scenario == "mixed":
        observed[:, 5:11] = np.roll(observed[:, 5:11], 12, axis=0)
    elif scenario == "wrapped":
        observed = np.roll(observed, 12, axis=0)
    else:
        observed[[0, 23]] = np.nan
    observed[13, 1] = np.nan
    timestamps = pd.date_range("2024-01-01", periods=observed.size, freq="h")
    sensor = np.arange(observed.size, dtype=float)
    sensor[-2] = np.nan
    source = pd.DataFrame(
        {"power": observed.ravel(order="F"), "sensor": sensor}, index=timestamps
    )
    handler = DataHandler(source)
    kwargs = {"correct_tz": correct_tz} if correct_tz is not None else {}

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        handler.run_pipeline(
            extra_cols="sensor", verbose=False, linearity_threshold=0.5, **kwargs
        )

    for frame in (handler.data_frame_raw, handler.data_frame):
        pd.testing.assert_frame_equal(
            frame[source.columns], source, check_index_type=False, check_freq=False
        )
    pd.testing.assert_index_equal(handler.day_index, timestamps[::24], exact=False)
    np.testing.assert_array_equal(handler.raw_data_matrix, observed)
    np.testing.assert_array_equal(
        handler.boolean_masks.missing_values, np.isnan(observed)
    )
    np.testing.assert_array_equal(handler.corrected_observed_matrix, observed)
    observed_mask = np.isfinite(observed)
    np.testing.assert_array_equal(
        handler.filled_data_matrix[observed_mask], observed[observed_mask]
    )
    np.testing.assert_allclose(
        handler.daily_signals.energy, handler.filled_data_matrix.sum(axis=0)
    )
    assert handler.tz_correction == 0
    assert handler.possible_timezone_misalignment is expected
    advisory = [
        w for w in caught if "Significant production near midnight" in str(w.message)
    ]
    assert len(advisory) == int(expected is True)
    deprecated = [w for w in caught if "correct_tz is deprecated" in str(w.message)]
    assert len(deprecated) == int(correct_tz is not None)

    if scenario == "wrapped":
        assert (
            "no nonzero observations within the estimated daylight window"
            in handler._error_msg
        )
        assert handler.daily_flags is None
        fig = handler.plot_heatmap()
        try:
            plotted = fig.axes[0].images[0].get_array().filled(np.nan)
            np.testing.assert_array_equal(plotted, observed / 1000)
            assert fig.axes[1].get_ylabel() == "kW"
        finally:
            plt.close(fig)
        np.testing.assert_array_equal(handler.raw_data_matrix, observed)
        return

    np.testing.assert_array_equal(
        handler.extra_matrices["sensor"], sensor.reshape(observed.shape, order="F")
    )
    assert np.isfinite(handler.daytime_analysis.sunrise_estimates).all()
    assert np.isfinite(handler.daytime_analysis.sunset_estimates).all()
    report = handler.report(verbose=False, return_values=True)
    assert report["time zone correction"] == 0
    assert report["possible timezone misalignment"] is expected
