import numpy as np
import pytest

from solardatatools import DataHandler
from solardatatools.algorithms import SunriseSunset
from solardatatools.data_quality import make_linearity_scores, make_quality_flags


def test_straight_ramp_score_counts_daylight_interiors_and_is_unit_invariant():
    filled = np.tile((100 + 0.0001 * np.arange(1000) ** 2)[:, None], (1, 3))
    daylight = np.zeros_like(filled, dtype=bool)
    daylight[100:900, :2] = True
    filled[120:151, 0] = np.linspace(160, 199, 31)
    filled[200:241, 0] = np.linspace(599, 555, 41)
    filled[300:330, 0] = np.linspace(200, 210, 30)
    filled[500:540, 0] = 200
    filled[905:950, 0] = np.linspace(200, 210, 45)
    filled[50:90, 2] = np.linspace(700, 800, 40)
    expected = np.zeros_like(daylight)
    expected[121:150, 0] = True
    expected[201:240, 0] = True

    watts, mask = make_linearity_scores(
        filled, 1000, sampling_interval="1s", daylight_mask=daylight
    )
    kilowatts, kw_mask = make_linearity_scores(
        filled / 1000, 1, sampling_interval="1s", daylight_mask=daylight
    )

    np.testing.assert_array_equal(mask, expected)
    np.testing.assert_array_equal(kw_mask, expected)
    np.testing.assert_allclose(watts, [(29 + 39) / 800, 0, np.nan])
    np.testing.assert_array_equal(kilowatts, watts)
    _, passed = make_quality_flags(np.ones(3), watts)
    np.testing.assert_array_equal(passed, [True, True, False])


@pytest.mark.parametrize(
    "interval, insufficient_steps, qualifying_steps",
    [("500ms", 59, 60), ("5s", 5, 6), ("1h", 2, 3)],
)
def test_straight_ramp_requires_elapsed_span_and_two_interiors(
    interval, insufficient_steps, qualifying_steps
):
    trace = np.tile((10 + 0.0001 * np.arange(200) ** 2)[:, None], (1, 2))
    trace[4 : 5 + insufficient_steps, 0] = np.linspace(40, 55, insufficient_steps + 1)
    trace[4 : 5 + qualifying_steps, 1] = np.linspace(40, 55, qualifying_steps + 1)
    daylight = np.ones_like(trace, dtype=bool)
    scores, mask = make_linearity_scores(
        trace, 100, sampling_interval=interval, daylight_mask=daylight
    )
    expected = np.zeros_like(daylight)
    expected[5 : 4 + qualifying_steps, 1] = True
    np.testing.assert_array_equal(mask, expected)
    np.testing.assert_array_equal(scores, [0, (qualifying_steps - 1) / len(trace)])


def test_adjacent_straight_ramps_are_independent_of_input_dtype():
    seconds = np.arange(121)
    filled = (400 + seconds + np.maximum(seconds - 60, 0) / 256)[:, None]
    daylight = np.ones_like(filled, dtype=bool)
    expected = daylight.copy()
    expected[[0, 60, 120], 0] = False

    for dtype in (np.float32, np.float64):
        stored = filled.astype(dtype)
        np.testing.assert_array_equal(stored, filled)
        scores, mask = make_linearity_scores(
            stored, 1000, sampling_interval="1s", daylight_mask=daylight
        )
        np.testing.assert_array_equal(mask, expected)
        np.testing.assert_array_equal(scores, [118 / 121])


def test_ramps_stop_at_missing_or_outside_daylight_samples():
    filled = np.tile(np.linspace(60, 200, 61)[:, None], (1, 2))
    daylight = np.ones_like(filled, dtype=bool)
    filled[30, 0] = np.nan
    daylight[30, 1] = False
    scores, mask = make_linearity_scores(
        filled, 1000, sampling_interval="1s", daylight_mask=daylight
    )
    np.testing.assert_array_equal(scores, [0, 0])
    assert not mask.any()


def test_no_positive_capacity_is_unavailable():
    filled = np.zeros((24, 1))
    scores, mask = make_linearity_scores(
        filled,
        0,
        sampling_interval="1h",
        daylight_mask=np.ones_like(filled, dtype=bool),
    )
    assert np.isnan(scores).all()
    assert not mask.any()


def test_legacy_linearity_call_keeps_modal_score_and_mask():
    scores, mask = make_linearity_scores(np.ones((24, 2)), 1, np.ones(2))
    np.testing.assert_array_equal(scores, np.ones(2))
    assert mask.all()


def test_pipeline_scores_filled_ramps_on_existing_daylight_grid(monkeypatch):
    raw = np.zeros((8640, 2))
    daylight = np.zeros_like(raw, dtype=bool)
    daylight[2160:6480] = True
    raw[2160:6480, :] = (300 + 4 * (np.arange(4320) / 4320) ** 2)[:, None]
    raw[2700, 0] = raw[2703, 0] = 400
    raw[2701:2703, 0] = np.nan
    raw[5100, 0], raw[5103, 0] = 400, 600
    raw[5101:5103, 0] = np.nan
    expected = np.zeros_like(daylight)
    expected[5101:5103, 0] = True
    final_daylight = daylight.copy()
    final_daylight[6300:6480] = False
    calls = []

    def publish_daylight(analysis, mask):
        analysis.sunrise_estimates = np.full(2, 6.0)
        analysis.sunset_estimates = np.full(2, 18.0)
        analysis.sunup_mask_estimated = mask

    def provisional(analysis, data, **kwargs):
        calls.append("provisional")
        publish_daylight(analysis, daylight)

    def final(analysis, data, **kwargs):
        calls.append("final")
        publish_daylight(analysis, final_daylight)

    def density(handler, **kwargs):
        handler.daily_scores.density = np.ones(2)

    monkeypatch.setattr(SunriseSunset, "run_optimizer", provisional)
    monkeypatch.setattr(SunriseSunset, "calculate_times", final)
    monkeypatch.setattr(DataHandler, "get_density_scores", density)

    handler = DataHandler(raw_data_matrix=raw)
    handler.run_pipeline(verbose=False, linearity_threshold=2 / 4320)

    assert calls == ["provisional", "final"]
    np.testing.assert_array_equal(handler.boolean_masks.linearity_daylight, daylight)
    np.testing.assert_array_equal(handler.boolean_masks.daytime, final_daylight)
    np.testing.assert_array_equal(handler.boolean_masks.infill, expected)
    np.testing.assert_array_equal(handler.daily_scores.linearity, [2 / 4320, 0])
    np.testing.assert_array_equal(handler.daily_flags.linearity, [False, True])


def test_quality_flags_keep_other_days_usable_when_linearity_is_unavailable():
    handler = DataHandler()
    handler.num_days = 2
    handler.daily_scores.density = np.ones(2)
    handler.daily_scores.linearity = np.array([np.nan, 0])
    handler.get_daily_flags()
    np.testing.assert_array_equal(handler.daily_flags.no_errors, [False, True])
    np.testing.assert_array_equal(handler.daily_scores.quality_clustering, [-1, -1])


def test_pipeline_recovers_ramps_between_float32_observations():
    seconds = np.arange(86400)
    readings = (1000 * np.sin(np.pi * (seconds - 21600) / 43200)).clip(0)
    raw = np.tile(readings.astype(np.float32)[:, None], (1, 7))
    gaps = np.zeros_like(raw, dtype=bool)
    for hour in range(9, 15):
        start = 3600 * hour
        gaps[start : start + 900, 0] = True
    raw[gaps] = np.nan

    single = DataHandler(raw_data_matrix=raw)
    double = DataHandler(raw_data_matrix=raw.astype(np.float64))
    for handler in (single, double):
        handler.run_pipeline(verbose=False)
        np.testing.assert_array_equal(handler.raw_data_matrix, raw)
        np.testing.assert_array_equal(handler.boolean_masks.infill, gaps)
        assert not handler.daily_flags.linearity[0]
        assert handler.daily_flags.linearity[1:].all()

    np.testing.assert_array_equal(
        single.daily_scores.linearity, double.daily_scores.linearity
    )


def test_unscored_days_leave_quality_clustering_unchanged():
    labels_by_record_size = []
    for unavailable in (0, 600):
        handler = DataHandler()
        handler.num_days = 20 + unavailable
        handler.daily_scores.density = np.ones(handler.num_days)
        handler.daily_scores.linearity = np.r_[
            np.full(3, 0.11), np.full(17, 0.2), np.full(unavailable, np.nan)
        ]
        handler.get_daily_flags()
        labels = handler.daily_scores.quality_clustering
        labels_by_record_size.append(labels[:20])
        assert np.all(labels[20:] == -1)
        assert not handler.normal_quality_scores

    assert np.all(labels_by_record_size[0] >= 0)
    np.testing.assert_array_equal(*labels_by_record_size)


def test_get_linearity_scores_after_failed_filling(monkeypatch):
    hours = np.arange(96) / 4
    daylight_curve = np.maximum(0, np.sin(np.pi * (hours - 6) / 12))
    handler = DataHandler(raw_data_matrix=np.tile(daylight_curve[:, None], (1, 12)))

    def lose_filled_values(**kwargs):
        handler.filled_data_matrix = np.zeros_like(handler.raw_data_matrix)

    monkeypatch.setattr(handler, "make_filled_data_matrix", lose_filled_values)
    handler.run_pipeline(verbose=False)
    result = handler.report(verbose=False, return_values=True)["stage outcomes"]
    assert result["provisional daylight"]["status"] == "success"
    assert result["matrix filling"]["status"] == "failed"

    handler.get_linearity_scores()

    assert handler.daily_scores is None
    assert handler.boolean_masks.infill is None
    assert handler.boolean_masks.linearity_daylight is None
    np.testing.assert_equal(
        handler.report(verbose=False, return_values=True)["stage outcomes"], result
    )
