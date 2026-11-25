import numpy as np
import pytest

from solardatatools import data_quality
from solardatatools.data_quality import (
    daily_missing_data_simple,
    dataset_quality_score,
    make_density_scores,
    make_quality_flags,
)
from solardatatools.signal_decompositions import DecompositionResult, SolveDiagnostics


def _baseline_result(seasonal):
    if seasonal is None:
        return DecompositionResult(
            np.ones(3, dtype=bool),
            None,
            SolveDiagnostics("CLARABEL", {}, failure_reason="Test solver failure."),
        )
    return DecompositionResult(
        np.ones(len(seasonal), dtype=bool),
        {"seasonal": seasonal},
        SolveDiagnostics("CLARABEL", {}, solver_status="optimal"),
    )


def test_advanced_quality_score_counts_passing_days(monkeypatch):
    data = np.tile([0.0, 0.0, 10.0, 10.0], (8, 1)).T
    data[:, -1] = np.nan

    def fit(signal, *, use_ixs, **kwargs):
        np.testing.assert_array_equal(signal, [0.5] * 7 + [0.0])
        np.testing.assert_array_equal(use_ixs, [True] * 7 + [False])
        # One passing ratio, both strict bounds, two out-of-range ratios,
        # two unavailable scores, and a missing day with a usable baseline.
        return _baseline_result(
            np.array([0.5, 0.5 / 0.6, 0.5 / 1.05, 2.0, 0.0625, 0.0, np.nan, 0.5])
        )

    monkeypatch.setattr(data_quality, "tl1_l2d2p365", fit)
    assert dataset_quality_score(data) == 1 / 8
    assert dataset_quality_score(data, threshold=-1) == 1 / 8


def test_density_entry_points_fit_a_baseline_with_default_arguments():
    data = np.tile([[0.0], [0.0], [10.0], [10.0]], (1, 8))
    data[:, 3] = np.nan
    np.testing.assert_allclose(
        make_density_scores(data), [1, 1, 1, 0, 1, 1, 1, 1], atol=1e-5
    )
    assert dataset_quality_score(data) == 7 / 8


def test_daily_ratios_and_dataset_score_fit_the_same_days():
    data = np.full((4, 21), np.nan)
    data[:, 0] = [0, 0, 10, 10]
    threshold = -0.01
    scores = make_density_scores(data, threshold=threshold)
    passed, _ = make_quality_flags(scores, np.zeros(21))
    assert passed.sum() == 1
    assert dataset_quality_score(data, threshold=threshold) == passed.mean()


def test_baseline_distinguishes_observed_zero_density_from_missing(monkeypatch):
    data = np.array([[0, 0, np.nan], [0, 0, np.nan], [10, 0, np.nan], [10, 0, np.nan]])

    def fit(signal, *, use_ixs, **kwargs):
        np.testing.assert_array_equal(signal, [0.5, 0, 0])
        np.testing.assert_array_equal(use_ixs, [True, True, False])
        return _baseline_result(np.full(3, 0.5))

    monkeypatch.setattr(data_quality, "tl1_l2d2p365", fit)
    np.testing.assert_array_equal(make_density_scores(data, threshold=-0.01), [1, 0, 0])
    assert dataset_quality_score(data, threshold=-0.01) == 1 / 3


def test_density_scoring_uses_only_finite_power_observations():
    data = np.tile([[0.0], [0.0], [10.0], [10.0]], (1, 30))
    data[1, 0] = -np.inf
    data[:, -1] = [np.nan, np.nan, np.inf, np.inf]
    expected_density = np.r_[np.full(29, 0.5), 0]
    density_scores, advanced_density, baseline = make_density_scores(
        data, return_density_signal=True, return_fit=True
    )
    simple_flags, simple_density = daily_missing_data_simple(
        data, return_density_signal=True
    )
    np.testing.assert_array_equal(advanced_density, expected_density)
    np.testing.assert_array_equal(simple_density, expected_density)
    np.testing.assert_allclose(baseline, np.full(30, 0.5), atol=1e-5)
    np.testing.assert_allclose(density_scores, np.r_[np.ones(29), 0], atol=1e-5)
    np.testing.assert_array_equal(simple_flags, np.r_[np.ones(29, dtype=bool), False])
    assert dataset_quality_score(data) == 29 / 30
    assert dataset_quality_score(data, use_advanced=False) == 29 / 30


def test_simple_quality_score_applies_threshold_and_counts_missing_days():
    data = np.array(
        [[0, 0, np.nan], [0, 10, np.nan], [10, 10, np.nan], [10, 10, np.nan]]
    )
    assert dataset_quality_score(data, threshold=0.5, use_advanced=False) == 1 / 3
    assert dataset_quality_score(data, threshold=-1, use_advanced=False) == 2 / 3
    assert dataset_quality_score(np.full((4, 3), np.nan), use_advanced=False) == 0


def test_supplied_boolean_mask_gives_the_passing_fraction():
    data = np.full((4, 3), np.nan)
    assert dataset_quality_score(data, good_days=[True, False, True]) == 2 / 3


@pytest.mark.parametrize("hidden_value", [True, False])
def test_supplied_masked_boolean_decisions_count_as_failures(hidden_value):
    good_days = np.ma.array([True, hidden_value, False], mask=[False, True, False])
    assert dataset_quality_score(np.zeros((4, 3)), good_days=good_days) == 1 / 3


@pytest.mark.parametrize("mask", [[True], [[True, False]], [1, 0], [np.nan, 0]])
def test_supplied_mask_must_be_boolean_and_aligned_with_days(mask):
    with pytest.raises(ValueError, match="good_days must be a Boolean mask"):
        dataset_quality_score(np.zeros((4, 2)), good_days=mask)


@pytest.mark.parametrize("shape", [(4, 0), (0, 2)])
def test_empty_matrix_raises(shape):
    with pytest.raises(ValueError, match="at least one row and one column"):
        dataset_quality_score(np.empty(shape))


def test_advanced_quality_score_requires_days_for_fitting(monkeypatch):
    def unexpected_fit(*args, **kwargs):
        pytest.fail("a baseline fit requires eligible days")

    monkeypatch.setattr(data_quality, "tl1_l2d2p365", unexpected_fit)
    with pytest.raises(ValueError, match="No eligible days"):
        dataset_quality_score(np.zeros((4, 3)))
    with pytest.raises(ValueError, match="No observations"):
        dataset_quality_score(np.full((4, 3), np.nan))


@pytest.mark.parametrize("fit", [None, np.array([np.nan, 0.0, np.inf])])
def test_advanced_quality_score_rejects_unusable_fit(monkeypatch, fit):
    monkeypatch.setattr(
        data_quality, "tl1_l2d2p365", lambda *args, **kwargs: _baseline_result(fit)
    )
    data = np.tile([[0.0], [0.0], [10.0], [10.0]], (1, 3))
    with pytest.raises(ValueError, match="density baseline fit returned no"):
        dataset_quality_score(data)


def test_advanced_quality_score_propagates_solver_failure(monkeypatch):
    def fail_fit(*args, **kwargs):
        raise RuntimeError("fit failed")

    monkeypatch.setattr(data_quality, "tl1_l2d2p365", fail_fit)
    data = np.tile([[0.0], [0.0], [10.0], [10.0]], (1, 3))
    with pytest.raises(RuntimeError, match="fit failed"):
        dataset_quality_score(data)
