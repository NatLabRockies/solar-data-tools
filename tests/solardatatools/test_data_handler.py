import unittest
from pathlib import Path
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import pytest
from solardatatools import DataHandler


@pytest.mark.parametrize(
    ("timestamps", "expected_timestamps", "retained", "changed"),
    [
        (["2024-01-15 12:00"], ["2024-01-15 12:00"], [0], False),
        (["2024-07-15 12:00"], ["2024-07-15 11:00"], [0], True),
        (
            ["2024-03-10 01:30", "2024-03-10 02:30", "2024-03-10 03:30"],
            ["2024-03-10 01:30", "2024-03-10 02:30"],
            [0, 2],
            True,
        ),
        (["2024-11-03 01:30", "2024-11-03 02:30"], ["2024-11-03 02:30"], [1], True),
    ],
    ids=["winter", "summer", "spring-gap", "fall-overlap-removal-only"],
)
def test_fix_dst_reports_per_call_changes(
    timestamps, expected_timestamps, retained, changed, capsys
):
    index = pd.DatetimeIndex(timestamps, name="time")
    source = pd.DataFrame({"power": np.arange(len(index))}, index=index)
    handler = DataHandler(source)
    expected = handler.data_frame_raw.iloc[retained].set_axis(
        pd.DatetimeIndex(expected_timestamps, name="time")
    )

    assert handler.fix_dst() is changed
    pd.testing.assert_frame_equal(handler.data_frame_raw, expected)
    assert capsys.readouterr().out == ""

    assert handler.fix_dst() is False
    assert "already performed" in capsys.readouterr().out
    pd.testing.assert_frame_equal(handler.data_frame_raw, expected)


def test_fix_dst_propagates_failure_and_can_retry(monkeypatch):
    index = pd.DatetimeIndex(["2024-07-15 12:00"])
    handler = DataHandler(pd.DataFrame({"power": [10]}, index=index))
    before = handler.data_frame_raw.copy()

    def fail_conversion(*args, **kwargs):
        raise RuntimeError("test DST conversion failure")

    with monkeypatch.context() as patch:
        patch.setattr(pd.DataFrame, "tz_convert", fail_conversion)
        with pytest.raises(RuntimeError, match="test DST conversion failure"):
            handler.fix_dst()
    pd.testing.assert_frame_equal(handler.data_frame_raw, before)

    assert handler.fix_dst() is True
    expected = before.set_axis(pd.DatetimeIndex(["2024-07-15 11:00"]))
    pd.testing.assert_frame_equal(handler.data_frame_raw, expected)


class TestDataHandler(unittest.TestCase):
    def test_plot_extra_matrix_preserves_watt_values(self):
        dh = DataHandler()
        dh.power_units = "W"
        watts = np.array([[0.0, np.nan], [1250.0, 2500.0]])
        dh.extra_matrices["sensor"] = watts.copy()

        for _ in range(2):
            fig = dh.plot_heatmap(matrix="sensor")
            try:
                np.testing.assert_array_equal(
                    fig.axes[0].images[0].get_array().filled(np.nan), watts / 1000
                )
                self.assertEqual(fig.axes[1].get_ylabel(), "kW")
            finally:
                plt.close(fig)

        np.testing.assert_array_equal(dh.extra_matrices["sensor"], watts)

    def test_load_and_run(self):
        filepath = Path(__file__).parent.parent
        data_file_path = filepath / "fixtures" / "data_transforms" / "timeseries.csv"
        df = pd.read_csv(data_file_path, parse_dates=[0], index_col=0)
        dh = DataHandler(df)
        dh.fix_dst()
        dh.run_pipeline(power_col="ac_power_01", fix_shifts=True, verbose=False)
        # dh.report()
        self.assertAlmostEqual(dh.capacity_estimate, 6.7453649044036865, places=2)
        self.assertAlmostEqual(dh.data_quality_score, 0.9948186528497409, places=3)
        self.assertAlmostEqual(dh.data_clearness_score, 0.49222797927461137, places=3)
        self.assertTrue(dh.inverter_clipping)
        self.assertFalse(dh.time_shifts)
        scores_path = filepath / "fixtures" / "scoring" / "clipping_1.csv"
        with open(scores_path) as file:
            expected_scores = np.loadtxt(file, delimiter=",")
        np.testing.assert_allclose(
            dh.daily_scores.clipping_1, expected_scores, atol=1e-3
        )
        scores_path = filepath / "fixtures" / "scoring" / "clipping_2.csv"
        with open(scores_path) as file:
            expected_scores = np.loadtxt(file, delimiter=",")
        np.testing.assert_allclose(
            dh.daily_scores.clipping_2, expected_scores, atol=2e-3
        )
        scores_path = filepath / "fixtures" / "scoring" / "density.csv"
        with open(scores_path) as file:
            expected_scores = np.loadtxt(file, delimiter=",")
        # The old fixture counted copies at the partial first day's boundary.
        # Count the actual source observations on that day independently.
        first_day = dh.day_index[0].date().isoformat()
        observed = dh.data_frame_raw.loc[first_day, "ac_power_01"]
        density = np.count_nonzero(observed > 0.02 * dh.capacity_estimate)
        density /= dh.raw_data_matrix.shape[0]
        self.assertAlmostEqual(
            dh.daily_scores.density[0],
            density / dh.daily_signals.seasonal_density_fit[0],
        )
        np.testing.assert_allclose(
            dh.daily_scores.density[1:], expected_scores[1:], atol=1e-3
        )
        scores_path = filepath / "fixtures" / "scoring" / "linearity.csv"
        with open(scores_path) as file:
            expected_scores = np.loadtxt(file, delimiter=",")
        np.testing.assert_allclose(
            dh.daily_scores.linearity, expected_scores, atol=2e-2
        )
        scores_path = filepath / "fixtures" / "scoring" / "quality_clustering.csv"
        with open(scores_path) as file:
            expected_scores = np.loadtxt(file, delimiter=",")
        np.testing.assert_allclose(
            dh.daily_scores.quality_clustering, expected_scores, atol=1e-3
        )
