import unittest
from pathlib import Path
import numpy as np
import pandas as pd
from solardatatools import standardize_time_axis, make_2d


class TestStandardizeTimeAxis(unittest.TestCase):
    def test_standardize_time_axis(self):
        filepath = Path(__file__).parent.parent
        data_file_path = filepath / "fixtures" / "data_transforms" / "timeseries.csv"
        data = pd.read_csv(data_file_path, index_col=0, parse_dates=True)
        expected_index = pd.date_range(
            data.index.min().normalize(),
            data.index.max().normalize() + pd.Timedelta(days=1),
            freq="5min",
            inclusive="left",
        )
        # The acquisition timestamps already lie on the five-minute grid.
        expected_output = data.reindex(expected_index)
        actual_output, _ = standardize_time_axis(data, timeindex=True)
        pd.testing.assert_frame_equal(
            expected_output, actual_output, check_index_type=False
        )


class TestMake2D(unittest.TestCase):
    def test_make_2d_with_freq_set(self):
        filepath = Path(__file__).parent.parent
        data_file_path = (
            filepath / "fixtures" / "data_transforms" / "timeseries_standardized.csv"
        )
        data = pd.read_csv(data_file_path, index_col=0, parse_dates=True)
        expected_data_file_path = (
            filepath / "fixtures" / "data_transforms" / "power_mat.csv"
        )
        with open(expected_data_file_path) as file:
            expected_output = np.genfromtxt(file, delimiter=",")
        data.index.freq = pd.tseries.offsets.Second(300)
        key = data.columns[0]
        actual_output = make_2d(data, key=key, trim_start=True, trim_end=True)
        np.testing.assert_array_almost_equal(expected_output, actual_output)

    def test_make_2d_no_freq(self):
        filepath = Path(__file__).parent.parent
        data_file_path = (
            filepath / "fixtures" / "data_transforms" / "timeseries_standardized.csv"
        )
        data = pd.read_csv(data_file_path, index_col=0, parse_dates=True)
        expected_data_file_path = (
            filepath / "fixtures" / "data_transforms" / "power_mat.csv"
        )
        with open(expected_data_file_path) as file:
            expected_output = np.genfromtxt(file, delimiter=",")
        key = data.columns[0]
        actual_output = make_2d(data, key=key, trim_start=True, trim_end=True)
        np.testing.assert_array_almost_equal(expected_output, actual_output)


if __name__ == "__main__":
    unittest.main()
