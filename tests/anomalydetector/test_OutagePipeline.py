import unittest
from pathlib import Path
import numpy as np
import pandas as pd
from anomalydetector import MultiDataHandler,OutagePipeline


class TestOutagePipeline(unittest.TestCase):

    def setUp(self):
        self.sites = ['ac_power_inv_30339','ac_power_inv_30342','ac_power_inv_31746']
        self.target = 'ac_power_inv_30339'

    def test_fit_quantiles(self):

        filepath = Path(__file__).parent.parent 
        data_file_path = (
            filepath
            / "fixtures"
            / "anomaly_detection"
            / f"one_month_{self.target}.csv"
        )
        with open(data_file_path) as file:
            data = np.loadtxt(file, delimiter=",")

        common_file_path = (
            filepath
            / "fixtures"
            / "anomaly_detection"
            / "common_day.csv"
        )
        with open(common_file_path) as file :
            common_days = pd.to_datetime(np.loadtxt(file, delimiter=","), unit='ns')
        expected_data_file_path =  (
            filepath
            / "fixtures"
            / "anomaly_detection"
            / f"one_month_transform_{self.target}.csv"
        )

        with open(expected_data_file_path) as file :
            expected_output = np.loadtxt(file, delimiter=",")
        

        #Reconstruct the MultidataHandler from the data
        dhs = MultiDataHandler(data_frames=[],datetime_col='measured_on')
        dhs.dil_mat = {self.target : data}
        dhs.common_days = common_days
        dhs.target = self.target
        dhs.generate_failure(target=dhs.target)
        model = OutagePipeline([self.target],
                dhs.ndil(),
                target=self.target,
                solver_quantiles='mosek')
        model.fit_quantiles(dhs)
        actual_output = model.quantile_train[self.target]
        np.testing.assert_array_almost_equal(actual_output,expected_output,4)
        
    def test_fit_linear(self):
        filepath = Path(__file__).parent.parent 

        data_expected_file_path = (
                filepath
                / "fixtures"
                / "anomaly_detection"
                / "linear_coef.csv"
            )
        with open(data_expected_file_path) as file:
            expected_output = np.loadtxt(file, delimiter=",")
            
        model = OutagePipeline(self.sites,
                101,
                target=self.target,
                solver_quantiles='mosek')
        model.quantile_train = {}
        #reconstruct the OutagePipeline object 
        for site in self.sites :
            data_file_path = (
                filepath
                / "fixtures"
                / "anomaly_detection"
                / f"one_month_transform_{site}.csv"
            )
            with open(data_file_path) as file:
                model.quantile_train[site] = np.loadtxt(file, delimiter=",")

        data_file_path = (
                filepath
                / "fixtures"
                / "anomaly_detection"
                / f"one_month_transform_failure.csv"
            )
        with open(data_file_path) as file:
                data_transform = np.loadtxt(file, delimiter=",")
        model.quantile_failure = data_transform
        model.fit_linear()
        actual_output = model.linear_coeff
        np.testing.assert_array_almost_equal(actual_output,expected_output,5)


if __name__ == "__main__":
    unittest.main()


    