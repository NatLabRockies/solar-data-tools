import unittest
from pathlib import Path
import numpy as np
import pandas as pd
from anomalydetector import MultiDataHandler

#TO DO
""""
class TestMultiDataHandler(unittest.TestCase):

    def setUp(self):
        self.test_site = 'ac_power_inv_30339'

    def test_multidatahandler(self):
        ### Pick the data 
        ###expected_output
        ###dfs
        dhs = MultiDataHandler(data_frames=list(dfs.values()),datetime_col='measured_on')
        dhs.align()
        dhs.dilate(ndil = 51)
        actual_output =  dhs.dil_mat[self.test_site]
        np.testing.assert_array_almost_equal(actual_output,expected_output,3)
"""