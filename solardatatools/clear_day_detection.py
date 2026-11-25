# -*- coding: utf-8 -*-
"""Clear Day Detection Module

This module contains functions for detecting clear days in historical PV solar data sets.

"""

import numpy as np
from solardatatools._tl1_l2d2p365 import _seasonal_component
from solardatatools.signal_decompositions import tl1_l2d2p365
from solardatatools.utilities import basic_outlier_filter


class ClearDayDetection:
    def __init__(self):
        self.y = None
        self.tc = None
        self.de = None
        self.x = None
        self.density_signal = None
        self.filtered_signal = None
        self.weights = None
        self._fit_warnings = {}

    @property
    def fit_warnings(self):
        """Accuracy warnings for the baseline fits."""
        return tuple(
            warning for warnings in self._fit_warnings.values() for warning in warnings
        )

    def _fit_baseline(self, signal, *, name, tau, solver, solver_options, use_ixs=None):
        fit_warnings = []
        self._fit_warnings[name] = fit_warnings
        result = tl1_l2d2p365(
            signal,
            use_ixs=use_ixs,
            tau=tau,
            solver=solver,
            solver_options=solver_options,
        )
        return _seasonal_component(result, name, fit_warnings)

    def filter_for_sparsity(
        self, data, w1=6e3, solver="CLARABEL", *, solver_options=None
    ):
        capacity_est = np.nanquantile(data, 0.95)
        # set nans to zero to avoid issues w/ summing
        data_copy = np.copy(data)
        data_copy[np.isnan(data)] = 0.0
        foo = data_copy > 0.02 * capacity_est  # 2% of 95th perc
        self.density_signal = np.sum(foo, axis=0) / data.shape[0]
        use_days = np.logical_and(self.density_signal > 0.2, self.density_signal < 0.8)

        self.filtered_signal = self._fit_baseline(
            self.density_signal,
            name="clear-day density baseline",
            use_ixs=use_days,
            tau=0.85,
            solver=solver,
            solver_options=solver_options,
        )
        mask = basic_outlier_filter(
            self.density_signal - self.filtered_signal, outlier_constant=5.0
        )
        return mask

    def find_clear_days(
        self,
        data,
        smoothness_threshold=0.9,
        energy_threshold=0.8,
        boolean_out=True,
        solver="CLARABEL",
        *,
        solver_options=None,
    ):
        """
        This function quickly finds clear days in a PV power data set. The input to this function is a 2D array containing
        standardized time series power data. This will typically be the output from
        `solardatatools.data_transforms.make_2d`. The filter relies on two estimates of daily "clearness": the smoothness
        of each daily signal as measured by the l2-norm of the 2nd order difference, and seasonally-adjusted daily
        energy. Seasonal adjustment of the daily energy if obtained by solving a local quantile regression problem, which
        is a convex optimization problem and is solvable with cvxpy. The parameter `th` controls the relative weighting of
        the daily smoothness and daily energy in the final filter in a geometric mean. A value of 0 will rely entirely on
        the daily energy and a value of 1 will rely entirely on daily smoothness.

        :param D: A 2D numpy array containing a solar power time series signal.
        :param th: A parameter that tunes the filter between relying of daily smoothness and daily energy
        :return: A 1D boolean array, with `True` values corresponding to clear days in the data set
        """
        self._fit_warnings = {}
        # Take the norm of the second different of each day's signal. This gives a rough estimate of the smoothness of
        # day in the data set
        tc = np.linalg.norm(data[:-2] - 2 * data[1:-1] + data[2:], ord=1, axis=0)
        # Shift this metric so the median is at zero
        # tc = np.percentile(tc, 50) - tc
        # Normalize such that the maximum value is equal to one
        tc /= np.nanmax(tc)
        self.tc = 1 - tc
        # Seasonal renormalization: estimate a "baseline smoothness" based on local
        # 90th percentile of smoothness signal. This has the effect of increasing
        # the score of days if there aren't very many smooth days nearby
        # 7/24/23 SM: Adjusted weight down from 2.5e6 to 2.5e5
        # due to failed decompositions on some datasets (TABJC1001611)
        self.y = self._fit_baseline(
            self.tc,
            name="clear-day smoothness baseline",
            tau=0.9,
            solver=solver,
            solver_options=solver_options,
        )
        # A zero smoothness proxy and zero baseline give an undefined ratio.
        tc = np.divide(
            self.tc,
            self.y,
            out=np.full_like(self.tc, np.nan),
            where=(self.tc != 0) | (self.y != 0),
        )
        # Take the positive part function, i.e. set the negative values to zero.
        # This is the first metric
        tc = np.clip(tc, 0, None)
        # Calculate the daily energy
        de = np.sum(data, axis=0)
        # Scale by max
        self.de = de / np.nanmax(de)
        # Solve a convex minimization problem to roughly fit the local 90th
        # percentile of the data (quantile regression)
        self.x = self._fit_baseline(
            self.de,
            name="clear-day energy baseline",
            tau=0.9,
            solver=solver,
            solver_options=solver_options,
        )
        # x gives us the local top 90th percentile of daily energy, i.e. the very sunny days. This gives us our
        # seasonal normalization.
        de = np.clip(np.divide(self.de, self.x), 0, 1)
        # Take geometric mean
        weights = np.multiply(np.power(tc, 0.5), np.power(de, 0.5))
        # Set values less than 0.6 to be equal to zero
        # weights[weights < 0.6] = 0.
        # Selection rule
        selection = np.logical_and(tc > smoothness_threshold, de > energy_threshold)
        weights[~selection] = 0.0
        # Apply filter for sparsity to catch data errors related to non-zero nighttime data
        msk = self.filter_for_sparsity(
            data, solver=solver, solver_options=solver_options
        )
        self.weights = weights * msk.astype(int)
        if boolean_out:
            return self.weights >= 1e-3
        else:
            return self.weights

    def plot_analysis(self, figsize=None):
        if self.tc is not None and self.de is not None:
            tc = self.tc
            de = self.de
            x = self.x
            y = self.y
            density_signal = self.density_signal
            filtered_signal = self.filtered_signal
            import matplotlib.pyplot as plt

            figsize = (8, 8) if figsize is None else figsize

            fig, ax = plt.subplots(nrows=3, sharex=True, figsize=figsize)
            ax[0].plot(tc, marker=".", linewidth=1)
            ax[0].plot(y, linewidth=1, label="estimated")
            ax[0].set_title("baseline smoothness")
            ax[1].plot(de, linewidth=1, marker=".")
            ax[1].plot(x, linewidth=1, label="estimated")
            ax[1].set_title("daily energy")
            ax[2].plot(density_signal, linewidth=1, marker=".")
            ax[2].set_title("daily density")
            ax[2].plot(filtered_signal, linewidth=1, label="estimated")
            plt.legend()
            plt.tight_layout()
            return fig
