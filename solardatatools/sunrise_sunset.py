"""Sunrise and Sunset Estimation Module

This module contains functions for estimating sunrise and sunset times from an
unlabel PV power dataset.
"""

import numpy as np
from solardatatools._tl1_l2d2p365 import _seasonal_component
from solardatatools.signal_decompositions import tl1_l2d2p365


def rise_set_rough(bool_msk):
    nvals = bool_msk.shape[0]
    hour_of_day = 24 * np.arange(nvals) / nvals
    sunrise_idxs = np.argmax(bool_msk, axis=0)
    sunset_idxs = nvals - np.argmax(np.flip(bool_msk, axis=0), axis=0) - 1
    sunrises = np.nan * np.ones_like(sunrise_idxs)
    sunsets = np.nan * np.ones_like(sunset_idxs)
    sunrises[sunrise_idxs != 0] = hour_of_day[sunrise_idxs][sunrise_idxs != 0]
    sunsets[sunset_idxs != nvals - 1] = hour_of_day[sunset_idxs][
        sunset_idxs != nvals - 1
    ]
    # sunrises[np.isnan(sunrises)] = 1000
    # sunrises[sunrises > 12] = np.nan
    # sunsets[np.isnan(sunsets)] = -1000
    # sunsets[sunsets < 12] = np.nan
    return {"sunrises": sunrises, "sunsets": sunsets}


def rise_set_smoothed(
    rough_dict, sunrise_tau=0.1, sunset_tau=0.9, solver=None, *, solver_options=None
):
    return _smooth_sunrise_sunset(
        rough_dict, sunrise_tau, sunset_tau, solver, solver_options
    )


def _smooth_sunrise_sunset(
    rough_dict, sunrise_tau, sunset_tau, solver, solver_options, fit_warnings=None
):
    sunrises = rough_dict["sunrises"]
    sunsets = rough_dict["sunsets"]

    sunrise_fit = tl1_l2d2p365(
        sunrises, tau=sunrise_tau, solver=solver, solver_options=solver_options
    )
    sunset_fit = tl1_l2d2p365(
        sunsets, tau=sunset_tau, solver=solver, solver_options=solver_options
    )
    return {
        "sunrises": _seasonal_component(sunrise_fit, "sunrise", fit_warnings),
        "sunsets": _seasonal_component(sunset_fit, "sunset", fit_warnings),
    }
