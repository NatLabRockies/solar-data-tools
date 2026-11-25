"""Input and solve settings shared by native signal decompositions."""

from collections.abc import Mapping
from copy import deepcopy

import cvxpy as cp
import numpy as np


_FEASIBILITY_TOLERANCE = 1e-6
_RESERVED_OPTIONS = {
    "solver",
    "verbose",
    "solver_path",
    "method",
    "gp",
    "qcp",
    "requires_grad",
}


def _observation_mask(signal, use_ixs):
    mask = np.isfinite(signal)
    if use_ixs is not None:
        use_ixs = np.asarray(use_ixs)
        if use_ixs.shape != signal.shape or use_ixs.dtype.kind != "b":
            raise ValueError("use_ixs must be a Boolean array matching signal.")
        mask &= use_ixs
    return mask


def _solver_configuration(solver, solver_options):
    solver = "CLARABEL" if solver is None else solver
    if not isinstance(solver, str) or solver not in cp.installed_solvers():
        raise ValueError(f"Solver {solver!r} is unavailable in CVXPY.")
    if solver_options is not None and not isinstance(solver_options, Mapping):
        raise TypeError("solver_options must be a mapping.")
    options = dict(solver_options) if solver_options is not None else {}
    if reserved := _RESERVED_OPTIONS.intersection(options):
        raise ValueError(
            f"Unsupported solver_options keys: {', '.join(sorted(reserved))}."
        )
    return solver, options


def _copy_solver_options(options):
    snapshot = {}
    for name, value in options.items():
        try:
            snapshot[name] = deepcopy(value)
        except TypeError:
            # Some backends accept native handles, such as a solver environment.
            snapshot[name] = value
    return snapshot
