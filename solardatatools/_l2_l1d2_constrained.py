r"""Decompose a sampled curve into a residual and an endpoint-anchored trend.

Let :math:`y` be a curve sampled on a regular grid of :math:`T` points.
The two components :math:`x_0, x_1 \in \mathbf{R}^T` are defined over the
whole grid. The known set :math:`\mathcal{K}` contains the indices of the
finite samples admitted by the observation mask. The decomposition solves

.. math::

    \begin{array}{ll}
    \text{minimize over } x_0, x_1 &
        w_0 \|x_0\|_2^2 + w_1 \|D^2 x_1\|_1 \\
    \text{subject to} & x_{0,t} + x_{1,t} = y_t,\quad t \in \mathcal{K}, \\
                      & x_{1,0} = 0,\quad x_{1,T-1} = 1.
    \end{array}

Each component is specified by its cost and its feasible set.

* The residual :math:`x_0` has cost :math:`w_0\|x_0\|_2^2` and feasible
  set :math:`\mathbf{R}^T`. It accounts for departures from the fitted trend.
* The trend :math:`x_1` has cost :math:`w_1\|D^2x_1\|_1` and feasible
  set :math:`\{x_1 \in \mathbf{R}^T \mid x_{1,0}=0,\ x_{1,T-1}=1\}`.
  The second difference :math:`(D^2x_1)_t=x_{1,t+2}-2x_{1,t+1}+x_{1,t}`
  is the change in successive increments. Its :math:`\ell_1` cost favors
  sparse changes in slope and hence a piecewise-linear trend.

The consistency constraint couples the components only on :math:`\mathcal{K}`.
At missing or held-out points, minimizing the residual cost gives zero
when :math:`w_0>0`. With :math:`w_0=0` those residual entries are
undetermined, and the returned result chooses zero. The trend supplies the
fitted values at those points.

For the CVXPY implementation, see
``_L2L1D2ConstrainedWorkspace._build_problem`` below: ``cost`` defines the
objective, and ``constraints`` defines the consistency and endpoint equations.
The full-grid variables are declared in ``_L2L1D2ConstrainedWorkspace.__init__``.

In clipping detection, :math:`y` samples the empirical CDF of normalized
daily power maxima on an evenly spaced power grid. The first differences of
the fitted trend estimate the shape of the density used to locate candidate
clipping levels.
"""

from collections.abc import Mapping
from copy import deepcopy
from dataclasses import replace

import cvxpy as cp
import numpy as np

from solardatatools._decomposition_result import DecompositionResult, SolveDiagnostics


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


def _validate_signal(signal):
    signal = np.asarray(signal)
    if signal.ndim != 1 or signal.size < 3 or signal.dtype.kind not in "biuf":
        raise ValueError("signal must be a real 1-D array with at least three entries.")
    return signal.astype(float, copy=False)


def _validate_weights(w0, w1):
    weights = np.asarray([w0, w1])
    if (
        weights.shape != (2,)
        or weights.dtype.kind not in "biuf"
        or not np.all(np.isfinite(weights))
        or np.any(weights < 0)
        or not np.any(weights > 0)
    ):
        raise ValueError(
            "w0 and w1 must be finite nonnegative scalars, with one positive."
        )
    return tuple(map(float, weights))


def _copy_solver_options(options):
    snapshot = {}
    for name, value in options.items():
        try:
            snapshot[name] = deepcopy(value)
        except TypeError:
            # Some backends accept native handles, such as a solver environment.
            snapshot[name] = value
    return snapshot


class _L2L1D2ConstrainedWorkspace:
    """Reuse a DPP problem for a fixed known set and return fit snapshots."""

    def __init__(self, size):
        self.size = size
        self._known = None
        self._w0 = cp.Parameter(nonneg=True)
        self._w1 = cp.Parameter(nonneg=True)
        self._trend = cp.Variable(size, name="trend")
        self._residual = cp.Variable(size, name="residual")

    def _build_problem(self, known):
        self._known = known.copy()
        self._data = cp.Parameter(int(known.sum()))
        cost = self._w0 * cp.sum_squares(self._residual)
        cost += self._w1 * cp.norm1(cp.diff(self._trend, k=2))
        constraints = [
            self._residual[known] + self._trend[known] == self._data,
            self._trend[0] == 0,
            self._trend[-1] == 1,
        ]
        self._problem = cp.Problem(cp.Minimize(cost), constraints)
        self._solver_settings = None

    def solve(
        self,
        signal,
        *,
        w0=1,
        w1=5,
        use_ixs=None,
        solver=None,
        solver_options=None,
        verbose=False,
    ):
        signal = _validate_signal(signal)
        if signal.size != self.size:
            raise ValueError("signal length must match the filter workspace.")
        mask = np.isfinite(signal)
        if use_ixs is not None:
            use_ixs = np.asarray(use_ixs)
            if use_ixs.shape != signal.shape or use_ixs.dtype.kind != "b":
                raise ValueError("use_ixs must be a Boolean array matching signal.")
            mask &= use_ixs
        w0, w1 = _validate_weights(w0, w1)
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
        diagnostics = SolveDiagnostics(solver, _copy_solver_options(options))

        if not mask.any():
            return DecompositionResult(
                mask,
                None,
                replace(diagnostics, failure_reason="No admitted observations."),
            )
        if w1 == 0 and not mask[1:-1].all():
            raise ValueError("w1 must be positive when interior samples are excluded.")

        if not np.array_equal(mask, self._known):
            self._build_problem(mask)
        self._data.value = signal[mask]
        self._w0.value = w0
        self._w1.value = w1

        # Some CVXPY adapters reuse solver settings as well as numerical state.
        # Start with a fresh solver when the caller changes the configuration.
        settings = (solver, _copy_solver_options(options), verbose)
        try:
            same_settings = self._solver_settings == settings
        except (TypeError, ValueError):
            same_settings = False
        if self._solver_settings is not None and not same_settings:
            self._problem = cp.Problem(
                self._problem.objective, self._problem.constraints
            )
        self._solver_settings = settings

        try:
            self._problem.solve(solver=solver, verbose=verbose, **options)
        except cp.error.SolverError as error:
            self._solver_settings = None
            self._problem = cp.Problem(
                self._problem.objective, self._problem.constraints
            )
            # Some adapters wrap a configuration error in SolverError.
            if isinstance(
                error.__cause__ or error.__context__, (TypeError, ValueError)
            ):
                raise ValueError(
                    f"{solver} rejected the solver configuration: {error}"
                ) from error
            return DecompositionResult(
                mask,
                None,
                replace(diagnostics, failure_reason=f"{solver} failed: {error}"),
            )

        return self._snapshot(signal, mask, diagnostics)

    def _snapshot(self, signal, mask, diagnostics):
        status = self._problem.status
        objective = self._problem.value
        trend, residual = self._trend.value, self._residual.value
        finite_objective = objective is not None and np.isfinite(objective)
        finite_components = all(
            values is not None and np.isfinite(values).all()
            for values in (trend, residual)
        )
        diagnostics = replace(
            diagnostics,
            solver_status=status,
            objective_value=float(objective) if finite_objective else None,
        )
        if finite_components:
            linking_error = float(
                np.max(np.abs(residual[mask] + trend[mask] - signal[mask]))
            )
            endpoint_error = float(max(abs(trend[0]), abs(trend[-1] - 1)))
            diagnostics = replace(
                diagnostics,
                constraint_errors={
                    "linking": linking_error,
                    "endpoint": endpoint_error,
                },
            )

        if status not in (cp.OPTIMAL, cp.OPTIMAL_INACCURATE):
            reason = f"{diagnostics.solver} returned {status}."
        elif not (finite_components and finite_objective):
            reason = "The solver did not return finite components and objective."
        elif (
            not np.isfinite(linking_error)
            or linking_error
            > _FEASIBILITY_TOLERANCE * max(1, np.max(np.abs(signal[mask])))
            or endpoint_error > _FEASIBILITY_TOLERANCE
        ):
            reason = (
                f"Feasibility check failed (linking error {linking_error:.3g}, "
                f"endpoint error {endpoint_error:.3g})."
            )
        else:
            if status == cp.OPTIMAL_INACCURATE:
                diagnostics = replace(
                    diagnostics,
                    warnings=(f"{diagnostics.solver} returned optimal_inaccurate.",),
                )
            fitted_residual = np.zeros(self.size)
            fitted_residual[mask] = signal[mask] - trend[mask]
            components = {"residual": fitted_residual, "trend": trend.copy()}
            return DecompositionResult(mask, components, diagnostics)

        return DecompositionResult(
            mask, None, replace(diagnostics, failure_reason=reason)
        )
