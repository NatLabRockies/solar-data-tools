# -*- coding: utf-8 -*-
"""
Signal Decompositions Module

This module contains standardized signal decomposition models for use in the
SDT algorithms. The defined signal decompositions are:

1. `l2_l1d1_l2d2p365`: separating a piecewise constant component from a smooth
   and seasonal component, with Gaussian noise
   - l2: gaussian noise, sum-of-squares small or l2-norm squared
   - l1d1: piecewise constant heuristic, l1-norm of first order differences
   - l2d2p365: small second order diffs (smooth) and 365-periodic

2. `tl1_l2d2p365`: fitting an annual Fourier component to a quantile of a
   daily signal with pinball loss and a quadratic penalty on the harmonics.

3. `l1_l1d1_l2d2p365`: like (1) but with an asymmetric residual cost instead
   of Gaussian residuals
   - l1: l1-norm
   - l1d1: piecewise constant heuristic, l1-norm of first order differences
   - l2d2p365: small second order diffs (smooth) and 365-periodic

4. `l2_l1d2_constrained`:
   - l2: gaussian noise, sum-of-squares small or l2-norm squared
   - l1d2: piecewise linear heuristic
   - constrained to have first val at 0 and last val at 1

"""

from solardatatools._osd_signal_decompositions import (
    _osd_l2_l1d1_l2d2p365,
)
from solardatatools._cvx_signal_decompositions import (
    _cvx_l2_l1d1_l2d2p365,
    _cvx_l1_pwc_smoothper_trend,
)
from solardatatools._decomposition_result import DecompositionResult, SolveDiagnostics
from solardatatools._l2_l1d2_constrained import (
    _L2L1D2ConstrainedWorkspace,
    _validate_signal,
)
from solardatatools._tl1_l2d2p365 import _fit_quantile

__all__ = [
    "DecompositionResult",
    "SolveDiagnostics",
    "l1_pwc_smoothper_trend",
    "l2_l1d1_l2d2p365",
    "l2_l1d2_constrained",
    "tl1_l2d2p365",
]


def l2_l1d1_l2d2p365(
    signal,
    use_ixs=None,
    w1=50,
    w2=1e-3,
    yearly_periodic=False,
    return_all=False,  # for unit tests only
    solver="CLARABEL",
    sum_card=False,  # OSD only
    transition_locs=None,  # CVX only
    verbose=False,
):
    """
    Used in: solardatatools/algorithms/time_shifts.py

    This performs total variation filtering with the addition of a seasonal
    baseline fit. This introduces a new signal to the model that is smooth and
    periodic on a yearly time frame. This does a better job of describing real,
    multi-year solar PV power data sets, and therefore does an improved job of
    estimating the discretely changing signal. Default solver is QSS and timeshift
    algorithm takes the final solution from solving the nonconvex problem
    with sum_card=True.

    :param signal: A 1d numpy array (must support boolean indexing) containing
        the signal of interest
    :param w0: Weight on the residual component
    :param w1: The regularization parameter to control the total variation in
        the final output signal
    :param w2: The regularization parameter to control the smoothness of the
        seasonal signal
    :param yearly_periodic: Adds periodicity constraint to signal decomposition
    :param return_all: Returns all components and the objective value. Used for tests.
    :param solver: Solver to use for the decomposition. Supported solvers are CLARABEL and MOSEK.
    :param sum_card: Boolean for using the nonconvex formulation using the cardinality penalty,
        Supported only using OSD with the QSS solver.
    :param transition_locs: List of indices where transitions are located. Only used in CVXPY problems.
    :param verbose: Sets verbosity
    :return: A tuple with two 1d numpy arrays containing the two signal component estimates
    """
    if solver == "MOSEK":
        res = _cvx_l2_l1d1_l2d2p365(
            signal=signal,
            use_ixs=use_ixs,
            w0=1,
            w1=w1,
            w2=w2,
            yearly_periodic=yearly_periodic,
            return_all=return_all,
            transition_locs=transition_locs,
            solver=solver,
            verbose=verbose,
        )
    elif yearly_periodic:
        res = _osd_l2_l1d1_l2d2p365(
            signal=signal,
            use_ixs=use_ixs,
            w1=w1,
            w2=w2,
            yearly_periodic=yearly_periodic,
            return_all=return_all,
            transition_locs=transition_locs,
            solver=solver,
            sum_card=False,
            verbose=verbose,
        )
    else:
        res = _osd_l2_l1d1_l2d2p365(
            signal=signal,
            use_ixs=use_ixs,
            w1=w1,
            w2=w2,
            yearly_periodic=yearly_periodic,
            return_all=return_all,
            transition_locs=transition_locs,
            solver=solver,
            sum_card=False,
            verbose=verbose,
        )

    return res


def tl1_l2d2p365(
    signal,
    use_ixs=None,
    tau=0.75,
    w0=1,
    *,
    solver="CLARABEL",
    solver_options=None,
    verbose=False,
) -> DecompositionResult:
    """Fit a three-harmonic annual Fourier quantile model to daily samples.

    The 365.2425-day basis spans the full input grid and includes an
    unpenalized offset. The quadratic penalty on the sine and cosine
    coefficients is proportional to the squared harmonic numbers.

    :param signal: Nonempty real one-dimensional array with one entry per day.
    :param use_ixs: Optional Boolean mask matching signal. Admitted samples
        are the selected finite entries, or all finite entries by default.
    :param tau: Quantile level strictly between zero and one.
    :param w0: Finite positive weight on the sum of pinball losses.
    :param solver: CVXPY solver name. CLARABEL is the default.
    :param solver_options: Keyword arguments forwarded to the selected solver.
    :param verbose: Enable solver output.
    :returns: Full-grid seasonal and residual components, the admitted mask,
        and solve diagnostics. Components are unavailable when the fit fails.
        The residual is zero outside the admitted observations.
    """
    return _fit_quantile(signal, use_ixs, tau, w0, solver, solver_options, verbose)


def l1_pwc_smoothper_trend(
    signal,
    use_ixs=None,
    w2=2e1,
    w3=1,
    w4=1e1,
    solver="CLARABEL",
    verbose=False,
    return_all=False,
):
    """
    Used in solardatatools/algorithms/capacity_change.py

    We solve a convex signal decomposition problem, making use of the l1-sparsity heuristic for estimating a
    piecewise constant component. We use a single pass of iterative reweighting on this term to "polish" the sparse
    solution (see: https://web.stanford.edu/~boyd/papers/rwl1.html).

    :param signal: A 1d numpy array (must support boolean indexing) containing
        the signal of interest
    :param use_ixs: List of booleans indicating indices to use in signal.
        None is default (uses the entire signal).
    :param w2: Weight on the piecewise constant component
    :param w3: Weight on the smooth, periodic component
    :param w4: Weight on the slope of the trend term (discourages large trends)
    :param solver: Solver to use for the decomposition. Supported solvers are CLARABEL and MOSEK.
    :param verbose: Sets verbosity
    :return: A tuple with three 1d numpy arrays containing the three non-noise signal component estimates
    """
    res = _cvx_l1_pwc_smoothper_trend(
        signal=signal,
        use_ixs=use_ixs,
        w2=w2,
        w3=w3,
        w4=w4,
        solver=solver,
        verbose=verbose,
        return_all=return_all,
    )
    return res


def l2_l1d2_constrained(
    signal,
    *,
    w0=1,
    w1=5,
    use_ixs=None,
    solver="CLARABEL",
    solver_options=None,
    verbose=False,
) -> DecompositionResult:
    """Fit a trend with endpoints fixed at zero and one.

    Minimize the weighted sum of squared residuals and the weighted L1 norm
    of the sample-index second difference. Both components span the input
    grid. The consistency constraint applies at the admitted observations.
    Off that set, the squared residual cost gives zero for positive w0,
    and the returned result chooses zero when w0 is zero.

    :param signal: Real one-dimensional signal with at least three entries.
    :param w0: Nonnegative weight on the sum of squared residuals.
    :param w1: Nonnegative weight on absolute second differences. It must be
        positive if an interior sample is excluded. At least one weight must
        be positive.
    :param use_ixs: Optional Boolean mask. Only selected finite values enter
        the consistency constraint.
    :param solver: CVXPY solver name. CLARABEL is the default.
    :param solver_options: Keyword arguments forwarded to CVXPY's solve method.
        The default quadratic CLARABEL path adds ``ignore_dpp=True`` to avoid
        a dense compilation intermediate. Explicit ``ignore_dpp``,
        ``enforce_dpp=True``, or conic ``use_quad_obj=False`` choices are
        preserved. Effective options are recorded in the diagnostics.
    :param verbose: Enable solver output.
    :returns: Named trend and residual components with an observation mask and
        solve diagnostics. Components are unavailable when the fit fails.
    """
    signal = _validate_signal(signal)
    return _L2L1D2ConstrainedWorkspace(signal.size).solve(
        signal,
        w0=w0,
        w1=w1,
        use_ixs=use_ixs,
        solver=solver,
        solver_options=solver_options,
        verbose=verbose,
    )
