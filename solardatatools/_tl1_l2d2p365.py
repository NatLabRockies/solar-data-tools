r"""Annual Fourier quantile model on a daily grid.

The seasonal component :math:`x_1=B\theta` uses an unpenalized constant
and cosine/sine pairs at harmonics 1 through 3 of the 365.2425-day period.
For coefficient order (constant, cosine 1, sine 1, cosine 2, sine 2,
cosine 3, sine 3), let

.. math::

    D=\frac{2\pi\,10^{-3}}{\sqrt{365.2425}}\operatorname{diag}(0,1,1,2,2,3,3).

The objective is the weighted sum of pinball losses on the full-grid
residual :math:`x_0` plus :math:`\tfrac12\|D\theta\|_2^2`. The linking
constraint :math:`x_{0,t}+x_{1,t}=y_t` applies on the known set
:math:`\mathcal{K}`. The implementation eliminates the residual entries
outside that set at their unique minimum, zero.
"""

from dataclasses import replace

import cvxpy as cp
import numpy as np

from solardatatools._decomposition_result import DecompositionResult, SolveDiagnostics
from solardatatools._decomposition_utils import (
    _FEASIBILITY_TOLERANCE,
    _copy_solver_options,
    _observation_mask,
    _solver_configuration,
)


_PERIOD_DAYS = 365.2425
_FOURIER_WEIGHT = 1e-3


def _recover_harmonics(basis, mask, centered_fit, coefficients, harmonics):
    """Minimize the harmonic penalty with the admitted fitted values fixed."""
    waves = basis[:, 1:]
    admitted_waves = waves[mask]
    # Subtract the first row to eliminate the free intercept. In weighted
    # coordinates, the minimum-penalty solution is the minimum-norm solution.
    contrast = admitted_waves[1:] - admitted_waves[0]
    fitted_values = centered_fit[mask]
    weighted, _, rank, _ = np.linalg.lstsq(
        contrast / harmonics, fitted_values[1:] - fitted_values[0], rcond=None
    )
    if rank == harmonics.size:
        return centered_fit, coefficients[1:]
    amplitudes = weighted / harmonics
    seasonal = fitted_values[0] + (waves - admitted_waves[0]) @ amplitudes
    return seasonal, amplitudes


def _pinball_sum(residual, tau):
    return np.maximum(tau * residual, (tau - 1) * residual).sum()


def _fit_quantile(signal, use_ixs, tau, w0, solver, solver_options, verbose):
    signal = np.ma.asarray(signal)
    if signal.ndim != 1 or not signal.size or signal.dtype.kind not in "biuf":
        raise ValueError("signal must be a nonempty real one-dimensional array.")
    signal = signal.astype(float, copy=False).filled(np.nan)
    if np.ma.isMaskedArray(use_ixs):
        use_ixs = use_ixs.filled(False)
    mask = _observation_mask(signal, use_ixs)

    tau, w0 = np.asarray(tau), np.asarray(w0)
    if tau.ndim or tau.dtype.kind not in "biuf" or not 0 < float(tau) < 1:
        raise ValueError("tau must be a real scalar strictly between 0 and 1.")
    if w0.ndim or w0.dtype.kind not in "biuf" or not 0 < float(w0) < np.inf:
        raise ValueError("w0 must be a finite positive real scalar.")
    tau, w0 = float(tau), float(w0)

    solver, options = _solver_configuration(solver, solver_options)
    diagnostics = SolveDiagnostics(solver, _copy_solver_options(options))
    if not mask.any():
        diagnostics = replace(diagnostics, failure_reason="No admitted observations.")
        return DecompositionResult(mask, None, diagnostics)

    days = np.arange(signal.size)
    harmonics = np.arange(1, 4)
    phase = (2 * np.pi / _PERIOD_DAYS) * days[:, None] * harmonics
    basis = np.ones((signal.size, 1 + 2 * harmonics.size))
    basis[:, 1::2] = np.cos(phase)
    basis[:, 2::2] = np.sin(phase)
    harmonic_numbers = np.repeat(harmonics, 2)
    weights = (2 * np.pi * _FOURIER_WEIGHT / np.sqrt(_PERIOD_DAYS)) * harmonic_numbers

    coefficients = cp.Variable(basis.shape[1], name="fourier_coefficients")
    residual = cp.Variable(int(mask.sum()), name="observed_residual")
    observed = signal[mask]
    offset = np.quantile(observed, tau, method="inverted_cdf")
    with np.errstate(over="ignore"):
        centered_observations = observed - offset
    if not np.isfinite(centered_observations).all():
        offset, centered_observations = 0.0, observed
    centered_seasonal = basis @ coefficients
    pinball = cp.maximum(tau * residual, (tau - 1) * residual)
    # Translate the free intercept and scale the entire objective to improve
    # numerical resolution when the residual weight is small.
    objective_scale = min(1.0, np.sqrt(w0))
    cost = (w0 / objective_scale) * cp.sum(pinball) + 0.5 * cp.sum_squares(
        cp.multiply(weights / np.sqrt(objective_scale), coefficients[1:])
    )
    problem = cp.Problem(
        cp.Minimize(cost),
        [residual + centered_seasonal[mask] == centered_observations],
    )
    try:
        problem.solve(solver=solver, verbose=verbose, **options)
    except cp.error.SolverError as error:
        if isinstance(error.__cause__ or error.__context__, (TypeError, ValueError)):
            raise ValueError(
                f"{solver} rejected the solver configuration: {error}"
            ) from error
        diagnostics = replace(diagnostics, failure_reason=f"{solver} failed: {error}")
        return DecompositionResult(mask, None, diagnostics)

    status = problem.status
    objective = None if problem.value is None else objective_scale * problem.value
    centered_fit = centered_seasonal.value
    fitted = None if centered_fit is None else centered_fit + offset
    residual_values = residual.value
    finite_objective = objective is not None and np.isfinite(objective)
    finite_components = all(
        values is not None and np.isfinite(values).all()
        for values in (fitted, residual_values)
    )
    diagnostics = replace(
        diagnostics,
        solver_status=status,
        objective_value=float(objective) if finite_objective else None,
    )
    linking_tolerance = _FEASIBILITY_TOLERANCE * max(1, np.max(np.abs(observed)))
    if finite_components:
        linking_error = float(np.max(np.abs(residual_values + fitted[mask] - observed)))
        diagnostics = replace(diagnostics, constraint_errors={"linking": linking_error})

    reason = None
    if status not in (cp.OPTIMAL, cp.OPTIMAL_INACCURATE):
        reason = f"{solver} returned {status}."
    elif not (finite_components and finite_objective):
        reason = "The solver did not return finite components and objective."
    elif not np.isfinite(linking_error) or linking_error > linking_tolerance:
        reason = f"Feasibility check failed (linking error {linking_error:.3g})."
    if reason is not None:
        return DecompositionResult(
            mask, None, replace(diagnostics, failure_reason=reason)
        )

    residual = np.zeros(mask.size)
    if not centered_observations.any():
        # The observed constant is the unique zero-cost fit.
        seasonal = np.full(mask.size, offset)
        objective = 0.0
    else:
        try:
            centered_seasonal, amplitudes = _recover_harmonics(
                basis, mask, centered_fit, coefficients.value, harmonic_numbers
            )
        except np.linalg.LinAlgError as error:
            reason = f"Fourier component recovery failed: {error}"
            return DecompositionResult(
                mask, None, replace(diagnostics, failure_reason=reason)
            )
        if (
            np.max(np.abs(centered_seasonal[mask] - centered_fit[mask]))
            > linking_tolerance
        ):
            reason = "Fourier component recovery changed the admitted fitted values."
            return DecompositionResult(
                mask, None, replace(diagnostics, failure_reason=reason)
            )
        seasonal = centered_seasonal + offset
        residual[mask] = observed - seasonal[mask]
        weighted_amplitudes = weights * amplitudes
        objective = w0 * _pinball_sum(residual[mask], tau) + 0.5 * (
            weighted_amplitudes @ weighted_amplitudes
        )
    if not (
        np.isfinite(seasonal).all()
        and np.isfinite(residual).all()
        and np.isfinite(objective)
    ):
        reason = "Fourier component recovery did not produce a finite fit."
        return DecompositionResult(
            mask, None, replace(diagnostics, failure_reason=reason)
        )
    diagnostics = replace(diagnostics, objective_value=float(objective))
    # Pinball loss is 1-Lipschitz. Allow the admitted linking tolerance per
    # observation before rejecting a fit worse than the feasible constant.
    constant_loss = _pinball_sum(centered_observations, tau)
    tolerance = centered_observations.size * linking_tolerance
    if objective > w0 * (constant_loss + tolerance):
        reason = "The fitted objective exceeds the feasible constant objective."
        return DecompositionResult(
            mask, None, replace(diagnostics, failure_reason=reason)
        )
    if diagnostics.solver_status == cp.OPTIMAL_INACCURATE:
        diagnostics = replace(
            diagnostics,
            warnings=(f"{diagnostics.solver} returned optimal_inaccurate.",),
        )
    components = {"residual": residual, "seasonal": seasonal}
    return DecompositionResult(mask, components, diagnostics)


def _seasonal_component(result, fit_name, fit_warnings=None):
    if not result.has_fit:
        raise ValueError(
            f"The {fit_name} fit returned no result. {result.diagnostics.failure_reason}"
        )
    if fit_warnings is not None:
        fit_warnings.extend(
            f"{fit_name}: {warning}" for warning in result.diagnostics.warnings
        )
    return result.components["seasonal"]
