import cvxpy as cp
import numpy as np
import pytest

from solardatatools import DataHandler
from solardatatools.algorithms import sunrise_sunset_estimation
from solardatatools.signal_decompositions import DecompositionResult, tl1_l2d2p365


def test_quantile_objective_has_exact_coefficient_weights_and_pinball_scaling(
    monkeypatch,
):
    original_solve = cp.Problem.solve
    tau, w0 = 0.2, 2.5

    def solve(problem, **options):
        variables = {variable.name(): variable for variable in problem.variables()}
        coefficients = variables["fourier_coefficients"]
        residual = variables["observed_residual"]
        residual.value = [-2.0, 1.0]
        expected_residual_cost = w0 * (2 * (1 - tau) + tau)
        weights = 2 * np.pi**2 * 1e-6 / 365.2425 * np.array([0, 1, 1, 4, 4, 9, 9])
        for direction, weight in zip(10 * np.eye(7), weights):
            coefficients.value = direction
            assert problem.objective.value == pytest.approx(
                expected_residual_cost + 100 * weight, abs=1e-13, rel=0
            )
        assert options == {"solver": "CLARABEL", "verbose": False}
        return original_solve(problem, **options)

    monkeypatch.setattr(cp.Problem, "solve", solve)
    result = tl1_l2d2p365([2.0, 2.0], tau=tau, w0=w0)
    assert result.has_fit


@pytest.mark.parametrize(
    ("solver", "options"),
    [("CLARABEL", {}), ("SCS", {"eps_abs": 1e-7, "eps_rel": 1e-7})],
)
def test_annual_quantile_matches_independent_coefficient_problem(solver, options):
    days = np.arange(740)
    phase = 2 * np.pi * days / 365.2425
    basis = np.column_stack(
        [np.ones(days.size)]
        + [wave(k * phase) for k in (1, 2, 3) for wave in (np.cos, np.sin)]
    )
    signal = basis @ [7.0, 2.0, -1.5, 1.0, 0.5, -0.75, 0.25]
    signal[100:108] -= 1.5
    signal[300:312] += 0.5
    signal[[3, 350]] = [np.nan, np.inf]
    selected = np.ones(days.size, dtype=bool)
    selected[[0, 20, 160, -1]] = False
    signal[~selected] = 100.0
    original = signal.copy()
    known = selected & np.isfinite(signal)
    tau, w0 = 0.7, 2.5

    theta = cp.Variable(7)
    error = signal[known] - basis[known] @ theta
    loss = cp.sum(tau * cp.pos(error) + (1 - tau) * cp.pos(-error))
    weights = (2 * np.pi**2 * 1e-6 / 365.2425) * np.array([0, 1, 1, 4, 4, 9, 9])
    reference = cp.Problem(
        cp.Minimize(w0 * loss + cp.sum(cp.multiply(weights, cp.square(theta))))
    )
    reference.solve(solver="CLARABEL")
    assert reference.status == cp.OPTIMAL

    result = tl1_l2d2p365(
        signal, selected, tau, w0, solver=solver, solver_options=options
    )
    assert isinstance(result, DecompositionResult)
    assert result.has_fit, result.diagnostics
    assert result.diagnostics.solver == solver
    assert result.diagnostics.solver_options == options
    assert result.diagnostics.constraint_errors["linking"] < 1e-6
    assert result.diagnostics.objective_value == pytest.approx(
        reference.value, abs=2e-5
    )
    seasonal, residual = result.components["seasonal"], result.components["residual"]
    np.testing.assert_allclose(seasonal, basis @ theta.value, atol=1e-5, rtol=0)
    np.testing.assert_array_equal(result.observation_mask, known)
    np.testing.assert_array_equal(residual[~known], 0)
    np.testing.assert_allclose(seasonal[known] + residual[known], signal[known])
    np.testing.assert_array_equal(signal, original)
    assert np.max(np.abs(seasonal[365:] - seasonal[:-365])) > 1e-3


def test_quantile_passes_through_mosek_without_a_model_switch(monkeypatch):
    original_solve = cp.Problem.solve
    monkeypatch.setattr(cp, "installed_solvers", lambda: ["CLARABEL", "MOSEK"])
    options = {"mosek_params": {"MSK_IPAR_NUM_THREADS": 1}}

    def solve(problem, *, solver, verbose, **kwargs):
        assert solver == "MOSEK"
        assert not verbose
        assert kwargs == options
        return original_solve(problem, solver="CLARABEL")

    monkeypatch.setattr(cp.Problem, "solve", solve)
    result = tl1_l2d2p365([4.0], solver="MOSEK", solver_options=options)
    assert result.has_fit
    np.testing.assert_allclose(result.components["seasonal"], [4.0])
    assert result.diagnostics.solver == "MOSEK"
    assert result.diagnostics.solver_options == options


def test_quantile_excludes_hidden_values_from_both_masks():
    fits = []
    for hidden, selected in [(0, True), (12, False)]:
        signal = np.ma.array(
            [6, hidden, 6, hidden, 100], mask=[False, True, False, True, False]
        )
        use_ixs = np.ma.array(
            [True, True, True, True, selected], mask=[False, False, False, False, True]
        )
        result = tl1_l2d2p365(signal, use_ixs=use_ixs)
        assert result.has_fit
        np.testing.assert_array_equal(
            result.observation_mask, [True, False, True, False, False]
        )
        fits.append(result.components["seasonal"])
    np.testing.assert_allclose(fits, np.full((2, 5), 6.0), atol=1e-7, rtol=0)


def test_quantile_sparse_constant_fit_with_default_scs():
    known = np.zeros(400, dtype=bool)
    known[[0, 1]] = True
    signal = np.full(known.size, 10.0)
    result = tl1_l2d2p365(signal, use_ixs=known, solver="SCS")
    assert result.has_fit, result.diagnostics
    np.testing.assert_allclose(result.components["seasonal"], 10.0, atol=1e-6, rtol=0)
    assert result.diagnostics.objective_value < 1e-8


def test_quantile_recovers_minimum_penalty_curve_from_two_admitted_values():
    signal = np.full(400, np.nan)
    signal[[0, 30]] = [6.0, 7.0]
    phase = 2 * np.pi * np.arange(signal.size) / 365.2425
    waves = np.column_stack(
        [wave(k * phase) for k in (1, 2, 3) for wave in (np.cos, np.sin)]
    )
    contrast = waves[30] - waves[0]
    squared_harmonics = np.array([1, 1, 4, 4, 9, 9])
    # With y[30] - y[0] = 1, the minimum weighted norm has this closed form.
    amplitudes = (contrast / squared_harmonics) / (
        (contrast**2 / squared_harmonics).sum()
    )
    expected = 6.0 + (waves - waves[0]) @ amplitudes
    result = tl1_l2d2p365(signal, tau=0.85, solver="SCS")
    assert result.has_fit, result.diagnostics
    np.testing.assert_allclose(
        result.components["seasonal"], expected, atol=1e-4, rtol=0
    )


def test_quantile_small_weight_fit_reports_original_objective():
    signal = np.full(365, 20.0)
    signal[::3] = 5.0
    w0 = 1e-12
    result = tl1_l2d2p365(signal, tau=0.5, w0=w0)
    assert result.has_fit, result.diagnostics
    np.testing.assert_allclose(result.components["seasonal"], 20.0, atol=1e-6, rtol=0)
    constant_objective = 0.5 * w0 * np.abs(signal - 20).sum()
    assert result.diagnostics.objective_value == pytest.approx(
        constant_objective, rel=1e-5, abs=0
    )


def test_quantile_rejects_a_feasible_fit_worse_than_the_constant_fit(monkeypatch):
    signal = np.full(365, 20.0)
    signal[::3] = 5.0

    def solve(problem, **options):
        variables = {variable.name(): variable for variable in problem.variables()}
        # A feasible fit at -1758, with the intercept translated by 20.
        variables["fourier_coefficients"].value = [-1778.0, 0, 0, 0, 0, 0, 0]
        variables["observed_residual"].value = signal + 1758.0
        problem._status = cp.OPTIMAL_INACCURATE
        problem._value = problem.objective.value

    monkeypatch.setattr(cp.Problem, "solve", solve)
    result = tl1_l2d2p365(signal, tau=0.5, w0=1e-14, solver="SCS")
    assert not result.has_fit
    assert result.diagnostics.solver_status == cp.OPTIMAL_INACCURATE
    assert result.diagnostics.constraint_errors["linking"] == 0
    assert (
        "exceeds the feasible constant objective" in result.diagnostics.failure_reason
    )
    assert result.diagnostics.objective_value == pytest.approx(
        0.5e-14 * np.abs(signal + 1758.0).sum(), rel=1e-12, abs=0
    )


@pytest.mark.parametrize(
    "parameters",
    [
        {"w0": 0},
        {"w0": np.inf},
        {"tau": 0},
        {"tau": 1},
        {"tau": np.nan},
    ],
)
def test_quantile_rejects_values_outside_the_public_domain(parameters):
    with pytest.raises(ValueError, match=next(iter(parameters))):
        tl1_l2d2p365([1.0], **parameters)


@pytest.mark.parametrize(
    "retired_option", [{"return_all": True}, {"yearly_periodic": False}]
)
def test_quantile_rejects_retired_options(retired_option):
    with pytest.raises(TypeError, match=next(iter(retired_option))):
        tl1_l2d2p365([1.0], **retired_option)


def test_quantile_returns_failure_for_an_empty_mask_or_solver_failure(monkeypatch):
    def fail_solve(*args, **kwargs):
        raise cp.error.SolverError("test failure")

    monkeypatch.setattr(cp.Problem, "solve", fail_solve)
    empty = tl1_l2d2p365([np.nan, np.inf])
    assert not empty.has_fit
    assert empty.diagnostics.failure_reason == "No admitted observations."
    np.testing.assert_array_equal(empty.observation_mask, [False, False])
    failed = tl1_l2d2p365([2.0, np.nan])
    assert not failed.has_fit
    assert failed.diagnostics.solver == "CLARABEL"
    assert failed.diagnostics.failure_reason == "CLARABEL failed: test failure"
    np.testing.assert_array_equal(failed.observation_mask, [True, False])


def test_sunrise_sunset_optimizer_uses_full_grid_holdout_predictions(monkeypatch):
    hours = np.arange(96)[:, None] / 4
    days = np.arange(365)
    daylight_halfwidth = 6 + np.sin(2 * np.pi * days / 365.2425)
    data = np.maximum(0, 1 - ((hours - 12) / daylight_halfwidth) ** 2)
    holdout_calls = []
    options = {"max_iter": 100}

    def fit(signal, *, use_ixs, **kwargs):
        result = tl1_l2d2p365(signal, use_ixs=use_ixs, **kwargs)
        assert result.has_fit
        assert result.diagnostics.solver_options == options
        np.testing.assert_array_equal(result.observation_mask, use_ixs)
        withheld = np.isfinite(signal) & ~use_ixs
        assert withheld.any()
        assert np.isfinite(result.components["seasonal"][withheld]).all()
        np.testing.assert_array_equal(result.components["residual"][withheld], 0)
        holdout_calls.append(withheld.sum())
        return result

    monkeypatch.setattr(sunrise_sunset_estimation, "tl1_l2d2p365", fit)
    analysis = sunrise_sunset_estimation.SunriseSunset()
    analysis.run_optimizer(data, random_seed=42, search_pts=3, solver_options=options)
    assert len(holdout_calls) == 6
    assert np.isfinite(analysis.sunrise_estimates).all()
    assert np.isfinite(analysis.sunset_estimates).all()
    assert analysis.sunup_mask_estimated.shape == data.shape


def test_sunrise_sunset_optimizer_fits_a_single_day_without_a_holdout():
    hours = np.arange(96)[:, None] / 4
    data = np.maximum(0, 1 - ((hours - 12) / 6) ** 2)
    data[0, 0] = 0.001
    analysis = sunrise_sunset_estimation.SunriseSunset()
    analysis.run_optimizer(data, search_pts=3)
    assert analysis.threshold > 1e-5
    np.testing.assert_allclose(
        analysis.sunrise_estimates, analysis.sunrise_measurements
    )
    np.testing.assert_allclose(analysis.sunset_estimates, analysis.sunset_measurements)


@pytest.mark.parametrize("method", ["calculate_times", "run_optimizer"])
def test_failed_sunrise_sunset_rerun_invalidates_prior_fit(method):
    hours = np.arange(96)[:, None] / 4
    day = np.maximum(0, 1 - ((hours - 12) / 6) ** 2)
    analysis = sunrise_sunset_estimation.SunriseSunset()
    kwargs = {"threshold": 0.01} if method == "calculate_times" else {"search_pts": 3}
    fit = getattr(analysis, method)
    fit(day, **kwargs)
    assert analysis.sunrise_estimates.size == 1
    wrapped_days = np.tile(np.roll(day, 48, axis=0), (1, 3))
    with pytest.raises(ValueError, match="No admitted observations"):
        fit(wrapped_days, **kwargs)
    assert all(
        getattr(analysis, attribute) is None
        for attribute in (
            "sunrise_estimates",
            "sunset_estimates",
            "sunrise_measurements",
            "sunset_measurements",
            "sunup_mask_measured",
            "sunup_mask_estimated",
            "total_rmse",
        )
    )
    assert analysis.fit_warnings == ()


def test_pipeline_reports_accepted_quantile_accuracy_warnings(monkeypatch):
    original_solve = cp.Problem.solve

    def solve(problem, *args, **kwargs):
        value = original_solve(problem, *args, **kwargs)
        if problem.status == cp.OPTIMAL and any(
            variable.name() == "fourier_coefficients"
            for variable in problem.variables()
        ):
            problem._status = cp.OPTIMAL_INACCURATE
        return value

    monkeypatch.setattr(cp.Problem, "solve", solve)
    hours = np.arange(24)[:, None]
    days = np.arange(36)
    widths = 6 + np.sin(days / 5)
    data = np.maximum(0, 1 - ((hours - 12) / widths) ** 2)
    handler = DataHandler(raw_data_matrix=data)
    handler.run_pipeline(verbose=False)
    stages = handler.report(verbose=False, return_values=True)["stage outcomes"]
    for stage in (
        "provisional daylight",
        "daily scores",
        "clear days",
        "final daylight",
    ):
        assert stages[stage]["status"] == "success", stages[stage]
        assert "optimal_inaccurate" in stages[stage]["warning"]
