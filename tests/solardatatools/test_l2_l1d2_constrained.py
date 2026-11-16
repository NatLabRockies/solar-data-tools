import pickle
from dataclasses import replace

import cvxpy as cp
import numpy as np
import pytest

from solardatatools import DataHandler
from solardatatools._l2_l1d2_constrained import _L2L1D2ConstrainedWorkspace
from solardatatools.algorithms.clipping import ClippingDetection
from solardatatools.signal_decompositions import (
    DecompositionResult,
    SolveDiagnostics,
    l2_l1d2_constrained,
)


@pytest.mark.parametrize("solver", ["CLARABEL", "OSQP"])
def test_filter_weights_match_analytic_three_point_solution(solver):
    options = {"eps_abs": 1e-8, "eps_rel": 1e-8} if solver == "OSQP" else {}
    for w0, w1 in [(1, 0.1), (2, 0.1), (2, 0.2), (1, 0), (0, 1)]:
        result = l2_l1d2_constrained(
            [0, 0.9, 1], w0=w0, w1=w1, solver=solver, solver_options=options
        )
        assert result.has_fit, result.diagnostics
        # With endpoints fixed, the interior minimizes w0*(0.9-z)**2
        # + 2*w1*abs(z-0.5).
        middle = max(0.5, 0.9 - w1 / w0) if w0 else 0.5
        np.testing.assert_allclose(
            result.components["trend"], [0, middle, 1], atol=1e-6
        )
        objective = w0 * (0.9 - middle) ** 2 + 2 * w1 * abs(middle - 0.5)
        assert result.diagnostics.objective_value == pytest.approx(objective, abs=1e-6)
        assert result.diagnostics.solver_options == options


def test_filter_residual_and_anchors_respect_missing_and_held_out_values():
    signal = np.array([0, 0.8, np.nan, 0.6, 1])
    selection = np.array([False, True, True, False, False])
    result = l2_l1d2_constrained(signal, use_ixs=selection)
    assert result.has_fit, result.diagnostics
    trend = result.components["trend"]
    residual = result.components["residual"]
    np.testing.assert_array_equal(
        result.observation_mask, [False, True, False, False, False]
    )
    np.testing.assert_allclose(trend[[0, -1]], [0, 1], atol=1e-6)
    np.testing.assert_array_equal(residual[~result.observation_mask], 0)
    assert trend[1] + residual[1] == pytest.approx(signal[1])
    with pytest.raises(ValueError, match="interior"):
        l2_l1d2_constrained(signal, w1=0)


@pytest.mark.parametrize("w0", [0, 2])
def test_off_set_residual_changes_only_the_full_grid_cost(w0):
    workspace = _L2L1D2ConstrainedWorkspace(4)
    result = workspace.solve([0, np.nan, 0.8, 1], w0=w0)
    assert result.has_fit, result.diagnostics
    assert result.components["residual"][1] == 0

    # A nonzero residual outside the known set is feasible.
    variables = {
        variable.name(): variable for variable in workspace._problem.variables()
    }
    variables["trend"].value = np.linspace(0, 1, 4)
    residual = np.array([0, 0, 0.8 - 2 / 3, 0])
    variables["residual"].value = residual
    initial_cost = workspace._problem.objective.value
    residual[1] = 3
    variables["residual"].value = residual
    for constraint in workspace._problem.constraints:
        np.testing.assert_allclose(constraint.violation(), 0, atol=1e-12)
    assert workspace._problem.objective.value - initial_cost == pytest.approx(9 * w0)


def test_no_observations_returns_without_solving(monkeypatch):
    def unexpected_solve(*args, **kwargs):
        pytest.fail("The solver ran with no admitted observations.")

    monkeypatch.setattr(cp.Problem, "solve", unexpected_solve)
    result = l2_l1d2_constrained([np.nan, np.inf, 1], use_ixs=[True, True, False])
    assert not result.has_fit
    assert result.components is None
    assert result.diagnostics.solver_status is None
    assert result.diagnostics.failure_reason == "No admitted observations."


def test_configuration_errors_remain_errors():
    with pytest.raises(ValueError, match="unavailable"):
        l2_l1d2_constrained([0, 0.5, 1], solver="unknown-solver")
    with pytest.raises(ValueError, match="solver_path"):
        l2_l1d2_constrained([0, 0.5, 1], solver_options={"solver_path": ["SCS"]})
    with pytest.raises(TypeError, match="unrecognized solver setting"):
        l2_l1d2_constrained([0, 0.5, 1], solver_options={"bad_setting": True})


def test_reused_workspace_does_not_inherit_iteration_limit():
    workspace = _L2L1D2ConstrainedWorkspace(3)
    signal = [0, 0.9, 1]
    with pytest.warns(UserWarning, match="Solution may be inaccurate"):
        limited = workspace.solve(signal, solver_options={"max_iter": 0})
    assert not limited.has_fit
    recovered = workspace.solve(signal)
    assert recovered.has_fit, recovered.diagnostics
    assert recovered.diagnostics.solver_options == {}


@pytest.mark.parametrize(
    ("status", "values", "has_fit", "reason"),
    [
        (cp.OPTIMAL_INACCURATE, [0, 0.5, 1], True, None),
        (cp.USER_LIMIT, [0, 0.5, 1], False, "user_limit"),
        (cp.OPTIMAL, [0.001, 0.5, 1], False, "endpoint error"),
        (cp.OPTIMAL, [0, 0.6, 1], False, "linking error"),
    ],
)
def test_filter_accepts_only_usable_solver_results(
    monkeypatch, status, values, has_fit, reason
):
    def solve(problem, *, solver, verbose, max_iter):
        assert (solver, verbose, max_iter) == ("CLARABEL", True, 100)
        variables = {variable.name(): variable for variable in problem.variables()}
        variables["trend"].value = np.array(values)
        variables["residual"].value = np.zeros(3)
        monkeypatch.setattr(cp.Problem, "status", property(lambda self: status))
        monkeypatch.setattr(cp.Problem, "value", property(lambda self: 0.0))

    monkeypatch.setattr(cp.Problem, "solve", solve)
    result = l2_l1d2_constrained(
        [0, 0.5, 1], verbose=True, solver_options={"max_iter": 100}
    )
    assert result.has_fit is has_fit
    assert isinstance(result, DecompositionResult)
    assert isinstance(result.diagnostics, SolveDiagnostics)
    assert result.diagnostics.solver_status == status
    assert result.diagnostics.constraint_errors == pytest.approx(
        {
            "linking": max(abs(np.array(values) - [0, 0.5, 1])),
            "endpoint": max(abs(values[0]), abs(values[-1] - 1)),
        }
    )
    if has_fit:
        assert "optimal_inaccurate" in result.diagnostics.warnings[0]
    else:
        assert result.components is None
        assert reason in result.diagnostics.failure_reason


def test_clipping_reuse_matches_a_fresh_fit_and_clears_failed_repeat(monkeypatch):
    first_data = np.tile([0, 5, 10, 10, 10, 5, 0], (12, 1)).T.astype(float)
    second_data = first_data * np.linspace(0.3, 1.0, 12)
    detector = ClippingDetection()
    detector.check_clipping(first_data, solver_options={"max_iter": 100})
    first = detector.fit_result
    first_trend = first.components["trend"].copy()
    detector.find_clipped_times()
    assert detector.clipping_mask is not None
    workspace = detector._filter_workspace
    assert workspace._problem.is_dcp(dpp=True)

    detector.check_clipping(second_data, weight=2)
    fresh = ClippingDetection()
    fresh.check_clipping(second_data, weight=2)
    assert detector._filter_workspace is workspace
    assert detector.fit_result.diagnostics.solver_options == {}
    np.testing.assert_allclose(detector.y_hat, fresh.y_hat, atol=1e-8)
    np.testing.assert_array_equal(detector.clipped_days, fresh.clipped_days)
    np.testing.assert_array_equal(first.components["trend"], first_trend)
    assert not np.allclose(detector.y_hat, first_trend)
    assert detector.clipping_mask is None

    restored = pickle.loads(pickle.dumps(detector))
    assert restored._filter_workspace is None
    np.testing.assert_array_equal(restored.y_hat, detector.y_hat)
    restored.check_clipping(second_data, weight=2)
    np.testing.assert_allclose(restored.y_hat, fresh.y_hat, atol=1e-8)

    def solver_failure(*args, **kwargs):
        raise cp.error.SolverError("test clipping solver failure")

    monkeypatch.setattr(cp.Problem, "solve", solver_failure)
    detector.check_clipping(first_data)
    detector.find_clipped_times()
    assert not detector.fit_result.has_fit
    assert (
        "test clipping solver failure" in detector.fit_result.diagnostics.failure_reason
    )
    assert detector.y_hat is detector.clipped_days is detector.clipping_mask is None
    assert detector.inverter_clipping is detector.point_mass_locations is None
    np.testing.assert_array_equal(first.components["trend"], first_trend)


def test_pipeline_reports_clipping_fit_failure_and_warning(monkeypatch):
    samples = np.tile(
        np.r_[np.zeros(24), np.sin(np.linspace(0, np.pi, 48)), np.zeros(24)], (21, 1)
    ).T
    original = ClippingDetection.get_l2_l1d2
    mode = "failed"

    def filter_with_diagnostics(detector, *args, **kwargs):
        original(detector, *args, **kwargs)
        result = detector.fit_result
        if mode == "failed":
            detector.fit_result = replace(
                result,
                components=None,
                diagnostics=replace(
                    result.diagnostics, failure_reason="test clipping failure"
                ),
            )
            detector.y_hat = None
        else:
            detector.fit_result = replace(
                result,
                diagnostics=replace(
                    result.diagnostics, warnings=("test inaccurate fit",)
                ),
            )

    monkeypatch.setattr(ClippingDetection, "get_l2_l1d2", filter_with_diagnostics)
    handler = DataHandler(raw_data_matrix=samples)
    handler.run_pipeline(verbose=False)
    report = handler.report(verbose=False, return_values=True)
    outcome = report["stage outcomes"]["clipping"]
    assert outcome["status"] == "failed"
    assert outcome["reason"] == "test clipping failure"
    assert report["inverter clipping"] is None
    assert handler.boolean_masks.clipped_times is None

    mode = "warning"
    handler.run_pipeline(verbose=False)
    outcome = handler.report(verbose=False, return_values=True)["stage outcomes"][
        "clipping"
    ]
    assert outcome["status"] == "success"
    assert outcome["warning"] == "test inaccurate fit"
