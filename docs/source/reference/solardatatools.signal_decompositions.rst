Signal Decompositions
=====================

.. automodule:: solardatatools.signal_decompositions
   :members:
   :undoc-members:
   :show-inheritance:
   :imported-members:

Migrating the constrained filter
--------------------------------

``l2_l1d2_constrained`` keeps its import path and returns a
``DecompositionResult``. Use keyword arguments for weights and solver
settings. Replace the fitted curve from the old tuple with
``result.components["trend"]`` and the objective from the old problem with
``result.diagnostics.objective_value``. The filter now has one formulation
for every solver. To reproduce the previous MOSEK objective weights, pass
``w0=1, w1=10`` explicitly.

.. code-block:: python

   from solardatatools.signal_decompositions import l2_l1d2_constrained

   result = l2_l1d2_constrained(
       signal, w0=1, w1=5, solver="CLARABEL", solver_options={"max_iter": 100}
   )
   if result.has_fit:
       trend = result.components["trend"]
       residual = result.components["residual"]
       admitted = result.observation_mask
       objective = result.diagnostics.objective_value
   else:
       print(result.diagnostics.failure_reason)

Both components cover the full grid. The consistency constraint applies at the
admitted observations. Outside that set, a positive residual weight gives a
zero residual at the minimum. With zero weight, the returned result chooses
zero for those entries. ``diagnostics.solver_status`` retains the backend status.
An ``optimal_inaccurate`` result can carry a usable fit and a warning when its
objective, component values, and feasibility checks pass. For this filter,
``diagnostics.constraint_errors["linking"]`` measures the maximum linking error
on the admitted observations and must be within
``1e-6 * max(1, max(abs(signal[admitted])))``.
``diagnostics.constraint_errors["endpoint"]`` measures the maximum endpoint
error and must be within ``1e-6`` of the anchors.

Direct calls to ``DataHandler.clipping_check`` and
``ClippingDetection.check_clipping`` accept the same solver options. Inspect
``ClippingDetection.fit_result`` for the latest fit diagnostics. The pipeline
continues to select the clipping solver through ``solver_convex``.

Compilation and repeated fits
----------------------------

For this filter's default quadratic CLARABEL path, SDT adds
``ignore_dpp=True`` to the effective CVXPY solve options. This avoids a large
dense parameter-coefficient intermediate observed in CVXPY 1.7.5, 1.8.1,
1.8.2, and 1.9.3. The model, weights, constraints, and CLARABEL numerical
defaults are unchanged. Inspect ``result.diagnostics.solver_options`` for
the effective options, including this default.

The clipping workspace still reuses its expressions for a fixed observation
mask and returns independent result snapshots. CVXPY recompiles on each solve
instead of using its disciplined parametrized programming (DPP) cache.
Backend reuse depends on the solver and settings. This trades compilation
time on repeated fits for avoiding the observed dense intermediate; it does
not guarantee a particular memory bound or speedup.

Explicit caller choices take precedence. Pass
``solver_options={"ignore_dpp": False}`` to retain DPP caching, or
``solver_options={"enforce_dpp": True}`` to require it. Explicit
``use_quad_obj=False`` retains conic canonicalization without adding the
workaround. CVXPY continues to reject conflicting explicit
``ignore_dpp=True`` and ``enforce_dpp=True`` flags. Other solvers' compilation
defaults are unchanged. Direct filter and clipping calls use the same rules.
