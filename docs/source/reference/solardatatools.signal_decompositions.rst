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
