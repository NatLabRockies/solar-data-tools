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

Migrating the annual quantile smoother
--------------------------------------

``tl1_l2d2p365`` keeps its import path and fits one annual Fourier model to
daily samples. The basis has a 365.2425-day period, an unpenalized constant,
and three cosine/sine pairs. The objective adds the sum of pinball losses
weighted by ``w0`` to a quadratic penalty on the Fourier coefficients.
For harmonic :math:`k`, the penalty on each coefficient has weight
:math:`2\pi^2(10^{-3})^2 k^2/365.2425`.
Use a finite positive ``w0`` and a ``tau`` strictly between zero and one.

Remove ``return_all`` from calls. Read ``result.components["seasonal"]``
for the fitted curve. Use ``result.diagnostics.objective_value`` in place of
the second item from the old MOSEK tuple or the ``objective_value`` attribute
of the old OSD problem.

Remove ``yearly_periodic`` from calls. The smoother always uses annual Fourier
seasonality. Calls requesting ``yearly_periodic=False`` raise ``TypeError``.
The previous MOSEK call used a second-difference smoother and may produce
different fitted values from the annual model.

.. code-block:: python

   from solardatatools.signal_decompositions import tl1_l2d2p365

   result = tl1_l2d2p365(
       daily_signal, use_ixs=selected_days, tau=0.8, solver="CLARABEL",
       solver_options={"max_iter": 100},
   )
   if result.has_fit:
       seasonal = result.components["seasonal"]
       residual = result.components["residual"]
       admitted = result.observation_mask
   else:
       print(result.diagnostics.failure_reason)

Both components cover every day on the input grid, including missing and
held-out days. The admitted mask selects the finite entries that enter the
fit. Masked entries in a NumPy masked array are excluded from both the
observations and the optional Boolean selector. The residual is zero
outside the admitted mask, and the seasonal component
supplies the prediction there. Diagnostics retain the backend's status and
maximum absolute linking error on admitted days. A usable fit needs finite
values and a linking error within
``1e-6 * max(1, max(abs(daily_signal[admitted])))``. An accepted
``optimal_inaccurate`` result retains that status and carries a warning.
The solver translates the intercept by the admitted sample quantile when
that translation gives finite values. It uses zero otherwise. It divides
the objective by :math:`\min(1,\sqrt{w_0})`. The reported
seasonal values and objective are in the original units. With identical
admitted values, an accepted solve returns the exact constant fit with zero
objective. When the admitted fitted values underdetermine the Fourier
coefficients, recovery selects the coefficients with the smallest harmonic
penalty at those values. The linking error records the backend values before
recovery. The returned objective is evaluated on the recovered fit. A result
is rejected if that objective exceeds the objective of the constant fit at
the translation value by more than the residual weight times the number of
admitted days times the linking tolerance.

CLARABEL is the default. Supply the ``solver`` and ``verbose`` arguments
by keyword. Pass ``solver="MOSEK"`` or another suitable installed CVXPY
solver to select the backend, and pass its settings with ``solver_options``.
The same settings can be passed to
``make_density_scores``, ``ClearDayDetection.find_clear_days``, and
``SunriseSunset.run_optimizer`` or ``calculate_times``. In
``DataHandler.run_pipeline``, ``solver_convex`` selects the solver for
daylight, density, and clear-day fits, subject to the existing MOSEK preflight.
The corresponding stage outcomes include warnings from accepted inaccurate
fits.
If either provisional daylight fit has no admitted observations, the pipeline
reports that daylight estimate as unavailable and skips matrix filling. The
handler keeps the raw observations, and the partial report retains the
timezone advisory.
