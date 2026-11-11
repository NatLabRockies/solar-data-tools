# Public API review: proposed issues and corrections

- Date: 2026-10-21
- Review register: public tool
- Reviewed revision: `e1cd896e6bd86f2fb12bd4c588af4f3a64103e92` (`main` at review time)
- Branch when this plan was recorded: `cleanup`, at the same revision
- Last revised: 2026-11-11, DH-01 completed and DH-02 opened for specification
- Status: working document. Implementation requires issue-specific authorization.

## Purpose and authorization

This working document organizes the repository review by code location. Revise
findings and proposals as decisions and evidence develop, retaining issue IDs and
recording superseded dispositions where needed. It records reported defects,
their evidence, proposed corrections, and proposed verification. It is not an
approved implementation specification.

Writing this plan does **not** approve the proposed issues for edits or changes.
Work will proceed methodically, with substantial human input and issue-specific
direction. Agreement that an issue exists does not by itself authorize a
particular correction, experiment, or change to the public contract.

For each selected issue or explicitly bounded group:

1. Review the current source and the evidence below. Confirm, revise, reject, or
   defer the finding with the owner; do not assume the original diagnosis or
   proposed remedy is final.
2. Resolve the intended mathematical/API contract and compatibility implications.
   Identify the smallest correction and a focused verification procedure.
3. Obtain issue-specific authorization for the work to be performed. Record the
   approved scope, including any tests, numerical experiments, or dependency work.
4. Carry out only that scope, then present the resulting diff and verification
   evidence for human review. Preserve the distinction between an observed
   result, an inference, and an unresolved limitation.
5. Record the disposition and remaining questions before moving to another item.
   Approval for one item does not approve the rest of this document. Commits,
   publication, deployment, and live cloud operations require their own applicable
   authorization.

Section 9 records issue-specific authorizations and completed work. No execution
schedule or release gate has been adopted. P1/P2 labels below preserve provisional
materiality assessments without establishing an implementation order or
superseding owner decisions. Design and performance proposals are identified
separately from demonstrated defects.

### Decision-pass status and remaining specifications (2026-11-11)

The owner-question pass has resolved the major model, reporting, retirement,
and continuation choices identified in the initial review. The November 11
follow-up assessment exposes interactions and public callers that those
decisions did not fully specify. The earlier statement that no additional
policy questions remained was premature. Preserve the settled policies while
resolving these narrower questions. Independent daylight-segment classification
and the daily-energy fitting/accounting policies are now agreed, as are
retraining for saved-model migration, default NaN padding for partial days
in `make_2d`, and fixed-round fitting for explicit XGBoost classification.
These decisions are recorded at their owning locations. The planning assessment
itself closed no issues. Section 9 records subsequent implementation and review.
The remaining work needs concrete specifications and issue-specific review.
Empirical choices also require validation.

| Code location / issue | Remaining specification or decision | Public verification path |
| --- | --- | --- |
| `data_handler.py`, dilation — DH-02/DH-07 | Keep observed values, daylight bounds, and masks in the same clock frame; define usable-result dependency gates and preserve partial reports. | Nonzero clock correction through PV quantiles/Bundt and mask export; failed optional stages through handler/Dask reports. |
| Time transformations — DH-04, TIME-01/TIME-02 | Specify the midnight-production warning and snapping ties; propagate exact cadence and implement the agreed NaN padding of partial days with explicit trimming retained. | Fractional-second and partial-day inputs through matrices, extra columns, mask export, and profiler coordinates. |
| Native decomposition — SD-01/SD-03/SD-05/SD-06 | Specify result fields, Fourier/penalty details, numerical tolerances, solver-option migration, and recurrence-compatible pooled support. | Pipeline solver pass-through and a recurring shift ending partway through a cycle, including its single fixed-support refit. |
| Capacity/time-shift selection — CAP-02/SHIFT-01 | Validate bounded endpoint extension, numerical constancy, screening, and the recurrence advisory; remove contradictory correction suppression. | Selection and plots share the verified reference, including zero error; accepted short-lived shifts survive correction application. |
| Clear-sky classification — CLEAR-01 | Apply the agreed independent-daylight-segment objective through the public wrapper. | `DataHandler.detect_clear_sky` agrees with exhaustive segment-wise path optimization. |
| LFA accounting — LFA-05 | Carry the agreed observed-only accounting, filled fitting vector, partial-day factors, measured-zero handling, and positivity-based admission with existing quality/outage flags. | Equivalent W/Wh and kW/kWh inputs through the handler to fit masks, totals, and Shapley factors. |
| Native LFA/Dask — LFA-03/LFA-06/LFA-07, DH-07 | Validate the transferred soiling model, distinguish incomplete numerical screening from no soiling, retain accepted snapshots, and remove the Dask duration gate. | Short-record and failed-candidate/refinement cases through direct LFA and per-key Dask reports. |
| Shading — SHADE-01 | Specify integration weights, endpoint conventions, empty target days, and eligibility through annotation aggregation. | Reconstruction and heatmap/polar outputs preserve energy/additivity where defined and exclude unavailable fractions. |
| Profiling — PROF-01/PROF-02/PROF-03 | Specify Fourier details and physical constraints, migrate the threshold solver, and separate daily-envelope from sample-incidence eligibility. | Default study path on the base install; direct and `estimate_all` orientation calls agree on masks without persistent mutation. |
| Native quantiles/saved models — PKG-03, QA-02 | Record independent fitting, rearrangement, interaction bases, coefficient evaluation, signed-time behavior, and the agreed retraining migration for legacy artifacts. | Unsorted levels and evaluation before/after the training origin in both consumers; installed-wheel save/load/predict. |
| Anomaly classifiers — OUT-02 | Apply the agreed configured-round XGBoost fit without early stopping; retain classifier validation data for scoring. | Every supported classifier's public fit path, with base-install and anomaly-extra coverage. |
| Package/distribution — PKG-01/QA-02/QA-04 | Establish compatible dependency bounds, installed-package behavior, numerical evidence, and migration. | Scoped wheel checks outside the checkout and independent numerical contract verification. |

Resolve the relevant specification before its dependent implementation.
Numerical experiments and source edits still require the issue-specific
direction stated above. Raise any newly exposed modeling or public-contract
choice with the owner; do not reopen a settled policy merely to choose a field
name or an empirical tolerance without supporting evidence.

## Evidence and scope

The review used source inspection, call-path tracing, existing-test inspection,
and algebraic counterexamples. It did not execute tests or numerical solves,
install dependencies, build distributions, measure performance, or operate cloud
infrastructure. Runtime reproductions below are **proposed**, not completed.

The reviewed areas were the core pipeline and time transformations, decomposition
implementations, principal analysis algorithms, profiler and anomaly APIs, Dask
orchestration, packaging, documentation, and relevant tests. Archived notebooks,
live cloud behavior, published distribution contents, numerical robustness across
datasets, and empirical performance were not validated. This is not a complete
security audit or certification of every code path.

The review applied `review-registers` at the public-tool register and used
`cvx-signal-decomposition` for mathematical interpretation. The owner subsequently
specified native CVXPY implementations in this project and retirement of OSD;
section 2 records that direction. This does not introduce a `signaldecomp`
dependency. Specific formulations, conventions, and inferential claims remain
subject to explicit discussion.

The focused decomposition re-review also inspected local dependency source and
metadata (`spcqe` 0.3.0 and `sig-decomp` 0.3.2 in the `sdt-dev` environment), the
CVXPY DPP/solver documentation, and the reweighted-l1 reference cited below.
Dependency observations are version-specific. Reported solve counts follow from
the source; no timing or memory benchmark was performed.

The degradation alignment review checked the supplied RdTools review against
`../rdtools/rdtools/signal_decomposition.py`, especially
`extract_degradation_rate_log` and the public `degradation` return contract, at
RdTools revision `d414a758ea85ae60482a931eda3efc188c3980f3` (clean working tree at
inspection). This establishes the local reference implementation's extraction
convention, not released-package status, empirical equivalence of the two
decomposition models, or uncertainty coverage. No RdTools files were modified.

The follow-up public-API review was checked against the same repository revision.
Its additional findings and refinements are incorporated at the owning code
locations below. Verification consisted of source and existing-test inspection;
the proposed runtime checks remain unexecuted.

Paths below are repository-relative. Line numbers refer to the reviewed revision
and are navigation aids; function names and issue IDs should remain the durable
references as code changes. Cross-references identify shared work, not duplicate
implementation requests.

## 1. Core data handling and time axes

### `solardatatools/data_handler.py`

#### DH-01 — Plotting an extra matrix mutates its stored values [P1]

- **Status:** complete. See the [completion record](#2026-11-11--dh-01-completed).
- **Location:** `DataHandler.plot_heatmap`, approximately lines 1674–1680.
- **Finding:** the extra-matrix branch aliases `self.extra_matrices[matrix]`, then
  applies `mat /= 1000` when plotting watts as kilowatts. Each call changes the
  stored scientific data again. The raw/filled branches already make copies.
- **Proposed correction:** use a copy or an out-of-place unit conversion. Preserve
  the source array regardless of which matrix is plotted.
- **Proposed verification:** plot the same extra matrix repeatedly and assert
  unchanged values and unchanged subsequent analysis. Coordinate with DIL-01.

#### DH-02 — Automatic time-shift retry mixes correction state from different runs [P1]

- **Status, 2026-11-11:** open for specification. The retry removal is settled.
  Confirm the public clock frame of undilated quantile outputs and sub-daily
  masks before implementation.
- **Location:** `run_pipeline`, approximately lines 603–617;
  `auto_fix_time_shifts`, approximately lines 1413–1415. Related:
  `algorithms/time_shifts.py`, `apply_corrections` and `invert_corrections`.
- **Finding:** the first detection modifies `filled_data_matrix`. The periodic
  fallback then operates on that corrected matrix. If the first analysis is
  preferred, only `time_shift_analysis` is restored; the matrix and `time_shifts`
  flag retain the second call's state. Each stored correction is relative to its
  own input, so extra-column correction and mask inversion can use an inconsistent
  transformation.
- **Owner disposition, 2026-10-21:** remove the automatic periodic retry;
  exact 365-day clock-shift recurrence is explicitly opt-in. Test a diagnostic
  recommendation flag during implementation, without automatically changing
  the model or applying another correction (SHIFT-01).
- **Proposed correction:** run the explicitly selected model against the
  original input and adopt its analysis, corrected matrix, and flags together.
  This supersedes the earlier proposal to repair the automatic two-candidate
  selection path. Any explicitly requested comparison must use the same original
  input for each model.
- **Additional caller finding, 2026-11-11:** correction updates the filled
  matrix, and the final daytime fit uses that corrected matrix. `Dilation`
  defaults to the original raw matrix but uses the corrected sunrise/sunset
  boundaries. `PVQuantiles` uses that default, and `plot_bundt` explicitly
  requests `matrix="raw"`. Removing the automatic retry does not repair this
  coordinate mismatch. Require the signal, daylight bounds, and masks entering
  dilation to share one clock frame. Preserve original observed values and
  missingness; an explicitly identified corrected observed-data view is a
  proposed solution, rather than substituting filled data for observations or
  silently redefining the raw matrix.
- **Mask export finding, 2026-11-11:** `augment_data_frame` currently
  inverse-rolls every sub-daily mask when `time_shifts` is true. This is
  appropriate for a mask computed in the corrected clock frame. The
  `missing_values` mask is computed before relative clock correction, so
  exporting it through that same path would move its entries away from their
  observed timestamps. Specify how callers identify the frame
  of a sub-daily mask and when export applies the inverse correction.
- **Proposed verification:** exercise recurrence off and explicitly enabled,
  verify that a recommendation cannot trigger a retry or alter corrected data,
  and check extra-column alignment and transformation/inverse-transformation
  consistency. Coordinate with SHIFT-01 and SD-03; support-index correctness
  remains distinct from consistent adoption of correction state.
  Carry a known nonzero correction through PV quantiles, Bundt plotting,
  dilation/undilation, and mask export; verify both coordinate alignment and
  preservation of the original observation mask.

#### DH-03 — Unit handling overrides explicit input and labels energy by magnitude [P2]

- **Location:** `run_pipeline`, lines 432–434; `make_filled_data_matrix`, lines
  1127–1130; `plot_daily_energy`, approximately line 2057;
  `run_loss_factor_analysis` and `algorithms/loss_factor_analysis.py` reports.
- **Finding:** input declared as watts is relabeled kilowatts when the estimated
  capacity is at most 500, without numerical conversion. A 300 W signal can be
  reported as approximately 300 kW. Daily energy plotting chooses conversion by
  magnitude, while loss-factor output is labeled kWh even when the supplied energy
  is in Wh.
- **Owner decision, 2026-10-22:** remove magnitude-based unit inference entirely.
  Retain the existing `units="W"` default and honor the caller's selected units.
  Do not introduce an automatic-inference mode or relabel a signal based on its
  estimated capacity or energy magnitude.
- **Proposed correction:** carry declared units through capacity estimation,
  daily energy, plotting, and loss-factor reporting. Integration over hours maps
  W to Wh and kW to kWh; propagate that energy unit from the handler to LFA.
  Make the energy-unit contract explicit for direct LFA calls as well. Where a
  plot or report requests another supported display unit, convert numerical
  values from the recorded source unit and label the result consistently.
  Remove the 500-unit capacity heuristic and magnitude-triggered daily-energy
  conversion. Preserve stored scientific arrays when converting for display,
  and document the changed behavior for callers that relied on implicit
  relabeling.
- **Proposed verification:** supply the same physical signal in W and kW,
  including a small system, and compare capacity, energy, plots, and loss reports
  after conversion to common units. Cover omitted units using the W default,
  values on either side of the old magnitude thresholds, and explicit display
  conversions. Verify consistent energy units through the handler/LFA boundary
  and unchanged source arrays after reporting.

#### DH-04 — Replace automatic timezone correction with a midnight-production warning [P2; revised disposition]

- **Locations:** `run_pipeline`, approximately lines 628–651; missing-value mask
  construction near line 435; extra-column processing near lines 659–679;
  `standardize_time_axis` in `time_axis_manipulation.py`, approximately lines
  137–192, and its caller around `data_handler.py:399–408`; pipeline reporting.
- **Finding:** the remaining-offset correction shifts the DataFrame along its
  chronological index but applies `np.roll(..., axis=0)` to each daily matrix.
  The latter wraps values within their original day instead of transferring
  them across midnight. The DataFrame path also applies `.fillna(0)` to missing
  values, while the previously constructed missing-observation mask is not
  shifted. Extra matrices subsequently generated from the DataFrame therefore
  need not align with the power matrices or their observation masks.
  Earlier preprocessing also infers an offset from average solar noon and
  shifts the index when its magnitude exceeds four hours. The final heuristic
  applies another offset when its magnitude exceeds one hour.
- **Owner direction, 2026-10-22:** replace automatic timezone correction with
  an advisory warning when significant power production near midnight suggests
  a substantial timezone misalignment. Apply this direction to both the initial
  preprocessing heuristic and the final pipeline heuristic. Do not shift
  timestamps or values to align inferred solar noon with clock noon.
- **Disposition and proposed correction:** supersede repair of the automatic
  offset transformation and the proposed choice between extending the date range
  and discarding boundary observations. Remove both automatic correction paths;
  retain a diagnostic for possible timezone problems based on meaningful
  midnight production. Specify and validate its near-midnight window, power
  threshold, and treatment of missing data during implementation; the existing
  solar-noon offset thresholds are not an adopted warning rule.
  The warning itself must leave timestamps, data, date range, and masks unchanged.
  Preserve ordinary timestamp standardization (TIME-01/TIME-02) and the separately
  requested relative clock-shift detection/correction (SD-01/SHIFT-01). Explicit
  utilities for caller-supplied timezone information are outside this retirement.
  Document retirement of `correct_tz`'s correction behavior and the former
  inferred-offset return/report fields in the migration guide. Expose the
  advisory result distinctly from an applied correction; it does not establish
  a particular timezone or offset.
- **Proposed verification:** shifted production profiles with and without
  meaningful midnight production, low-level noise, isolated anomalies, and
  missing midnight observations. Check warning behavior and unchanged timestamp,
  value, and mask alignment across all views, including extra columns and record
  boundaries. Verify that neither old correction path remains active and that
  the agreed relative clock-shift workflow is preserved. The initial heuristic
  was intended to avoid generation crossing midnight; evaluate downstream
  daytime extraction and daily analysis on such warned inputs rather than assume
  that a warning alone makes their results valid. This is separate from the
  candidate-state defect in DH-02 and timestamp snapping in TIME-01.

#### DH-05 — Bundt aggregation merges different calendar dates [P2]

- **Location:** `plot_bundt`, approximately lines 2502–2523.
- **Finding:** after removing February 29, the wrapper decrements every
  day-of-year greater than 59, including dates in nonleap years. Nonleap March 1
  is assigned to February 28, and March-through-December values in mixed-year
  records are aggregated under different calendar dates.
- **Proposed correction:** define the 365-day aggregation by month and day, or
  adjust only leap-year dates after February. Keep the February 29 exclusion
  explicit and preserve unobserved calendar bins as missing.
- **Proposed verification:** exercise `DataHandler.plot_bundt` with distinct
  calendar-day values in leap, nonleap, and mixed-year records. Check February
  28, March 1, and December 31, including the array passed to the renderer.
  Existing `test_bundt_plot.py` exercises only `plot_bundt_cake`.

#### DH-06 — Bundt initialization fails on an unprocessed handler [P2]

- **Locations:** `plot_bundt`, approximately lines 2488–2493; constructor
  initialization of `daytime_analysis`; `run_pipeline` signature.
- **Finding:** a fresh handler has `daytime_analysis=None`, but the guard
  dereferences `.sunrise_estimates` on that value. If the guard does reach its
  fallback, `run_pipeline(sunrise_sunset=True)` supplies an unsupported argument.
- **Owner decision, 2026-10-22:** require explicit preprocessing before plotting.
  `plot_bundt` must not run the pipeline automatically.
- **Proposed correction:** check that the required processed data and daytime
  estimates are available, handling absent or incomplete analysis explicitly.
  Otherwise raise a clear prerequisite error directing the caller to run
  `run_pipeline` with their chosen settings. Remove the implicit preprocessing
  fallback. A completion flag alone is insufficient if the required results
  are absent after a failed or partial pipeline run.
- **Proposed verification:** fresh, partially initialized, and processed
  handlers through the public wrapper. Verify that missing prerequisites produce
  the actionable error without starting preprocessing or mutating analysis
  state, and that a prepared handler plots successfully. Coordinate with
  DH-05's calendar checks.

#### DH-07 — Pipeline continuation loses numerical failure information [P2]

- **Locations:** `run_pipeline`, approximately lines 506–626 and 720;
  `report`, approximately lines 764–820; downstream stage calls including
  `run_loss_factor_analysis`, approximately lines 912–939; Dask result collection
  in `sdt_dask/dask_tool/runner.py`, approximately lines 110–173.
- **Finding:** the pipeline catches several stage failures and continues, but
  their outcomes are not consistently preserved in returned reports. A failed
  time-shift analysis sets `time_shifts=None`; `report()` converts this to
  `False`, losing the distinction from a successful no-shift result. The Dask
  wrapper primarily detects failure through exceptions and can leave its
  `run_pipeline error` field at `"No error"` after an internally handled stage
  failure. A completed call or `_ran_pipeline=True` does not establish that each
  requested analysis produced usable outputs.
- **Owner direction, 2026-11-11:** favor documented failures and continuation
  in pipeline applications, without requiring downstream users to configure
  special numerical-exception handling.
- **Proposed correction:** use the native failure-result contract in SD-01 and
  carry a small per-stage status record through `DataHandler` reports and Dask
  summaries. Preserve successful results and the explicitly agreed usable
  fallbacks. Continue stages whose required inputs are available; when a required
  upstream estimate is unavailable, record the dependent stage as skipped with
  its reason. Do not silently substitute a no-change finding, stale fit, or a
  different component model. Respect the configured dependencies: an optional
  input that was explicitly disabled does not become a required prerequisite.
  If time-shift correction has no usable fit, retain the pre-correction data and
  report correction failure; the remaining pipeline may use that data where its
  own prerequisites are satisfied. Keep numerical failure, a usable warned
  fallback, a successful negative finding, and an unrequested/skipped stage
  distinguishable. Reporting must retain available metrics even when another
  stage is unavailable. Record diagnostics independently of `verbose` printing.
- **Concrete dependency gates, follow-up review:** the final daytime fit in
  `run_pipeline` (around line 653) is outside the earlier exception handlers;
  preserve partial reporting if it fails and skip consumers requiring its
  unavailable bounds. `run_loss_factor_analysis` must not dereference capacity
  labels when that option is disabled. `detect_clear_sky(loss_correction=True)`
  currently checks only object existence and reads positional factors; require
  a usable loss fit and named degradation/soiling factors in the appropriate
  coordinates. Check these gates through the public wrappers. Dask must retain
  partial metrics together with stage failures and applicability warnings;
  LFA-06 owns removal of its independent short-record gate.
- **Proposed verification:** inject stage-level numerical failures without live
  solver or cloud dependencies. Verify continuation of independent stages,
  explicit skipping of dependent stages, preservation of successful results,
  and partial reports with the failure reasons intact. Exercise time-shift
  failure with unchanged input data and no `False` substitution for its detection
  result. Verify that Dask records the partial outcome and continues other data
  keys without requiring a user-supplied exception handler. Include a repeated
  run to exclude stale results from the reported outcome.
- **Related:** SD-01/SD-06 define numerical failure results; SD-03/SD-04 define
  usable fallbacks; DH-02 owns correction-state consistency. This item does not
  authorize a new pipeline framework or execution of the proposed checks.

### `solardatatools/time_axis_manipulation.py`

#### TIME-01 — Timestamp snapping fills absent observations [P2]

- **Location:** `standardize_time_axis`, approximately lines 244–250.
- **Finding:** nearest-neighbor reindexing with `limit=1` and no temporal tolerance
  copies a neighboring value into an isolated omitted scan. The downstream
  missing-value mask then treats that value as observed, affecting quality scores
  and fits using the raw matrix.
- **Owner decision, 2026-10-22:** when distinct observations map to the same
  regular-grid timestamp, retain the observation closest in time and report the
  collision. Do not average the colliding observations. Preserve the selected
  source row consistently across columns, and report which observations were
  retained or discarded.
- **Tolerance decision, 2026-11-11:** use $\Delta/2$ as the default maximum
  timestamp displacement, where $\Delta$ is the nominal sampling interval, and
  allow a caller override. Express the tolerance in physical time and use the
  same $\Delta$ as grid construction (TIME-02). Validate the rule with timestamp
  jitter and different acquisition cadences before adoption.
- **Proposed correction:** distinguish timestamp snapping from imputation. Map
  each source observation to its nearest grid timestamp when its displacement
  is within the configured tolerance, including equality at the limit, and to
  at most one grid timestamp. Validate the supplied tolerance and report
  observations excluded by it. Resolve competing observations at a grid timestamp with the
  agreed closest-observation rule. These are separate requirements: collision
  resolution alone does not prevent reuse of one observation at several grid
  entries. Do not move a discarded collision candidate into a neighboring empty
  bin. Preserve unmatched grid entries as missing and retain the observation
  mask. For a source timestamp exactly midway between two grid timestamps, use
  the earlier grid timestamp as the proposed deterministic tie-breaker. For
  equally close observations competing for one grid timestamp, use source order
  as the proposed tie-breaker, consistent with the current first-occurrence
  treatment of exact duplicate timestamps. A tolerance override does not relax
  the one-use or collision rules. Document the tolerance and assess
  compatibility with existing irregular-timestamp inputs during implementation.
- **Proposed verification:** an exact grid with one omitted interior sample;
  bounded timestamp jitter; collisions with different distances and tied
  distances; leading/trailing gaps. Cover displacements below, at, and above an
  overridden tolerance, exact half-interval ties, and different physical sampling
  intervals, including sub-minute and nonintegral-minute cadences. Verify
  closest-row selection, collision
  reporting, deterministic ties, and that no source observation fills more than
  one grid entry. Check unmatched bins and the observation mask, not only shape.

#### TIME-02 — Sampling intervals are rounded or truncated inconsistently [P2]

- **Locations:** `standardize_time_axis`, approximately lines 195–203;
  `data_handler.py`, `make_data_matrix`, lines 1104–1107;
  `matrix_embedding.py`, exported `make_2d`, lines 39–46;
  `algorithms/sunrise_sunset_estimation.py`, lines 78–84 and 299–305.
- **Finding:** ten-second rounding maps a five-second interval to zero and a
  fifteen-second interval to twenty seconds. Integer-minute conversion in sunrise
  estimation maps thirty-second sampling to a zero step and ninety-second sampling
  to a grid with the wrong number of rows. `DataHandler` also truncates
  nonintegral-minute intervals. Use of `.seconds` loses the day component of an
  elapsed duration. `make_2d` also uses `.seconds`; its fallback truncates elapsed
  seconds, and both paths truncate the inferred number of samples per day.
- **Proposed correction:** retain a single physical interval $\Delta$, use total
  elapsed seconds, and construct within-day coordinates as
  $24\operatorname{arange}(m)/m$ for an $m$-row daily matrix. Define and explicitly
  validate unsupported grids rather than changing their cadence silently. For
  `make_2d`, validate the regular grid and integral samples-per-day requirement
  before reshaping; do not infer a different cadence by integer truncation.
- **Additional caller findings, 2026-11-11:** extra-column embedding in
  `DataHandler.run_pipeline` converts the interval to integer seconds, making
  a 0.5-second cadence `0s`. Mask export in `augment_data_frame` reconstructs
  its own minute-based index, and profiler hour-angle coordinates are generated
  separately from `data_sampling`. Propagate the exact interval and aligned
  grid through all three paths; validate row agreement rather than inferring
  it from independently rounded periods.
- **Incomplete-day finding:** direct `make_2d` reshapes
  contiguous values but obtains its day axis by calendar resampling. A regular
  24-hour record beginning at 06:00 can produce one matrix column and two dates.
- **Incomplete-day decision, 2026-11-11:** by default, pad missing time slots
  in the first and last calendar days with NaN. Preserve observations in their
  actual time-of-day rows and return one matching date per matrix column.
  Retain explicit trimming options; do not discard a partial day merely to
  make the array reshape succeed. Validate the supported cadence and grid
  alignment before embedding; padding does not authorize resampling or moving
  observations to different timestamps. It supplies missing entries, not
  zero-valued production or model-imputed observations.
- **Proposed verification:** sub-minute, nonintegral-minute, and ordinary
  five/fifteen-minute data; agreement among timestamps, matrix rows, daytime masks,
  and reported sampling. Include direct `make_2d` calls, fractional-second
  intervals, and rejection of unsupported grids. Preserve the distinction
  between daily model periods and sub-daily acquisition cadence.
  Exercise extra columns, mask export, and profiler angle coordinates on the
  same fractional-second grid. Include direct partial-day calls and verify
  values, missingness, time-of-day rows, and returned dates under both default
  padding and explicit trimming. A supported 24-hour record beginning at 06:00
  must span two dated columns, with NaN before its first sample and after its
  last sample, while retaining every observation in its proper row.

### `solardatatools/matrix_embedding.py`

The exported `make_2d` cadence and reshape contract is tracked in TIME-02 with
the other physical-time conversions; no duplicate correction is intended here.

### `solardatatools/data_quality.py`

#### QUALITY-01 — The default dataset score is not the documented passing fraction [P2]

- **Location:** `dataset_quality_score`, approximately lines 113–130;
  `make_density_scores`, approximately lines 29–53.
- **Finding:** with `good_days=None` and the default `use_advanced=True`, the
  function averages continuous density-to-baseline ratios. Its docstring instead
  specifies the fraction of days passing the missing-data test, bounded by zero
  and one. Those ratios are not Boolean decisions and are not bounded by one.
- **Owner decision, 2026-10-22:** use the documented fraction of days passing
  the missing-data/density checks. Retain continuous density ratios as daily
  diagnostics, not as the aggregate quality score.
- **Proposed correction:** for $N>0$ days and Boolean pass indicators $g_t$,
  return $N^{-1}\sum_{t=1}^N g_t$. Use the same aggregation for the default,
  simple, and supplied-`good_days` paths; validate the supplied daily mask.
  For the advanced path, apply an explicit density acceptance rule before
  averaging. Reuse the existing density-flag criteria as the proposed default
  (currently $0.6<\mathrm{density\_score}<1.05$), rather than introduce another
  classification rule. Distinguish these acceptance thresholds from the
  existing `threshold=0.2`: in the simple path it classifies daily density,
  whereas in `make_density_scores` it selects observations for fitting the
  density baseline. Document that distinction and the migration of any renamed
  arguments. Specify unavailable-day and failed-score handling without silently
  treating a failed computation as a valid passing fraction.
  `DataHandler.score_data_set` already averages Boolean `no_errors` flags;
  retain that separate overall-quality definition, which also incorporates
  other quality checks. The standalone density-based score need not equal it.
- **Proposed verification:** distinguish continuous ratios from pass indicators
  using analytically specified cases; check default, simple, and supplied-mask
  paths against their documented semantics, including out-of-range ratios and
  unavailable days. Clipping a continuous mean to $[0,1]$ does not make it a
  fraction of passing days.

### `solardatatools/sensor_identification.py`

#### SENSOR-01 — Temporal validation receives time-of-day-major ordering [P2]

- **Location:** `SensorIdentification.identify`, approximately lines 78–88.
- **Finding:** two-dimensional Boolean indexing traverses the daily matrix in
  row-major order: all dates at one time of day precede the next time of day.
  `TimeSeriesSplit` therefore allows later observations into training before
  earlier observations in validation. The reported errors and sensor ranking do
  not represent the intended chronological validation.
- **Proposed correction:** flatten the common mask, response, and each sensor
  matrix with `order="F"` before applying the mask, or explicitly order by the
  corresponding timestamps.
- **Proposed verification:** multiple dates and missing sensor observations;
  recover timestamps for every partition and verify
  $\max t_{\mathrm{train}} < \min t_{\mathrm{validation}}$.
- **Related:** PKG-01 covers the separate `np.alltrue` compatibility failure.

#### SENSOR-02 — Unavailable subsets and failed folds break sensor identification [P2]

- **Location:** `SensorIdentification.identify`, approximately lines 74–125.
- **Finding:** the method always evaluates `no_errors`, `clear`, and `cloudy`
  subsets. Empty or undersized subsets fail when the split iterator advances,
  outside the fit exception handler. A failed fit appends scalar `np.inf` to a
  list subsequently passed to `np.concatenate`, causing another failure instead
  of producing a usable failed-fold result.
- **Owner decision, 2026-10-22:** return rankings for usable subsets when another
  subset has insufficient data, and explicitly flag incomplete coverage.
  An unavailable clear or cloudy comparison does not invalidate the other
  successfully evaluated subset results.
- **Proposed correction:** validate subset/split feasibility before fitting and
  represent unavailable subsets and failed folds explicitly. Return each usable
  subset's ranking with coverage information and reasons for unavailable
  comparisons. Do not silently rank NaNs or substitute an unavailable subset's
  result with another subset's winner. `consistent_answer` must not report
  agreement across all three conditions when any comparison is unavailable.
  Preserve any observed disagreement among the tested subsets while separately
  reporting incomplete coverage; missing evidence is not itself disagreement.
  Document the resulting selection/consistency contract and any changed return
  representation in the migration guide. Failed fits are not usable validation
  results; retain enough fold and observation coverage information to ensure
  that scores entering a ranking refer to comparable validation populations.
- **Proposed verification:** all-clear and all-cloudy records, too few retained
  observations, and forced fit/prediction failures. Verify that usable rankings
  remain available, unavailable comparisons are identified, and incomplete
  coverage never produces an all-conditions agreement claim. Include both
  agreement and disagreement among available subsets and no usable subsets.
  Check these semantics alongside the chronological partitions in SENSOR-01.

## 2. Signal-decomposition implementations and dispatch

### Owner direction and mathematical decisions

**Direction recorded on 2026-10-21:** retire the OSD path and implement simple
signal decompositions natively in this project with CVXPY, allowing any
appropriate solver, including CLARABEL and MOSEK. The focus is code
simplification, correct factoring, IRL1, Tier-3 parameter selection, and efficient
re-solves. This direction supersedes proposals to repair or restore OSD/QSS
features. It does not authorize implementation of the proposals below.

**Solver decision, 2026-11-11:** use CLARABEL with its default settings as the
default across native decomposition entry points. Provide pass-through to CVXPY
for another appropriate solver and for solver-specific settings; preserve these
choices through the calling pipeline and repeated solves (SD-01/SD-06).

**Caller migration, follow-up review:** explicitly replace the competing
`DataHandler.run_pipeline` options `solver` and `solver_convex` with the agreed
solver/options pass-through contract. Its current MOSEK probe returns on failure
before resetting prior results; retire that probe and solver-dependent model
dispatch. Start each requested run with explicit current-run status, route
expected solve failures through DH-07, and prevent prior estimates from being
reported as results of a failed new run. The profiler's separate threshold
solve is covered in PROF-02. Verify public callers as well as low-level builders.

Solver selection should affect numerical solution of an agreed problem, rather
than select a different estimator. Simply routing every solver to the existing
`_cvx_*` functions would change the current default models and preserve defects.
Before implementation, specify each model's residual loss, component roles,
feasible sets, penalty normalization, physical time scale, and extraction rule.

The proposed common convention is

$$
y_t=x_t^{\mathrm{residual}}+\sum_{k=1}^K x_t^{(k)},\qquad t\in\Omega,
$$

where $\Omega$ contains finite observations admitted by the caller's mask.
**Owner clarification, 2026-11-11:** all components, including the residual, are
defined on the entire model index: $x^{(k)}\in\mathbf R^T$ for every component.
The mask restricts the consistency equality, not the domain of the component
signals. For the predefined residual costs used here, the residual entries
outside $\Omega$ minimize their separable cost at zero, so
$x_t^{\mathrm{residual}}=0$ for $t\notin\Omega$. This is part of the model
construction, not a choice to replace solved residual entries with missing
values. Return the full-grid components and retain the known-set mask
separately. The structural reconstruction
$\widehat y_t=\sum_{k=1}^K x_t^{(k)}$ is defined throughout the grid; outside
$\Omega$ it also equals the sum of all components and supplies the imputed
signal. Raw missing observations remain missing in the input record.
Downstream code should access components by role. **Owner decision, 2026-10-21:**
use a Fourier basis with physical annual period $P=365.2425$ days as the common
default for smooth annual components, with explicit model-specific exceptions.
Harmonic count and regularization remain model-specific. **Owner decision,
2026-10-22:** the scalar quantile smoother used by density and sunrise/sunset
estimation supports annual Fourier mode only; retire its nonperiodic mode.
The owner has specified the level allocation: for time shifts, the Fourier
seasonal component carries the offset and serves as the solar-noon estimator;
for capacity changes, the PWC component carries the offset and serves as the
local-capacity estimator. **Owner decision, 2026-10-21:** set the first time-shift
PWC segment to zero. With $c$ the clock-shift component, impose $c_0=0$ on the
model grid; the Fourier offset gives the solar-noon level in that initial clock
regime. Removing a Fourier DC column and centering a component over a finite
record are different normalization conventions. Express periods relative to the
physical sample interval $\Delta$.

**Additional owner decisions, 2026-10-21:** retain exact 365-day recurrence of the
clock-shift component as an explicitly opt-in model assumption, default off,
independently of the Fourier seasonal period. Remove the automatic periodic
retry and test a recommendation flag during implementation; a recommendation
does not enable recurrence. Use IRL1 as the standard step-selection procedure
for both time shifts and capacity changes, followed by one convex fixed-support
refit for each estimator. Retain and document the existing model-specific
residual losses and normalization; broader model updates are future work.
Previously accepted design changes remain explicit exceptions to preservation
of the existing models. Retain $\epsilon=0.1$ in each estimator's existing units,
with tolerance-based IRL1 stopping and a hard iteration cap. For capacity, select
refit support from final IRL1 jumps exceeding 0.001 log units and form labels
after refitting. **Owner scope clarification, 2026-10-21:** preserve the capacity
selector's design; it is the archetype for the skill's Tier-3 discussion.
Capacity remains subject to focused review, without a sweeping selection
redesign. Update time-shift parameter selection to a Tier-3 rule using that
archetype, with validation before adoption. These decisions do not authorize
implementation. The provisional IRL1 defaults agreed on 2026-11-11 are one
ordinary-l1 initialization, up to twenty reweighted solves, and a relative merit
decrease tolerance of $10^{-4}$; SD-04 specifies the test and required validation.
The minimum meaningful clock jump is five minutes ($1/12$
hour), replacing the existing three-minute pooled support cutoff. Screen
time-shift candidates at no more than ten clock changes per year, using the
record-wide annualized count; this is a proposed domain bound to validate at
implementation time. Also require at least a 2% improvement in in-sample squared
residual error relative to a constant-shift fit, subject to validation. Choose
the smallest weight satisfying these screens. If no candidate qualifies, or
the constant-shift reference fit has zero residual error, return the constant
shift component: $c=0$ under the agreed reference, with no relative correction.
Round applied corrections
directly to the nearest sample by default, retaining whole-hour
rounding as an explicit option (SHIFT-01). The proposed abstention machinery,
stability-based ranking framework, and refitting of every distinct candidate
support are withdrawn. Retain the agreed single fixed-support refit after
selection for each estimator. All algorithm changes require thorough testing
at implementation time; the specification decisions here do not establish
validated performance or authorize execution now.
A cleaner breaking public API is permitted, with an explicit migration guide;
the low-level native decomposition functions will return one lightweight result
object with named components and diagnostics. Exact entry points and field names
remain to be specified before implementation.

**Owner decision, 2026-10-22:** when IRL1 reaches its iteration cap after
successful convex subproblem solves, continue from the last iterate through
selection and refitting, explicitly reporting iteration-limit termination.
This is not convergence and does not authorize using a failed solve's values
(SD-04).
If a later reweighted solve fails numerically after a successful fit, retain
the last successful iterate for that candidate and continue through selection
and refitting, reporting failed-refinement termination. Failure of the initial
fit leaves no usable candidate. These outcomes remain distinct from convergence
and iteration-limit termination. If the selected fixed-support refit fails,
return the selected IRL1 estimate and explicitly flag the failed refit (SD-03).
Weight searches continue over usable candidates after numerical initial-fit
failures, reporting the omitted weights. A missing required reference or a search
with no usable candidate is a numerical failure, not a no-change finding.

### `solardatatools/signal_decompositions.py`

#### SD-01 — Solver dispatch changes the estimator and discards options [P2]

- **Locations:** time-shift dispatch around lines 84–122; quantile dispatch around
  lines 163–183; constrained-filter dispatch around lines 246–254. Related:
  `_cvx_signal_decompositions.py:221–224`, `algorithms/time_shifts.py:73–78`, and
  QSS branches in `data_handler.py:1362–1410`.
- **Finding:** the current solver branches select the following different models.

  | Estimator | CLARABEL/default OSD path | MOSEK/CVXPY path |
  | --- | --- | --- |
  | Time shifts | Three-harmonic annual Fourier component and ordinary TV | Smooth 365-periodic vector and five IRL1 passes |
  | Seasonal quantile | Three-harmonic Fourier component; ignores `yearly_periodic=False` | Second-difference regularization with optional 365-periodicity |
  | Clipping CDF | Uses supplied `w0` and `w1` | Discards supplied `w0` and `w1` |

  Dispatch also forces `sum_card=False`. The capacity implementation replaces
  other solver choices with CLARABEL despite its convex formulation.
- **Time-shift compatibility record:** in the current CVXPY formulation,
  `s_seas` is 365-periodic whenever the horizon exceeds 365 samples,
  independently of `yearly_periodic`. Setting that argument additionally
  constrains the shift component `s_hat` to be 365-periodic. The fixed-support
  CVXPY branch retains both constraints. The OSD discovery branch can impose
  shift periodicity, but its fixed-support basis branch drops that constraint.
  The seasonal component is periodic in both OSD branches. Thus
  `yearly_periodic=False` in the time-shift API does not disable seasonality;
  the seasonal-quantile API uses the same name for a different component.
- **Scalar-quantile decision, 2026-10-22:** retain annual Fourier seasonality
  as the only mode of the scalar quantile smoother (`tl1_l2d2p365` and its
  replacement). Retire the nonperiodic second-difference smoothing option and
  its periodicity toggle. Use $P=365.2425$ days with the agreed physical-time
  convention, configurable quantile loss, and documented estimator-specific
  harmonic/regularization settings. Solver choice must not restore the retired
  vector smoother or change the annual model. This decision is specific to the
  scalar quantile estimator; the opt-in 365-day recurrence constraint on the
  clock-shift component remains supported.
  Document the removal of `yearly_periodic=False` from this quantile API in
  the migration guide. If a compatibility wrapper is retained, it must explicitly
  reject the retired nonperiodic request rather than silently fit an annual
  model. Migrate callers and tests to the annual-only contract.
- **Owner decision, 2026-10-21:** retain exact 365-day shift recurrence as an
  explicitly opt-in option in the native model, default off. Keep its meaning
  distinct from the 365.2425-day Fourier seasonal default. Remove the pipeline's
  automatic periodic retry; test a recommendation flag without enabling the
  constraint automatically (SHIFT-01 and DH-02). Preserve the constraint during
  any fixed-support refit when requested; public naming and behavior on grids
  without an integral 365-day lag remain to be specified.
- **Component-role decision, 2026-10-21:** the time-shift Fourier component
  includes the constant offset and is the solar-noon estimate; the PWC component
  represents clock shifts relative to its first segment, fixed at zero. This
  differs from current extraction, which treats `s1` as an absolute clock time
  and `s2` as a seasonal deviation. Translate downstream reference selection,
  plots, and diagnostics to the agreed roles. Any retained seasonal RMS
  diagnostic, including one considered for the proposed recommendation flag,
  must measure seasonal variation after removing the Fourier offset. Retire the
  RMS-triggered automatic retry in `DataHandler.run_pipeline`.
  For capacity, retain the level in the PWC component and the existing Fourier
  basis without a DC term; the linear trend remains anchored at its first value.
- **Reference-level decision, 2026-10-21:** use $c_0=0$ during discovery and
  refitting, so the first fitted PWC segment is zero. The Fourier constant term
  carries the offset without a shrinkage penalty on that level. This supersedes
  the current first-level/nearest-noon fallback for relative clock correction.
  Before rounding and conversion to samples, the relative correction is $-c_t$
  hours. Timezone checking is advisory under DH-04 and must not shift the data
  again or redefine the reported relative PWC reference.
- **API decision, 2026-10-21:** a cleaner breaking public API is permitted, with
  an explicit migration guide. Preservation of old signatures, tuple positions,
  or `return_all` behavior is not a requirement. Specify the replacement entry
  points and results, then document old-to-new calls, component meanings,
  solver/options handling, and dispositions of obsolete OSD-specific arguments.
  Acceptance of this design direction does not authorize source changes.
- **Solver/default decision, 2026-11-11:** default to CLARABEL using its default
  settings. Allow callers to select another appropriate solver and pass
  solver-specific settings through to CVXPY. Retire the restricted
  MOSEK/CLARABEL dispatch and its silent solver substitution; solver selection
  must not change the formulation. Document the default and pass-through calls
  in the migration guide, including how higher-level analysis and pipeline
  entry points forward those settings. Do not introduce package-specific
  tolerance or iteration-limit overrides as hidden defaults.
- **Result-object decision, 2026-10-22:** each low-level native decomposition
  function returns one lightweight result object containing components identified
  by role and the applicable solve diagnostics. Replace positional component
  tuples and `return_all` variants with this common convention. Include the
  accepted solver status and warnings, numerical-check outcomes, and, where
  applicable, IRL1 termination and fixed-support refit/fallback information.
  Distinguish stages that were not performed from successful stages. Consumers
  must be able to identify the returned estimate and its qualifications without
  interpreting backend-specific tuple positions or inspecting a CVXPY problem.
  Specify concrete role/field names and old-to-new access examples in the
  migration guide. This decision concerns the low-level decomposition functions;
  it does not require replacing every higher-level analysis class with a new API.
- **Failure-result decision, 2026-11-11:** an expected numerical failure with no
  accepted fit returns a result carrying explicit failure status and diagnostics,
  with component estimates unavailable. It must not require downstream callers
  to catch a dedicated numerical-solve exception. Do not fabricate zero or NaN
  component arrays to stand in for a nonexistent fit. A retained accepted fit
  under SD-03/SD-04 remains a usable result with its warning and provenance.
  Distinguish availability of a fit from the individual solver and refinement
  outcomes. The full-grid component contract below applies whenever an accepted
  fit is returned. Document these outcomes and their use in pipelines in the
  migration guide; DH-07 covers stage propagation and partial reporting.
- **Component-domain clarification, 2026-11-11:** return every component over
  the full time index, including the residual with its model-implied zeros
  outside $\Omega$ for the predefined residual costs. Include $\Omega$ in the
  result as the admitted-observation mask. The proposal to insert NaNs into the
  returned residual outside $\Omega$ is withdrawn. Preserve the structural
  reconstruction over the full grid and distinguish it from the original data
  and its missing entries. Fit-error summaries must respect their specified
  observation domain and normalization; a zero residual outside $\Omega$ does
  not make that index an observation.
- **Proposed correction:** use one agreed native CVXPY formulation per estimator,
  pass solver selection/options through to CVXPY, and preserve the meaning of
  public weights. Remove duplicate dispatch, solver-dependent search grids, and
  QSS/cardinality orchestration. Define retirement behavior for obsolete public
  arguments; do not silently restore a dead backend or substitute another solver.
  Retire the inconsistent `return_all` contracts in favor of the agreed result
  object and document the migration.
  Record every retained argument's meaning for each estimator: the observation
  domain selected by `signal`/`use_ixs`, the component and normalization controlled
  by each weight, the clock-shift component constrained by the retained
  recurrence option (distinct from the retired quantile toggle), the support
  convention in `transition_locs`, and the returned component roles. Distinguish
  these model arguments from solver/options/verbosity. Preserve the agreed
  component constraints during fixed-support refitting; changes to argument
  meanings and retired arguments must be explicit in the agreed API and its
  migration guide.
- **Proposed verification:** check objectives, constraints, masks, and parameter
  effects through public entry points. Compare solver results on the same
  mathematical problem, allowing for numerical tolerance and nonuniqueness.
  Verify role-based access and consistent diagnostic semantics across estimators
  and solvers, including qualified acceptance and retained-estimate fallbacks.
  Verify a numerical failure result with no component estimate, readable failure
  diagnostics, and no special exception-catching requirement at the public API.
  Verify full-grid component arrays, residual zeros outside $\Omega$ to the
  applicable numerical accuracy, consistency on $\Omega$, and reconstruction
  through internal gaps and excluded endpoints. Include finite held-out entries
  as well as missing input values; neither changes the full-grid component
  domain. Preserve the mask and raw-data missingness through extraction.
  Check periodicity of the shift and seasonal components separately with the
  flag enabled and disabled, both during discovery and fixed-support refitting;
  historical backend differences are not the new contract. For the scalar
  quantile smoother, verify annual Fourier behavior for every supported solver,
  explicit retirement of the nonperiodic option, and the density, clear-day,
  and sunrise/sunset callers against the agreed model. Do not preserve the old
  nonperiodic numerical fixtures as requirements of the replacement.
- **Reference:** [CVXPY solver selection and options](https://www.cvxpy.org/tutorial/solvers/index.html).

### `solardatatools/_osd_signal_decompositions.py`

#### SD-02 — OSD breakpoint repair superseded by backend retirement

- **Disposition:** retain the defect as evidence and a native regression case;
  do not schedule an independent OSD repair. Retirement is covered by SD-01 and
  PKG-03. Direct loss-factor use is covered by LFA-06.
- **Location:** `_osd_l2_l1d1_l2d2p365`, approximately lines 106–115. Caller:
  `algorithms/time_shifts.py`, approximately lines 125–135.
- **Finding:** `TimeShift` supplies indices of `np.diff(s1)`: Python index $j$
  denotes $x_{j+1}-x_j$. The basis treats $j$ as a segment boundary before sample
  $j$. For `[12, 12, 13, 13]`, transition index `1` yields segments `[0:1]` and
  `[1:4]`, instead of `[0:2]` and `[2:4]`. This can correct the day preceding a
  clock change. The CVXPY branch uses the difference-index convention.
- **Native contract to preserve:** a difference index $j$ permits the jump
  between samples $j$ and $j+1$; a segment-boundary representation must use
  `j + 1`. SD-03 and QA-04 cover first, interior, and last allowable transitions
  in the replacement. Tests need not preserve both implementations.

### `solardatatools/_cvx_signal_decompositions.py`

#### SD-03 — Fixed-breakpoint polishing changes the residual loss and repeats unchanged solves [P2]

- **Owner decision, 2026-10-21:** after support selection, perform one convex
  fixed-support refit for both time shifts and capacity changes. Remove the
  jump penalty, impose $(Dc)_i=0$ outside the selected support $J$, and retain
  the residual loss, other penalties, and structural constraints. For empty
  $J$, the PWC component is constant under the chosen level convention.
- **Refit-failure decision, 2026-10-22:** if the selected fixed-support refit
  fails numerically, return the selected IRL1 estimate and explicitly flag that
  the refit failed. Retain the candidate's accepted component values before
  attempting the refit; failed solver output must not replace them. This is a
  penalized, unrefitted estimate: jump shrinkage remains, and its differences may
  include small jumps excluded from the selected support $J$. Keep $J$ as a
  support-selection diagnostic without claiming that the returned estimate
  satisfies the refit constraints. Preserve the candidate's IRL1 termination
  diagnostics separately from the failed-refit flag.
- **Capacity-support decision, 2026-10-21:** use
  $J=\{i: |(Dc^{\mathrm{IRL1}})_i|>10^{-3}\}$ in log-capacity units, retaining
  the existing numerical jump tolerance. Form capacity labels from the refitted
  component, or from the returned IRL1 component under the explicit refit-failure
  fallback above. The 0.05-wide label bins and the existing 0.075 candidate-screening
  threshold are distinct from this support definition. Preserve the capacity
  screening design; CAP-02 is a focused review of its reference and fallback
  semantics. Refit the selected support once rather than adding a refit for
  every candidate. Refitting does not validate the selected support.
- **Time-shift-support decision, 2026-10-21:** retain the existing pooling of
  adjacent first differences and accept pooled jumps of magnitude at least
  five minutes ($1/12$ hour), replacing the current 0.05-hour cutoff. Apply this
  threshold in physical hours before rounding corrections; coordinate the
  minimum meaningful jump criterion in SHIFT-01 with this support definition.
- **Recurrence interaction, follow-up review:** ordinary linear-index pooling
  is not sufficient when exact 365-day recurrence is requested. On a 367-day
  grid, let $d=Dc$ have $d_0=d_1=d_{365}=0.1$ and $d_{364}=-0.2$ hours, all
  other entries zero. This is compatible with recurrence. Pooling adjacent
  differences at each group's last index retains $J=\{1,365\}$, with both
  pooled magnitudes exceeding five minutes. In the refit, $d_0=0$ forces
  $d_{365}=0$ by recurrence; closure of the first cycle then forces $d_1=0$.
  The refit succeeds but erases every shift. Specify phase-aware support
  selection for recurring mode, including cycle boundaries and partial final
  cycles, so the intended selected nonconstant structure remains representable.
  Merely replicating an already pooled support modulo 365 is not an established
  correction. Retain one final refit and the physical jump threshold; this
  resolves the interaction without changing the opt-in recurrence policy.
- **Location:** `_cvx_l2_l1d1_l2d2p365`, approximately lines 97–130.
- **Finding:** the unrestricted-breakpoint problem uses $\|r\|_2^2$, while the
  fixed-breakpoint branch uses $\|r\|_2$. The seasonal penalty is retained, so this
  changes the estimator and regularization tradeoff; it is not a monotone
  transformation of the complete objective. Documentation describes squared
  residuals. The fixed-support branch is still solved five times, although its
  objective no longer uses the updated IRL1 weights.
- **Proposed correction:** separate support discovery, IRL1 refinement, and
  fixed-support refitting. Preserve the residual loss and other component
  penalties and structural constraints during the refit, replacing only the
  sparsity penalty with fixed support constraints. This includes the agreed
  component-specific periodicity recorded in SD-01. Solve that convex refit
  once. Apply the difference-index convention retained in SD-02.
  Downstream labels, correction arrays, and plots must consistently use the
  returned set of components and expose whether it is refitted or the IRL1
  fallback. Do not combine a failed refit's partial values with retained
  candidate components or silently project the candidate onto $J$.
- **Proposed verification:** compare the assembled fixed-breakpoint problem with
  its agreed objective, feasible support, and solve count, rather than only a
  historical output fixture. Cover both estimators, including empty support,
  and form final correction/capacity outputs from the refitted components on
  success. Exercise numerical refit failure and verify exact retention of the
  selected IRL1 estimate, propagation of the failure flag, and consistent
  downstream outputs. Include a candidate with subthreshold jumps outside $J$
  to distinguish the fallback from a successful fixed-support refit.
  Include the recurring counterexample and a partial final cycle through the
  public time-shift call, checking recovered support and applied corrections,
  not only successful convex termination.

#### SD-04 — IRL1 needs explicit scaling, initialization, and termination

- **Owner decision, 2026-10-21:** IRL1 is the standard procedure for step
  selection in both time-shift and capacity-change estimation. Its use in these
  two estimators is settled, as is one subsequent fixed-support refit (SD-03).
  Retain $\epsilon=0.1$ in hours for time shifts and in log-capacity units for
  capacity. Replace fixed solve counts with tolerance-based stopping on the
  fixed log-sum merit function and a hard iteration cap. Provisional numerical
  defaults are specified below. Capacity support is defined
  in SD-03; retain existing model-specific loss normalization (SD-05). This
  procedure is distinct from the soiling selection and two interval-weighted
  refinement solves in LFA-06; SOIL-04 records retirement of the legacy
  standalone procedure.
- **Iteration-limit decision, 2026-10-22:** if the convex subproblems solve
  successfully but IRL1 reaches its cap before meeting the stopping tolerance,
  use the last iterate and continue through candidate selection and the selected
  fixed-support refit. Explicitly report that the cap was reached, with an
  iteration-limit warning and termination information. Do not classify this
  outcome as convergence or discard the candidate solely because of the cap.
  A subsequent successful convex refit does not change the recorded IRL1
  termination reason. Failed subproblem solves remain a separate case.
- **Failed-refinement decision, 2026-10-22:** if the initial fit succeeds but a
  later reweighted solve fails numerically, stop that candidate's IRL1 loop and
  retain its last successful iterate for selection and the selected fixed-support
  refit. Report the failed solve and the retained iteration explicitly; do not
  report convergence or an iteration-limit termination. A successful final
  refit does not erase the failed-refinement diagnostic. Failure of the initial
  fit leaves no usable candidate. This numerical-failure policy does not suppress
  invalid configuration or programming errors.
- **Candidate-search decision, 2026-10-22:** continue time-shift and capacity
  weight searches over usable candidates when some initial fits fail numerically.
  Record each failed weight and failure reason, exclude it from screening, and
  select the smallest qualifying weight among the usable candidates. Report that
  the evaluated path is incomplete; do not insert zero errors, zero jumps, or
  stale components for failed weights. A candidate retained after a later IRL1
  refinement failure remains usable under the preceding decision, with its own
  diagnostic. The required reference fit must also be usable. If it is
  unavailable, or the search has no usable candidate, report unsuccessful
  selection with the specific reason rather than infer no changes. Distinguish
  numerical fit failures from exhaustion of the bounded endpoint extension
  specified in CAP-02/SHIFT-01. Preserve the agreed no-qualifying-candidate
  fallback when screening usable fits with a valid reference, and preserve the
  time-shift zero-reference-error fallback without requiring an unnecessary
  candidate sweep. This policy does not change the screening criteria.
- **Locations:** time-shift loop around lines 88–130; capacity reweighting around
  lines 228–236. Related: SOIL-04.
- **Finding:** time shifts use an initial l1 solve followed by four reweighted
  solves; capacity uses an initial solve followed by one reweighted solve. The
  reciprocal updates have the standard IRL1 form. A fixed iteration budget is a
  truncated heuristic, not evidence of convergence or correct support. The same
  numerical `eps=0.1` has different meanings in hours and log capacity. Updates
  consume `.value` without checking status and finite numerical values.
- **Proposed mathematical contract:** let $z$ collect the model variables, let
  $D$ select and difference the component being regularized, and let $f$ contain
  the remaining convex objective. For fixed $\lambda>0$, $\epsilon>0$, and convex
  feasible set $\mathcal C$, use

  $$
  w_i^{(k)}=\frac{1}{\epsilon+|(Dz^{(k)})_i|},\qquad
  z^{(k+1)}\in\mathop{\rm argmin}_{z\in\mathcal C}
  \left(f(z)+\lambda\sum_i w_i^{(k)}|(Dz)_i|\right).
  $$

  This is a majorization–minimization step for

  $$
  F(z)=f(z)+\lambda\sum_i
  \log\left(1+\frac{|(Dz)_i|}{\epsilon}\right).
  $$

  The subproblems are convex; the overall procedure is nonconvex. First
  differences promote steps and second differences promote changes in slope.
  The ordinary-l1 initialization is distinct from the subsequent merit function;
  it is not a guarantee of correct support recovery after any fixed number of
  passes.
- **Provisional numerical defaults, owner decision 2026-11-11:** for each weight
  candidate in both estimators, initialize once with ordinary l1 (unit adaptive
  weights), then allow at most twenty reweighted solve attempts. The
  initialization and the selected final fixed-support refit do not count toward
  that cap. Retain $\epsilon=0.1$ and the agreed model-specific loss/penalty
  normalization. Let $F_k=F(z^{(k)})$ be the independently evaluated fixed merit
  at accepted iterate $k$, with $z^{(0)}$ the ordinary-l1 initialization. For
  $F_k>0$, the provisional stopping test is

  $$
  0\leq F_k-F_{k+1}\leq\tau F_k,\qquad \tau=10^{-4}.
  $$

  An accepted iterate with exactly zero merit stops immediately, including at
  initialization; the agreed merit is nonnegative. No absolute stopping term is
  added to this provisional test. A merit increase does not satisfy the test and
  must not be reported as convergence. Make the relative tolerance and iteration
  cap configurable. These are starting values for numerical and runtime
  validation before adoption, not validated production defaults. Record the
  effective settings and actual initialization/reweighting counts in diagnostics.
- **Merit-increase decision, owner decision 2026-11-11:** if an otherwise
  numerically acceptable convex solve produces a trial merit
  $\widehat F_{k+1}$ with $\widehat F_{k+1}-F_k$ exceeding the allowed numerical
  tolerance, reject that step and stop reweighting. Retain $z^{(k)}$ for candidate
  selection and the selected fixed-support refit, and issue an explicit warning.
  Record merit-increase termination separately from tolerance termination,
  iteration-limit termination, and a failed convex solve. Preserve the rejected
  solve's status, numerical-check results, and both merit values as diagnostics;
  its component values must not replace the retained estimate or update the
  adaptive weights. The rejected solve counts toward the reweighting cap.
- **Provisional merit-increase allowance, owner decision 2026-11-11:** for
  $F_k>0$, use the configurable relative allowance $\eta F_k$, with
  $\eta=10^{-6}$ and no absolute term. Reject and terminate as above when

  $$
  \widehat F_{k+1}-F_k>\eta F_k.
  $$

  A smaller positive increase, including equality at the allowance, may be
  accepted after the other numerical checks pass; record it and continue
  reweighting within the cap. It does not satisfy the convergence test. An
  accepted nonincreasing trial is evaluated against the separate stopping
  tolerance $\tau=10^{-4}$. Exactly zero merit retains the immediate-stop rule,
  so no division or numerical floor is needed at zero. Record the effective
  allowance in diagnostics. Validate this provisional value jointly with the
  stopping tolerance and cap before adoption. Keep support/component stability
  as a validation check; the provisional stopping rule is merit-based. Neither
  a merit increase nor reaching the hard cap is convergence.
- **Proposed correction:** specify component units, difference scaling,
  initialization, $\epsilon$, iteration limit, and stopping criteria. Validate
  solve status and finite values before updating weights or publishing results.
  Record progress in the fixed merit function $F$; successive weighted-surrogate
  objective values are not directly comparable. Retain a copy of each accepted
  iterate and its diagnostics before the next solve can overwrite variable
  values. Recovery must use that candidate's last accepted fit, never failed
  solver output or values left over from another weight, mask, or estimation
  run. Build once and update parameters as described in SD-06.
- **Proposed verification:** independently evaluate $F$ and its decrease within
  numerical tolerance, check the number of initial/reweighted solves, and inspect
  component/support stability. Cover zero merit, decreases on either side of
  the relative threshold, and increases that must not trigger convergence.
  Cover increases below, at, and above $\eta F_k$, including acceptance and
  continued iteration for allowed increases. Check the relative tests under
  common positive rescaling of the entire merit function.
  For a trial merit increase exceeding numerical tolerance, verify rejection,
  termination without another reweighted solve, exact preservation of the
  preceding iterate, no weight update from rejected components, and reporting
  through selection/refitting. Distinguish the rejected convex solve's status from the
  status of the returned estimate.
  Verify one initialization and at most twenty reweighted attempts at the
  provisional default, with the fixed-support refit counted separately. Compare
  the proposed stopping/increase tolerances with tighter tolerances and larger
  caps on representative
  time-shift and capacity cases, measuring component/support changes, selected
  weights, cap frequency, and runtime before adopting defaults. Exercise cap
  termination and verify that the last
  successful iterate remains available for selection/refitting, with the cap
  reported through the final result. Distinguish that case from tolerance
  termination and a failed subproblem solve. Exercise a failed refinement after
  successful iterations and verify exact retention of the last accepted fit,
  no weight update from failed output, and persistence of the failure diagnostic
  through selection/refitting. An initial-fit failure must not expose values from
  an earlier candidate or run. Exercise mixed successful/failed weight paths,
  selection over the usable subset, no usable candidates, and an unusable
  required reference. Distinguish numerical search failure from a valid
  no-qualifying-candidate fallback and verify reported omitted weights.
  Successful convex solves do not establish global
  optimality of $F$ or physical correctness of the decomposition.
- **Reference:** [Candès, Wakin, and Boyd, Enhancing Sparsity by Reweighted l1 Minimization](https://web.stanford.edu/~boyd/papers/pdf/rwl1.pdf).

#### SD-05 — Retain and document model-specific normalization and weight semantics

- **Location:** `make_l1_pwc_smoothper_trend_problem`, approximately lines 249–287.
- **Finding:** the residual is averaged by $T$, TV by $T-1$, the Fourier penalty
  is a coefficient-space squared norm, and trend regularization is $w_4T\beta^2$.
  Their relative meaning depends on horizon and observation coverage. In the
  inspected `spcqe` 0.3.0, `make_regularization_matrix` scales its matrix linearly
  with `w3`; the subsequent squared norm therefore introduces `w3**2`.
- **Owner decision, 2026-10-21:** retain the existing model choices and document
  them explicitly; normalization changes and broader model updates are future
  work. For time shifts, retain $\sum_{t\in\Omega} r_t^2$. For capacity, retain
  $T^{-1}\sum_{t\in\Omega}|r_t|$, with $T$ the full grid length, not the number
  of admitted observations. Retain the existing structural-penalty scaling and
  effective weight meanings unless a separate change is explicitly agreed.
  These conventions are not inherently invalid; mask- or horizon-dependent
  effects must be documented rather than silently removed by rescaling.
- **Factoring boundary:** removing redundant variables or factoring a fixed
  regularization matrix is permissible as a proposal for algebraic simplification,
  but must preserve the objective. In particular, if the current matrix is
  $W=w_3R$, its cost is $w_3^2\|R\theta\|_2^2$; replacing the coefficient by
  $w_3$ would change the model. No common observation-count normalization or
  reinterpretation of public weights is adopted by this refactor.
  Replacing the `spcqe` Fourier helpers with native helpers under PKG-03 must
  preserve these matrix and weight semantics.
- **Level convention:** the PWC term carries the local-capacity offset; retain
  the seasonal basis without a DC term and the trend anchored at zero initially.
  A finite partial-record mean of the nonconstant Fourier terms need not vanish.
- **Proposed simplification:** represent seasonality as $B\theta$ and the linear
  trend as $\beta t$, avoiding redundant length-$T$ variables and defining
  equalities. Retain the agreed intercept and slope constraints.
- **Proposed verification:** algebraic equivalence for reductions, objective-level
  weight checks, and the agreed behavior across record lengths and masks.

#### SD-06 — Reusable native problem construction [design/performance proposal]

- **Locations:** `_cvx_signal_decompositions.py` builders and their callers in
  time shifts, capacity, clipping, sunrise/sunset, and loss-factor analysis;
  the native quantile fitting used by PVQuantiles and anomaly detection (PKG-03).
- **Proposed organization:** keep model formulas explicit and share only small
  helpers for masks, Fourier construction, solve validation, and IRL1. A workspace
  containing the CVXPY `Problem`, parameters, and expressions keyed by component
  role is sufficient; a new component hierarchy or solver-dispatch framework is
  not proposed. The owner permits a cleaner breaking API with a migration guide
  (SD-01). Add a compatibility wrapper only where the agreed interface design
  gives it a concrete purpose.
- **Solve-configuration contract, 2026-11-11:** use the explicit CLARABEL
  default and CVXPY pass-through in SD-01, rather than a solver-specific adapter
  framework. Preserve caller-supplied solve options through construction,
  parameter updates, candidate sweeps and extensions, IRL1, fixed-support
  refits, and sensitivity draws. Inner helpers must not overwrite those options
  or silently substitute another solver. Specify a single, unambiguous route
  for solver selection and keyword arguments, with clear handling of conflicting
  settings; do not discard unfamiliar solver-specific options via a narrow
  package allowlist. Keep the chosen solver and supplied settings available in
  diagnostics. Default solver settings are distinct from the package's
  post-solve acceptance checks and the IRL1 stopping rules. Unsupported solver
  configurations are configuration problems, not evidence about the fitted
  components; retain the existing distinction between invalid configuration and
  expected numerical failure. Coordinate forwarding with DASK-01 and resolved
  configuration preservation with LFA-03.
- **Result lifetime:** the lightweight public result in SD-01 represents the
  returned fit, separately from the reusable CVXPY workspace. Later parameter
  updates or solves must not alter an earlier result's component values or
  diagnostics. Preserve the accepted values needed for the agreed IRL1 and
  refit fallbacks without introducing a second modeling framework.
- **Solver-status decision, 2026-10-22:** accept an `optimal_inaccurate` result
  when its values are finite and numerical feasibility checks pass, retaining
  the original status and an explicit warning. Do not relabel it `optimal`.
  Apply common solve validation before using component values, updating IRL1
  weights, or including a sensitivity draw in statistics. Check the returned
  components and objective for finiteness and evaluate the model constraints
  against documented, appropriately scaled tolerances. A nonfinite value or a
  failed feasibility check makes the solve unusable even when its status is
  otherwise accepted; apply the failure policy for the affected stage
  (SD-03, SD-04, LFA-06, or LFA-10). Numerical tolerances remain to be specified
  and validated during implementation. Record status and check results, and
  carry the qualification into reporting for estimates that use an accepted
  inaccurate solution. Passing feasibility checks does not establish exact
  optimality. Keep convex-solver status distinct from IRL1 termination and
  fixed-support refit outcome.
- **Numerical failure propagation:** translate expected solver failures and
  rejected numerical results into the common failure result when no accepted
  fit remains. Preserve the attempted solver status, failure reason, and any
  available numerical diagnostics even when failure occurs before a status is
  returned. Weight searches and sensitivity sampling inspect the result and
  continue under their agreed omission/budget rules; they must not depend on
  user-configured exception handlers. Apply the existing retained-fit policies
  before declaring a fit unavailable. This policy concerns expected numerical
  failures; it does not require suppressing invalid configuration or programming
  errors with a blanket exception handler. Keep the helper small and share its
  result semantics with the pipeline integration in DH-07.
- **DPP contract:** build once for fixed dimensions and structural choices. For
  IRL1, compute $a_i=\lambda/(\epsilon+|(Dz^{\mathrm{prev}})_i|)$ outside CVXPY,
  store it in one nonnegative `Parameter`, and use `a @ abs(D @ z)`. Multiplying
  independently parameterized $\lambda$ and reciprocal weights inside CVXPY
  violates DPP. Weights outside the absolute value also permit fixed epigraph
  constraints with changing objective coefficients. Verify reusable formulations
  with `problem.is_dcp(dpp=True)`.
- **Data and mask updates:** use finite parameter values and fixed graph
  dimensions while preserving the full-grid component construction in SD-01.
  With an explicit full-grid residual variable, apply its predefined separable
  cost over the full grid and mask only the consistency equality. The uncoupled
  residual entries outside $\Omega$ then minimize their cost at zero. An
  equivalent reduced formulation may eliminate the residual analytically:
  with $m=\mathbf 1_\Omega$ and finite $y^{\mathrm{safe}}$ agreeing with the data
  on $\Omega$, use $r=m\odot(y^{\mathrm{safe}}-\widehat y)$ as the full-grid
  residual expression. It is zero outside $\Omega$ by construction and must be
  returned as such. Merely zero-weighting residual costs while leaving explicit
  off-mask residual variables free does not establish this contract. Retain the
  agreed model-specific loss normalization (SD-05). Do not pass NaNs as parameter
  values or treat finite placeholders as observations. Precompute products such
  as $m\odot y^{\mathrm{safe}}$, and combinations of masks and loss coefficients
  where needed, to avoid parameter–parameter products.
- **Reuse limits:** DPP caches canonicalization. Solver warm starts and numerical
  factorization reuse are separate, solver-dependent benefits; a changing mask
  can change matrices and prevent factorization reuse. Rebuild for structural or
  dimensional changes. Reset the initial IRL1 weights for each structural
  candidate unless continuation is explicitly part of the algorithm; otherwise
  the sweep order becomes an unrecorded modeling choice.
- **Source-derived opportunities:**

  | Operation | Current repeated work | Proposed reduction |
  | --- | --- | --- |
  | Time-shift search through the CVXPY branch | 11 candidates plus full-data fit, each with five builds/solves: 60 total; 65 with polish | Reuse a graph across candidates and IRL1 subproblems; use the agreed full observation mask for the final estimate and refit its selected support once |
  | Capacity search | 13 candidates plus selected refit: 14 builds and 28 solves | Reuse the graph and retain the selected IRL1 candidate; perform the newly agreed single fixed-support refit for the final estimate |
  | Ordinary sunrise/sunset threshold search | Up to 44 solves for 21 valid candidates plus final fits, without true-time diagnostics | Reuse the seasonal-quantile graph across data, mask, and quantile updates |
  | Loss-factor sampling | Model/dictionary reconstruction and attribution on every draw | Replace the soiling dictionary with the direct formulation, reuse fixed structures, and compute only required sampled quantities; see LFA-06/LFA-07 |

  Counts assume the indicated source branches complete. They are not timings or
  predictions of speedup. A future benchmark should distinguish construction,
  canonicalization, solver time, and extraction, after correctness is established.
- **Proposed verification:** updated-parameter results agree with freshly built
  instances of the same problem, and previously returned results remain unchanged
  after a subsequent solve. Discarded/stale results cannot survive a failed
  solve as if newly fitted. Distinguish the explicitly retained successful
  candidate iterate under SD-04 from residual solver state after failure.
  Cover `optimal_inaccurate` results that pass validation and retain their
  warning, as well as nonfinite or infeasible returned values that must be
  rejected. Verify that the accepted status and numerical diagnostics survive
  extraction and reporting without being replaced by an unqualified success.
  Coordinate with CLIP-01, SHADE-03, and QA-04.
- **References:** [CVXPY DPP](https://www.cvxpy.org/tutorial/dpp/index.html),
  [CVXPY warm starts](https://www.cvxpy.org/tutorial/solvers/index.html#warm-start).

## 3. Analysis algorithms

### `solardatatools/algorithms/capacity_change.py`

#### CAP-01 — Reported holdout error is training error [P2]

- **Owner scope clarification, 2026-10-21:** preserve the capacity selector's
  design. Correcting the reporting defect is a bounded issue and does not
  require replacing the selector.
- **Location:** `optimize_weight`, approximately lines 135–155.
- **Finding:** the train/test split is discarded and both masks are assigned the
  same observations. `holdout_error` consequently equals in-sample error. The
  actual selection rule uses in-sample fit and structural jump conditions.
- **Proposed correction:** name reported metrics according to their actual
  meaning and remove the unused random split while retaining the structural
  screening rule. Its use of in-sample fit as one screening condition is not
  itself a defect. Do not turn this terminology correction into selection by
  reconstruction holdout error.
- **Proposed verification:** inspect the masks and reported metric semantics; if
  validation is retained, verify disjointness and exclusion from the fit.

#### CAP-02 — Check the constant-component endpoint of the capacity weight grid [focused review]

- **Owner scope clarification, 2026-10-21:** the capacity detector is the
  intended Tier-3 archetype. Preserve its component-informed screening and
  selection of the smallest weight satisfying the criteria. Review concrete
  implementation or reporting defects without introducing a new selection
  framework or sweeping changes to the thresholds.
- **Location:** `optimize_weight`, approximately lines 165–181; label extraction
  around lines 99–103.
- **Existing design:** screen candidates using fit improvement, jump count,
  and minimum jump amplitude, then choose the smallest qualifying weight.
  Current thresholds are a fit-error ratio at most 0.98, strictly fewer than
  five detected jumps, and minimum detected jump magnitude at least 0.075 log
  units. The minimum jump magnitude is the principal structural criterion
  identified in the source comments. These component checks implement the
  intended Tier-3 approach; reconstruction error is one screening condition.
- **Corrected mathematical interpretation:** the TV penalty has a finite
  critical weight above which the capacity component is constant. An explicit
  constant-component constraint is not necessary to obtain this solution. The
  present implementation assumes that the chosen grid reaches this regime
  without calculating or checking the threshold. The owner expects its upper
  endpoint is already sufficient. No failure of that assumption has been
  demonstrated; this item is a bounded verification opportunity, not evidence
  that the selector or its capacity labels are incorrect.
- **Cheap certificate for a convex subproblem:** retain the current scaling

  $$
  \frac{1}{T}\sum_{t\in\Omega}|r_t|
  +\frac{w_2}{T-1}\sum_{i=0}^{T-2}a_i|(Dc)_i|
  +g(\theta,\beta),\qquad a_i>0.
  $$

  For a constant-capacity optimum, let $p$ be a residual-loss subgradient that
  satisfies the remaining stationarity conditions. Then $p_t=0$ off $\Omega$,
  $|p_t|\leq 1/T$, and $\mathbf 1^T p=0$ because the capacity level is free.
  The first-difference KKT condition gives the sufficient weight

  $$
  \overline w_2=(T-1)\max_{0\leq i<T-1}
  \frac{\left|\sum_{t=0}^{i}p_t\right|}{a_i}.
  $$

  It takes linear time to evaluate once a suitable dual certificate is available.
  With absolute residual loss, certificates can be nonunique, so an arbitrary
  certificate provides a sufficient threshold, not necessarily the smallest
  critical weight. A conservative bound requiring no solve follows by replacing
  the numerator with $\min(n_i,m-n_i)/T$, where $m=|\Omega|$ and $n_i$ counts
  admitted observations through index $i$. With full observation and $a_i=1$,
  this gives $(T-1)\lfloor T/2\rfloor/T$. Such a bound can be loose; failing to
  exceed it does not establish that the current grid endpoint is insufficient.
  These certificates concern fixed positive IRL1 weights; account for the
  actual initialization and weight updates when using them for the full routine.
- **Endpoint decision, 2026-11-11:** automatic capacity and time-shift weight
  searches must verify that the upper endpoint yields a constant step
  component. Retain the existing grid when it is sufficient; extend its upper
  range only if needed, with a bounded extension and the added weights reported.
  This establishes the reference for the 2% fit-improvement screen without
  changing the component screening criteria or smallest-qualifying-weight rule.
- **Proposed endpoint procedure:** inspect the accepted upper-endpoint fit
  using a documented, validated numerical constancy tolerance. Check the
  component's variation over the full model grid; absence of jumps above the
  physical screening threshold or equality of reported labels is not a
  constancy certificate. If the component is constant, reuse that fit and its
  diagnostics. An applicable dual certificate from that accepted solve may
  supplement the check without another fit; computing the exact critical
  weight is not required by this decision. Otherwise, evaluate larger weights
  within a specified extension budget and stop once a usable constant-component
  reference is established. The extension rule, budget, and numerical tolerance
  remain implementation specifications to validate. Each added weight follows
  the same IRL1 initialization/reset, acceptance, screening, and diagnostic
  rules as the original candidates. Report the original grid, added weights,
  reference weight, constancy check, and extension termination.
- **Reference-error contract:** use the residual error of the accepted
  constant-component reference, evaluated with the same observation mask and
  model settings as the candidates. The current capacity source uses
  `max(train_r)`; do not assume this equals the verified endpoint's error,
  particularly along a nonconvex IRL1 path. Bind the denominator to the verified
  reference explicitly. Preserve the existing residual metric and 2% criterion.
  For an accepted constant reference with zero residual error, use the
  constant-component fallback: no candidate can improve a nonnegative error
  below zero. Do not divide by zero. Make `plot_weight_optimization` use this
  same verified reference instead of independently normalizing by the largest
  recorded error; display the zero-error case without a misleading ratio.
  The comment's "no more than five" wording should also be reconciled with the
  retained implemented strict jump-count bound.
- **Numerical search failures:** apply SD-04's decision to continue over usable
  candidates and report failed weights. Verify that the intended reference and
  fallback estimate are usable; omitting a failed endpoint must not silently
  substitute a fit with a different reference meaning or imply no capacity
  changes. A failed numerical solve does not establish that its weight is too
  small. If bounded extension ends without a usable constant reference, return
  an unsuccessful selection result with that reason and retain available
  candidate diagnostics. Do not report the no-change fallback in this case.
  This is failure handling for the existing selection design.
- **Proposed verification:** compare the certificate with the onset of constant
  capacity on representative cases, using the actual mask, normalization, and
  IRL1 weights. Distinguish mathematical threshold conditions from numerical
  solver tolerances. Verify that a sufficient existing endpoint causes no
  extension, an insufficient one triggers bounded extension, added weights
  obey the normal candidate rules, and reference error comes from the verified
  constant fit even when it is not the largest recorded residual error. Cover
  small differences with material cumulative drift, failed endpoint solves,
  and exhausted extension budgets without a false no-change finding. This
  derivation is algebraic; no solves have been run.
  Check zero reference error and agreement between selection diagnostics and
  the public weight plot.
- **Support/label boundary:** SD-03 now fixes capacity refit support at final
  IRL1 jumps exceeding 0.001 log units, with labels formed after refitting.
  If the refit fails, form labels from the returned IRL1 component and preserve
  the explicit failed-refit diagnostic, as specified in SD-03.
  Specify candidate selection separately from this numerical support extraction.
- **Related:** SD-04 owns the agreed IRL1 changes, SD-05 preserves objective
  normalization, and SD-06 owns reuse of the selected full-data candidate.
  These accepted changes remain separate from the preserved selection design.

### `solardatatools/algorithms/clear_sky_detection.py`

#### CLEAR-01 — Dynamic programming does not recover a minimum-cost path [P1]

- **Location:** `ClearSkyDetection.find_optimal_path`, approximately lines 84–104.
- **Finding:** prefix costs are accumulated, but the returned labels are their
  independent columnwise minimizers; there is no traceback from the terminal
  minimum. With `sig=[0,1,1]`, unit quantile estimates, and default transition
  costs, the emission losses are

  $$
  L=\begin{bmatrix}0&1&1\\1&0&0\end{bmatrix}.
  $$

  The returned sequence $(0,0,1)$ costs $5$, while $(1,1,1)$ costs $1$.
- **Proposed correction:** retain predecessor states and backtrack from a terminal
  minimizer. Specify tie behavior.
- **Public-wrapper finding, 2026-11-11:** `DataHandler.detect_clear_sky`
  (around lines 2451–2468) optimizes first and then overwrites nighttime states
  with zero. In the example above, making the first entry nighttime changes
  the optimal unconstrained path to $(0,1,1)$, costing $4$, although the
  forced-night feasible path $(0,0,0)$ costs $2$. Traceback alone does not fix
  the public result.
- **Owner decision, 2026-11-11:** optimize each contiguous daylight segment
  independently. Exclude nighttime from the transition model and report it as
  non-clear. Do not impose transitions across night or enter/leave penalties
  to an artificial nighttime state. Each daylight segment has its own terminal
  minimization and traceback. This supersedes continuous forced-night
  optimization as a proposed alternative; post-hoc editing of a continuous
  unconstrained optimum is not the selected objective.
- **Proposed verification:** compare path cost with exhaustive enumeration for
  short binary sequences, including ties and asymmetric transition costs.
  Exercise `DataHandler.detect_clear_sky` with multiple daylight segments,
  one-sample segments, and no daylight. Verify independent optimal paths,
  nighttime outputs, and independence from changes in a different segment.

### `solardatatools/algorithms/clipping.py`

#### CLIP-01 — Reusing the detector reuses a stale fitted CDF [P2]

- **Location:** `pointmass_detection`, approximately lines 138–144.
- **Finding:** later calls replace `self.y_param` and `self.weight` but do not
  recompute `self.y_hat`. These assignments do not update and solve an optimization
  problem. Detection can therefore use the old fit while exposing a new empirical
  CDF.
- **Proposed correction:** retain actual data/weight parameters and the CVXPY
  problem, update them, solve, and replace the extracted CDF. Rebuild when the
  problem dimensions or structural choices change. SD-06 covers the workspace;
  SD-01 covers the public weights currently dropped by the MOSEK route.
- **Proposed verification:** compare a reused detector on a second dataset/weight
  with a fresh detector on that same dataset/weight.

### `solardatatools/algorithms/dilation.py`

#### DIL-01 — Heatmap unit conversion mutates stored signals [P1]

- **Location:** `Dilation.plot_heatmap`, approximately lines 59–71.
- **Finding:** reshaped signal views are divided in place by 1000. Repeated plots
  repeatedly scale the stored data; depending on layout, storage can also be
  shared with the handler matrix.
- **Proposed correction:** copy before conversion or use an out-of-place
  expression. Coordinate the plotting-purity contract with DH-01.
- **Proposed verification:** repeated plots in both original and dilated spaces;
  assert unchanged source arrays and unchanged later computations.

### `solardatatools/algorithms/loss_factor_analysis.py`

#### LFA-01 — Annual-rate input and output maps are inconsistent [P2]

- **Locations:** fixed-rate construction near line 516; rate extraction near
  lines 204–207; final fit after Monte Carlo estimation.
- **Finding:** construction sets daily log slope
  $\beta=r/(100\cdot365)$, but extraction returns

  $$
  \widehat r=100\bigl(1-\exp(-365\beta)\bigr).
  $$

  Thus `deg_rate=-10` returns approximately $-10.517\%$/yr. For the linear model,
  the Monte Carlo median is also transformed when imposed in the final fit.
- **Scope distinction:** correcting this annual-rate conversion is separate from
  replacing the current median of overlapping annual changes with an endpoint
  chord. They agree for a linear log trend after consistent annualization, but
  generally differ for a nonlinear trend. LFA-08 owns the agreed estimand
  change; LFA-09 owns its interaction with the nonlinear tail and fixed-rate API.
- **Conversion contract and annualization decision, 2026-10-22:** use
  $P=365.2425$ days per year consistently. With $\beta$ the log slope per day,
  use the exact inverse maps

  $$
  \beta=\frac{\operatorname{log1p}(r/100)}{P},\qquad
  r=100\operatorname{expm1}(P\beta).
  $$

  Require finite $r>-100\%$, with negative values denoting decline. This year
  length matches the seasonal model and the inspected RdTools log-rate
  extractor. For sample spacing $\Delta$ days, a coefficient expressed per
  sample is $\Delta\beta$. Use the same documented constant in fixed-rate
  construction, overall-chord extraction, the median annual-change comparison,
  sampling summaries, and reporting.
- **Proposed verification:** fixed-rate round trips, a known exponential trend,
  flat and near-zero rates, and invalid-domain inputs. Verify that the chord and
  annualized 365-day-lag comparison agree for a linear log trend using the same
  $P=365.2425$. Preservation of a selected
  rate by the existing fixed-rate mechanism is a requirement for
  `deg_type="linear"`. Under the decision in LFA-09, nonlinear models reject
  fixed-rate input; verify that rejection and the freely fitted nonlinear path.

#### LFA-02 — Sensitivity percentiles are labeled as confidence bounds [P2]

- **Locations:** sampler near lines 135–140; plot near lines 390–392; report text
  in `data_handler.py`.
- **Finding:** the sampler varies quantile level, soiling stiffness, and retained
  observations. Its empirical percentiles describe sensitivity under those
  perturbations. Stabilization of the percentiles does not establish nominal
  confidence coverage.
- **Proposed correction:** label the output as a central 95% parameter/subsample
  sensitivity interval and document the perturbation distribution. Separate
  structural sensitivity analysis from uncertainty analysis conditional on a
  selected model, retaining component-level outputs for inspection.
  **Owner decision, 2026-10-22:** replace random soiling-stiffness perturbations
  with the RdTools component-based selection and post-selection refinement
  workflow (LFA-06), validated under SDT's retained quantile loss. Preserve SDT's
  quantile configuration and observation subsampling. Select the soiling model
  and weight once at the configured quantile and base observation mask, and hold
  that selection fixed across draws. Report the resulting sensitivity as
  conditional on that selection; it does not include variability from repeating
  the selection procedure.
  Changing rate extraction to a chord does not turn these percentiles
  into calibrated confidence bounds. A confidence claim requires a separately
  justified and calibrated inferential procedure; developing that procedure is
  a separate scope decision. RdTools alignment does not by itself adopt its
  resampling procedure or establish coverage for this model.
  Under LFA-04's headline-rate decision, report these percentiles with the
  separately named sensitivity median; do not relabel them as confidence bounds
  for the final-fit headline rate, or shift them to be centered on that rate.
  Update the ambiguous legacy `degradation_rate_lb`/`degradation_rate_ub`
  presentation and document the replacement sensitivity-result access.
- **Proposed verification:** consistent terminology in reports, plots, docstrings,
  and user guidance. Do not treat a wording correction as validation of coverage.

#### LFA-03 — Monte Carlo reconstruction loses resolved configuration [P2]

- **Locations:** `__init__`, approximately line 59; sampling problem construction,
  approximately lines 146–156.
- **Finding:** `user_settings` contains only supplied `kwargs`, but sampling
  indexes defaulted names unconditionally. Direct use
  `LossFactorAnalysis(y).estimate_degradation_rate()` raises a missing-key error.
  The `DataHandler` route supplies these defaults and avoids that specific failure.
  Sampling also drops an explicitly supplied `weight_deg_nonlinear`.
- **Proposed correction:** retain a complete resolved configuration and override
  only the intended sampled parameters. Ensure temporary observation-mask changes
  are restored on failure. Preserve SDT's quantile loss configuration throughout
  soiling selection, refinement, sampling, and final fitting; keep sampled
  quantile levels distinct from the configured final-fit level. Track the
  selected soiling model/weight explicitly rather than inheriting stale state
  or continuing the removed random stiffness draws (LFA-06/LFA-07).
  Reset adaptive interval weights for each draw before its uniform-L1 fit and
  two refinements; numerical warm starts must not carry another draw's penalty
  weights into this estimator.
- **Proposed verification:** direct default construction, explicit nonlinear
  settings, and failure-path restoration of the original mask/configuration.
- **Related:** LFA-10 defines sampling-budget validation and termination results;
  retain that resolved configuration with the model settings.

#### LFA-04 — Repeated uncertainty estimation fixes the previous rate [P2]

- **Location:** implicit Monte Carlo rate inheritance in `make_problem`,
  approximately lines 483–486.
- **Finding:** a second estimation call inherits the previous Monte Carlo rate
  during each sample, fixing linear degradation rather than estimating it. The
  rate-mapping discrepancy in LFA-01 can compound the effect.
- **Final-linear-fit decision, 2026-11-11:** after sensitivity sampling
  stabilizes, or exhausts its budget with at least twenty successful draws,
  when no user-supplied fixed rate takes precedence, fix the final full-data
  linear degradation rate to the median of that run's sampled endpoint-chord
  rates. For successful sampled rates $r^{(j)}$, set
  $$
  r_{\mathrm{final}}=\operatorname{median}_j r^{(j)},\qquad
  \beta_{\mathrm{final}}=
  \frac{\operatorname{log1p}(r_{\mathrm{final}}/100)}{365.2425}.
  $$
  Apply this constraint explicitly in the final-fit phase, using the base
  observation mask and configured final-fit quantile. Each sensitivity draw
  estimates its linear rate freely; neither this run's summary nor a previous
  run's fitted rate may constrain those draws. Retain the selected soiling
  model/weight and final refinement workflow from LFA-06. Keep the sampled
  summary separately recorded even though the final linear rate is constrained
  to agree with it. Nonlinear final fits remain free under their agreed model
  constraints and do not receive the sampled median as fixed-rate input.
  Insufficient diagnostic samples under LFA-10 must not be silently promoted
  to a completed sensitivity summary by this rule.
- **Budget-exhaustion decision, 2026-11-11:** with at least twenty successful
  draws, use the sampled median for the final linear fit even if sampling reaches
  its budget before the statistics stabilize. Retain the computed sensitivity
  percentiles and explicitly report budget exhaustion and lack of stabilization.
  Carry that qualification into the final fitted-rate and loss reports; a
  successful constrained final fit does not establish stabilization of the
  sampled summary. Do not exceed the budget or change to a freely estimated
  final rate solely because stabilization was not reached.
- **Incomplete-sensitivity decision, 2026-11-11:** if fewer than twenty
  successful draws are available and no fixed rate was supplied, attempt the
  final full-data fit with a freely estimated linear degradation rate. Retain
  the incomplete sensitivity status, successful/failed attempt counts, and
  collected diagnostic samples. Do not constrain the fit to their provisional
  median or inherit a rate from an earlier run. Restore the base observation
  mask, configured quantile, and selected soiling model/weight as for the usual
  final fit. This applies also when no sensitivity draw succeeds, provided the
  required base configuration and model selection are available; missing
  prerequisites still follow DH-07's stage-continuation contract.
  A successful final fit may supply loss attribution and its fitted rate,
  explicitly identified as freely estimated after incomplete sensitivity.
  That success does not complete the sensitivity analysis or create its normal
  summary/bounds. If the final fit also fails, retain both stages' diagnostics
  and the samples already collected, with final losses unavailable.
- **Resolved linear final-fit outcomes, absent a supplied fixed rate:**

  | Sensitivity outcome | Final linear rate | Sensitivity reporting |
  | --- | --- | --- |
  | At least twenty successful draws; stabilized | Fixed to sampled median | Summary with stabilization recorded |
  | At least twenty successful draws; budget exhausted before stabilization | Fixed to sampled median | Median and percentiles retained; explicit budget-exhausted/not-stabilized warning |
  | Fewer than twenty successful draws | Freely estimated on the full data | Incomplete diagnostic result; no normal sensitivity summary |

  These outcomes preserve the separate final-fit success/failure status and
  LFA-02's sensitivity interpretation; neither sample count nor stabilization
  establishes confidence coverage.
- **Headline-rate decision, 2026-11-11:** the headline `degradation_rate` is
  the accepted final fit's overall endpoint-chord rate under LFA-08, including
  its agreed reporting interval, annualization, and nonlinear cumulative-minimum
  reporting transform. Report the median across sensitivity draws and its
  percentiles separately. Also keep the final fit's median annual-change
  comparison separately named; it is not the sensitivity median. For nonlinear
  fits, these quantities can differ and must not overwrite each other. For a
  linear final fit constrained to the sampled median, verify their agreement
  without collapsing the separate results or their provenance. If no accepted
  final fit is available, mark the headline unavailable and retain any available
  sensitivity summary in its own result; do not silently substitute that summary
  or an earlier run's headline. Apply this reporting contract to the analysis
  result, DataHandler summaries, plots, and migration guidance.
- **Proposed correction:** separate the sampling/estimation phase from the final
  fit. Pass the agreed final linear rate explicitly rather than inferring it
  from mutable object history in the problem builder. Only the linear model
  may impose a chosen fixed rate; the nonlinear final fit remains free under
  its model constraints, as decided in LFA-09. Remove implicit inheritance of
  a Monte Carlo summary as a nonlinear `deg_rate` input.
  Keep the Monte Carlo rate summary and the final
  fitted rate as distinct results so one cannot silently overwrite the other.
  Record whether the final rate was constrained or freely estimated. Document
  repeated-call semantics explicitly and map any retained `degradation_rate`
  attribute to the final-fit headline defined above. Distinguish the median across
  Monte Carlo chord estimates from the within-fit median annual-change
  comparison retained in LFA-08.
- **Proposed verification:** compare repeated estimation with a fresh instance
  under controlled random inputs; inspect which components are fixed in each
  phase. Verify that successful linear draws estimate rates freely, the final
  linear fit uses the median of the current run's sampled annual rates with the
  exact inverse conversion, and extraction recovers that imposed rate to
  numerical accuracy. Check explicit caller-rate precedence, restored base
  masks and quantile settings, and rejection of an implicit summary constraint
  in the nonlinear path. Final loss estimation must preserve the recorded
  Monte Carlo summary and expose any distinct final fitted rate; unsuccessful
  final fitting must not erase a valid sampled summary. For zero, one, and
  nineteen successful draws, verify a free-rate final-fit attempt, restoration
  of base settings, and persistent incomplete-sensitivity reporting whether
  that final fit succeeds or fails. For at least twenty successful draws, cover
  both stabilization and budget exhaustion without stabilization; both use the
  sampled median, and the latter retains its warning and sensitivity percentiles
  through final fitting and reporting. Verify that the final fit
  does not become another sensitivity sample or change the recorded draw counts.
  Include a nonlinear case with distinct final-fit chord, sensitivity median,
  and within-fit median annual change; verify that all are retained with their
  correct labels and that the headline is the final-fit chord. Verify an
  unavailable headline after final-fit failure without loss of a usable
  sensitivity summary or reuse of a stale headline.

#### LFA-05 — Input dtypes, observation masks, and label indexing are inconsistent [P2]

- **Locations:** constructor around lines 47–56; energy totals around line 209;
  capacity-label basis around lines 537–540; `model_wrapper` around lines
  560–574 and `attribute_losses` around lines 620–653.
- **Findings:**
  - Integer energy is preserved in `log_energy`, which cannot safely represent
    NaNs or real logarithms.
  - Missing energy passes `~is_zero` and enters totals, invalidating attribution.
  - Documented integer outage flags are complemented before Boolean conversion:
    `0/1` becomes `-1/-2`, both truthy, so exclusion is ineffective.
  - The same `use_ixs` mask controls fitting, energy totals, and the binary outage
    factor. Consequently, finite positive energy on a flagged day is dropped
    from both totals and the final Shapley-model production.
  - Integer capacity labels are used directly as basis-column indices despite
    allocation by the number of unique labels; labels such as `{1,2}` fail.
  - `DataHandler.make_filled_data_matrix` integrates filled power into
    `daily_signals.energy`, which `run_loss_factor_analysis` passes to LFA.
    That vector alone does not identify measured production, filled values,
    or observation coverage. The whole-day gap policy cannot be enforced
    reliably without this provenance at the public boundary.
  - The absolute near-zero test `np.isclose(energy_data, 0, atol=0.1)` changes
    admission under a unit conversion: 50 Wh is admitted while the equivalent
    0.05 kWh is classified as zero. This is separate from the dtype defect.
- **Partial-day accounting decision, 2026-11-11:** accounted daily production
  is the integral of reported finite power samples, with missing sampling
  intervals contributing zero. Keep energy integrated from filled power as a
  distinct, labeled estimate. Preserve the original missingness and coverage;
  assigning zero for accounting does not turn a missing sample into a measured
  zero. Use the declared interval/integration weights and units consistently
  through clock correction and daily aggregation. This extends the existing
  gap-as-zero convention to partially observed days.
- **Fitting-input decision, 2026-11-11:** retain energy integrated from filled
  power for fitting on days admitted by the existing quality/outage flags.
  Represent this fitting vector separately from observed-only accounted energy,
  and retain observation coverage. Do not introduce artificial downward fit
  deviations by assigning zero to missing intervals in an otherwise admitted
  day. Flagged positive days remain excluded from fitting but included in
  accounting as previously agreed.
- **Partial-day factor decision, 2026-11-11:** on an admitted partial day with
  positive filled energy, let $E_t^{\mathrm{obs}}$ be observed-only accounted
  energy, $E_t^{\mathrm{fill}}>0$ the filled fitting energy, and $q_t>0$ fitted
  production before weather and outages. Use

  $$
  o_t=\frac{E_t^{\mathrm{obs}}}{E_t^{\mathrm{fill}}},\qquad
  w_t=\frac{E_t^{\mathrm{fill}}}{q_t}.
  $$

  Thus $q_to_tw_t=E_t^{\mathrm{obs}}$. Filling must preserve reported
  nonnegative samples and add only nonnegative production at missing entries,
  giving $0\leq E_t^{\mathrm{obs}}\leq E_t^{\mathrm{fill}}$ and
  $0\leq o_t\leq1$. The deficit assigned to missing intervals enters the
  outage factor; the filled-energy discrepancy remains in the weather/residual
  factor. This is the agreed accounting convention, not evidence that the
  missing production was physically zero. Flagged days retain their separate
  bounded-outage rule below, and fully missing days retain $o_t=0$, $w_t=1$.
- **Measured-zero decision, 2026-11-11:** exclude a genuinely measured-zero
  day from the log fit and use $o_t=0$, $w_t=1$, even without an outage flag.
  Preserve its measured-zero status separately from missing measurements.
  This makes the zero-production treatment explicit without inferring that
  absent measurements are observed zeros.
- **Admission decision, 2026-11-11:** remove the unit-dependent `0.1`
  near-zero cutoff. Admit finite, strictly positive fitting energy subject to
  the existing quality/outage flags; exclude exact zeros from the log fit.
  Do not introduce another low-energy threshold in this correction. An
  additional physical threshold would require a separate model specification.
  Positive reported production remains in accounting regardless of fit
  admission, preserving the flagged-positive-energy rule.
- **Owner accounting clarification, 2026-10-22:** preserve the convention of
  treating missing-energy gaps as energy not produced relative to the modeled
  baseline and including those gaps in the outage factor for Shapley attribution.
  Keep the baseline on those dates; use zero accounted production there. This
  is a third convention, distinct from omitting missing dates from accounting or
  substituting model-imputed production. It does not supply zero-valued
  observations to the log-domain fit.
- **Flagged-production decision, 2026-10-22:** retain finite positive reported
  energy in totals even when `outage_flags=True`, while excluding that day from
  fitting. The zero-production accounting convention applies to missing energy;
  an outage flag does not override an available positive energy measurement.
  Preserve the measured value and the flag as separate inputs.
- **Attribution contract:** encode each missing date with outage factor
  $o_t=0$ when that factor is active in the Shapley model. Retain the existing
  allocation of interactions among factors; do not add the gap's entire baseline
  deficit to the outage attribution after the Shapley calculation. Distinguish
  missing measurements from measured zeros in the input masks and document that
  the outage bucket includes unobserved production assigned zero by convention.
  This accounting assumption does not establish that a physical outage occurred.
  The degradation-rate reporting span in LFA-08 is unchanged.
- **Flagged-day factor decision, 2026-10-22:** adopt a bounded outage factor
  with excess production assigned to the weather/residual factor. For a flagged
  day with finite reported energy $E_t\geq0$, let $q_t>0$ be fitted production
  before weather and outages, and define the factors below. At the handler
  boundary, $E_t$ means the observed-only accounted energy specified above,
  not the filled fitting vector.

  $$
  u_t=E_t/q_t,\qquad o_t=\min\{1,u_t\},\qquad
  w_t=\max\{1,u_t\}.
  $$

  Thus $q_to_tw_t=E_t$: shortfalls enter the outage factor, and production above
  $q_t$ enters the weather/residual factor. For missing energy use $o_t=0$ and
  $w_t=1$, retaining the modeled baseline. This supersedes the proposed unbounded
  outage ratio with neutral weather. Clipping that ratio while keeping weather
  neutral would fail to reproduce reported energy.
  Construct the accounting factors from the final fit and hold them fixed
  across Shapley coalitions. With nonnegative multiplicative factors and
  $0\leq o_t\leq1$, the outage marginal contributions are nonpositive under the
  existing signed-energy convention; excess production cannot create an outage
  gain. Preserve the usual allocation of interactions and Shapley additivity.
- **Interpretation:** this is a post-fit accounting convention. Flagged days
  remain excluded from fitting. Given $q_t$, reported production identifies only
  the product $o_tw_t$; assigning all flagged-day shortfall to the outage factor
  does not separate physical outage losses from unobserved weather effects.
  The weather/residual factor retains its residual interpretation.
- **Proposed correction:** normalize energy to floating point, define the finite
  positive observation domain, convert outage flags to Boolean, and map labels to
  contiguous internal indices. Keep the log-fit observation mask, missing-data
  mask, and outage-accounting factor explicit. Missing measurements are excluded
  from fitting but receive the owner's zero-production accounting treatment;
  preserve their missing status in the source data. Temporary subsampling
  exclusions must not become outages. Include positive reported energy on
  outage-flagged dates in the accounting domain without admitting those dates
  to the fit, using the agreed bounded outage and residual-excess construction.
- **Physical input domain, owner clarification, 2026-10-22:** negative daily
  production is nonphysical for the PV system considered here and indicates a
  serious data-validity problem addressed by upstream SDT checks and processing.
  It is not an unresolved loss-accounting choice. Withdraw the proposed option
  to reinterpret negative production as missing energy or an outage. Any check
  at the direct LFA API boundary should expose a violation of this input domain;
  it does not introduce a new accounting convention or replace upstream checks.
- **Proposed verification:** integer energy, NaN gaps, measured zeros,
  integer flags, and noncontiguous labels; check totals and attribution identities.
  Verify that gaps retain their modeled baseline, contribute zero accounted
  production through the outage factor, and never enter logarithms or contaminate
  totals with NaNs. Check Shapley additivity including factor interactions, and
  independence of outage accounting from temporary fitting-mask perturbations.
  Verify that positive energy on flagged dates is retained in totals, excluded
  from fitting, and reproduced by the combined attribution factors. Include
  flagged-day cases $E_t=0$, $0<E_t<q_t$, $E_t=q_t$, and $E_t>q_t$; verify the
  factor bounds, neutral weather on missing dates, and nonpositive outage
  attribution under the signed-energy convention. The sum of attributions must
  match the resulting accounted-energy change relative to baseline, including
  cases with excess production.
  Trace finite observations, filled values, missing intervals, and coverage
  from a public `DataHandler` call to the fit mask and accounting factors.
  Include partial days, entirely missing days, unflagged measured zeros, and
  flagged positive days. Equivalent W/Wh and kW/kWh inputs must yield the same
  admitted mask and dimensionless factors, with energy totals and attributions
  related by the declared unit conversion.
  Include positive values on both sides of the retired `0.1` cutoff in each
  unit system, together with exact zero, to verify positivity-based admission
  without accidental reintroduction of an absolute numerical threshold.
  For admitted partial days verify both energy vectors, the filling invariant,
  bounded outage factors, and exact reconstruction of observed-only energy.
  Temporary sensitivity exclusions must not change measured coverage or be
  reclassified as outages. The residual factor still depends on the fitted $q_t$.

#### LFA-06 — Replace OSD with native CVXPY and the direct RdTools soiling formulation [model/refactoring decision]

- **Locations:** imports at lines 21–23; `make_problem`, approximately lines
  447–552; custom `SetEqual` at lines 743–756; extraction around lines 199–207.
- **Finding:** this module uses `gfosd` directly, independently of the public
  decomposition dispatch. Removing `_osd_signal_decompositions.py` would leave
  its `Problem`, component classes, and custom `GraphComponent` in use.
- **Supported-interface decision, 2026-11-11:** retire the existing standalone
  soiling algorithm and direct users to the main loss-factor analysis. Withdraw
  the earlier requirement to retain a thin standalone soiling interface. Keep
  soiling model construction,
  selection/refinement, and extraction in the native LFA implementation, without
  a second estimator or separately maintained standalone defaults. SOIL-01
  through SOIL-04 record the agreed retirement and migration direction for the legacy
  API. LFA's configurable quantile loss and default $\tau=0.9$ remain unchanged;
  the question about a new standalone quantile default is withdrawn.
- **Short-record decision, 2026-11-11:** for records shorter than one year,
  warn and attempt the explicitly configured LFA model. Retain warning-and-
  continuation rather than imposing a new one-year prerequisite. Record the
  applicability warning in the returned diagnostics and reports, independently
  of verbose printing. Do not silently remove seasonality, degradation, or other
  configured components based on record length; in particular, do not import
  the retiring standalone algorithm's automatic seasonal-removal branch.
  Ordinary input/configuration requirements and solve-acceptance checks still
  apply. A successful short-record fit does not establish reliable separation
  of its components. Validate usable record lengths and observation coverage
  under the configured model; one year is a warning boundary, not an empirical
  guarantee of identifiability.
- **Dask caller correction, follow-up review:** `sdt_dask/dask_tool/runner.py`
  (around line 114) skips loss analysis whenever `num_days <= 365`, independently
  of LFA's behavior. Remove this duration-only gate so the agreed configured
  fit is attempted, with the applicability warning propagated to the per-key
  report. A genuine missing prerequisite may still cause a recorded skip;
  record length alone is not such a prerequisite. Coordinate DH-07 so warnings,
  attempted-fit failures, and available partial metrics remain distinguishable.
- **Proposed correction:** formulate the residual quantile loss, Fourier baseline,
  degradation, capacity-label effects, and constrained soiling term directly in
  CVXPY. Use expressions keyed by role rather than reversing a positional array.
  Express linear degradation as $\beta t$; fixed components can be constants or
  parameters instead of custom graph classes. Apply the conversion contract
  from LFA-01, the selected estimand from LFA-08, and the nonlinear/fixed-rate
  decisions from LFA-09 consistently during estimation and final fitting.
- **Quantile-loss requirement, 2026-10-22:** SDT retains its quantile loss and
  configuration. Preserve the configured `tau` (currently defaulting to 0.9),
  the existing loss normalization, and the current Monte Carlo quantile-level
  perturbation (uniform on 0.85–0.95), unless separately revised. The soiling
  workflow must operate within that SDT formulation; validate or recalibrate
  transferred regularization weights and screening thresholds under this loss.
  RdTools' default residual-loss configuration is not adopted.
- **Soiling-model decision, 2026-10-22:** adopt the direct, lower-memory RdTools
  convention in loss-factor analysis, replacing the sawtooth dictionary with a
  nonpositive log-soiling signal $s\in\mathbf R^N$. With $D$ the first-difference
  operator and $(u)_-=\max(-u,0)$ elementwise, the base convex formulation is

  $$
  s\leq0,\qquad
  \phi_s(s)=\lambda_d\|(Ds)_-\|_2+\lambda_v\|s\|_1,
  \qquad \lambda_d,\lambda_v\geq0.
  $$

  The downward-increment penalty is an unsquared Euclidean norm. Construct the
  signal, differences, and penalties directly in CVXPY, eliminating the dense
  sawtooth-dictionary construction and its basis coefficients. This is an
  accepted model change requiring appropriate testing, not an algebraically
  equivalent port of the dictionary prior. Retain the agreed residual and other
  component specifications; map or recalibrate weights against those objective
  normalizations rather than copying old dictionary-stiffness values.
- **Soiling-workflow decision, 2026-10-22:** adopt RdTools' component-based
  weight selection followed by two cleaning-interval IRL1 refinement solves,
  with validation in the SDT loss-factor model. This replaces random
  soiling-stiffness perturbations. Apply the following sequence:

  1. Solve the uniform-L1 regularization path and evaluate soiling-component
     materiality, recovery frequency, and agreement with neighboring weights.
  2. Select the smallest qualifying weight among usable, evaluated candidates.
     If an adequately evaluated path has no qualifying candidate, fit the
     structural no-soiling model ($s=0$). Apply the numerical-search distinction
     below before interpreting a missing qualification as no soiling.
  3. For a selected soiling model, keep detection and the downward-increment
     weight fixed and perform two convex refinements of the magnitude penalty.
     Recompute cleaning intervals and interval weights from the preceding
     soiling estimate for each refinement. No-soiling results skip refinement.

- **RdTools reference and validation baseline:** the inspected builder
  in `../rdtools/rdtools/signal_decomposition.py` (around lines 219–243, revision
  `d414a758ea85ae60482a931eda3efc188c3980f3`) uses this base penalty, with uniform
  magnitude weights initially. `_solve_soiling_path`, `_soiling_candidate_metrics`,
  `_soiling_interval_value_weights`, and `_refine_soiling_irl1` specify the
  reference workflow (around lines 375–568). The reference uses nine log-spaced
  downward weights from $10^{-3.5}$ to $10^{-1.5}$ and magnitude coefficient
  $10^{-5.8}$. Interior candidates require 75th-percentile log-soiling depth at
  least 0.0075, at most 30 recoveries/year, neighboring-path normalized RMS
  differences at most 0.75, and centered correlations at least 0.90. Recovery
  boundaries are upward log increments of at least 0.005. These values provide
  the starting reference for LFA validation; their adequacy under SDT's quantile
  loss is not established by validation in RdTools.
- **Numerical-search distinction, follow-up review:** qualification uses
  neighboring fits. A failed candidate or neighbor can make a screen
  unavailable rather than false. Continue with usable candidates and record
  failed weights and unavailable comparisons. A no-soiling conclusion requires
  enough valid screening evidence to reject the required candidate set;
  unavailable comparisons are not rejections. Specify this completeness check
  before implementation. If the path establishes neither a qualifying choice
  nor a valid no-soiling outcome, return a recorded selection failure and apply
  the normal dependency/continuation rules, rather than silently fixing $s=0$.
  This addresses numerical failures, not a new statistical abstention policy
  or a redesign of the agreed component-based selector.
- **Refinement contract:** for each inferred cleaning interval $\mathcal I$,
  use magnitude weights

  $$
  a_i^{(k)}=
  \frac{\epsilon}{Q_{0.75}(|s^{(k)}_{\mathcal I}|)+\epsilon},
  \qquad i\in\mathcal I,
  $$

  with RdTools' reference $\epsilon=0.01$ in log units. Replace only the magnitude
  term by $\lambda_v\sum_i a_i^{(k)}|s_i|$ for each of the two refinement solves.
  Keep the quantile loss and its level for that fit, other components, constraints,
  and selected downward weight fixed. Record completed refinements and any
  failures; retain the entire last accepted component snapshot, weights, and
  diagnostics if a refinement fails. The inspected RdTools recovery resets
  weights and solves again; do not make SDT's fallback depend on that additional
  solve succeeding. Snapshot accepted values before each attempt and never
  combine failed-solve values with components from an earlier fit. These are
  fixed post-selection refinement
  steps, distinct from the convergence-based step-selection IRL1 in SD-04.
- **Sampling decision, 2026-10-22:** perform soiling selection once per
  estimation run, at SDT's configured quantile and base observation mask. Hold
  the selected soiling/no-soiling model and regularization coefficients fixed
  across all quantile/subsample sensitivity draws. For each draw with soiling,
  start from a uniform-L1 fit at the selected coefficients, then perform the
  two interval-weighted refinements using that draw's preceding fitted signal.
  Adaptive interval weights are recomputed within each draw; they are not the
  fixed selected regularization coefficient. A selected no-soiling model stays
  at $s=0$ in every draw and skips refinement. Select afresh for an independent
  estimation run; do not reuse a previous run's selection implicitly. Preserve
  SDT's quantile/subsample procedure and the reporting interval in LFA-08.
- **Proposed verification:** compare the assembled native formulation with its
  agreed mathematical specification, component roles, observation domain, and
  rate-extraction contract. Exercise known soiling accumulation and recovery,
  no-soiling cases, missing observations, and confounding with degradation or
  capacity changes; inspect component recovery and the reported loss attribution.
  Verify selection on uniform-L1 candidates, no-soiling fallback, the interval
  weight formula, exactly two successful refinement solves when neither fails,
  failure reporting, and preservation of SDT's quantile settings. Validate the
  path range and screening thresholds under SDT's loss and normalization.
  Inject failed candidates and neighbors to distinguish unavailable screening
  from an evaluated no-soiling outcome. Exercise refinement failure and verify
  exact retention of all accepted components without an extra recovery solve.
  Verify one selection path per estimation run, fixed model membership and
  selected coefficients across draws, per-draw reset of interval weights, and
  fresh selection on a subsequent independent estimation run.
  Include short records with explicit component configurations: verify the
  applicability warning, attempted fitting without automatic component removal,
  and unchanged numerical failure reporting. Assess component recovery and
  sensitivity to observation coverage rather than treating successful construction
  as short-record validation. Preserve LFA-08's unavailable annual-change
  comparison when the reporting interval contains no complete lag pair.
  Exercise the same short-record policy through the Dask per-key path, including
  usable fits, numerical failures, and genuine prerequisite skips; duration
  must not prevent the requested attempt.
  Verify DCP/DPP and parameterized re-solves against fresh builds of the same
  formulation. Historical dictionary outputs need not match after this model
  change. Measure peak memory and construction/canonicalization/solve time on
  representative record lengths, including the removed dictionary allocation,
  before claiming an achieved performance improvement. Resolve LFA-03 through
  LFA-05 before interpreting sampled distributions. Coordinate dependency
  retirement with PKG-03.

#### LFA-07 — Sampling repeats construction and unnecessary attribution [performance proposal]

- **Locations:** sampling loop around lines 132–167; `estimate_losses` around
  lines 199–235; dictionary construction around lines 759–780.
- **Finding:** every draw rebuilds the problem and fixed dictionaries and calls
  full loss estimation, including Shapley attribution, although the retained
  sampled quantity is the degradation rate.
- **Proposed correction:** construct fixed-grid Fourier and capacity matrices
  and the direct soiling expressions once; remove the sawtooth dictionary under
  LFA-06. Parameterize the changing quantile, agreed soiling penalty coefficients,
  refinement weights, and observation mask, using combined coefficients where
  necessary to preserve DPP. Replace the old random dictionary-stiffness draws
  with the adopted selection/refinement workflow. Run selection once on the base
  data, then reuse the selected model and coefficients across draws (LFA-06).
  Each selected-soiling draw has a uniform-L1 fit and two refinement solves;
  a no-soiling draw has only its no-soiling fit. Preserve SDT's quantile
  configuration and reset per-draw adaptive interval weights.
  Extract the degradation quantity directly during sampling, using the
  common estimand and reporting interval specified in LFA-08. Perform final
  attribution after selecting the final model, unless a separately specified
  analysis actually requires attribution distributions. Keep sampling parameters
  distinct from the final fit's configuration, including fixed-rate settings
  only for linear models, and retain the separate summary and fitted-rate
  results specified in LFA-04. After stabilization or budget exhaustion with at
  least twenty successful draws and no overriding caller rate, explicitly fix
  the final linear rate to that run's sampled median under LFA-04; retain any
  lack-of-stabilization warning and do not propagate the rate constraint back
  into sampling solves.
  Respect the sampling budget and report the termination reason specified in
  LFA-10; parameterized reuse must not change those public semantics.
- **Proposed verification:** sampled estimates agree with fresh builds under
  controlled perturbations; final attribution is unchanged for the same selected
  model. Measure peak memory and construction, canonicalization, solving, and
  attribution time separately during authorized implementation validation.

#### LFA-08 — Use the endpoint chord for the overall rate and retain median annual change for comparison [estimand decision]

- **Locations:** `estimate_losses`, approximately lines 204–207; degradation
  reports and sampling outputs. Reference: `../rdtools/rdtools/signal_decomposition.py`,
  `extract_degradation_rate_log` around lines 810–895 and public rate selection
  around lines 2009–2015 at the RdTools revision recorded above.
- **Owner decision, 2026-10-22:** adopt the compound endpoint chord of the fitted
  log degradation component as the reported overall degradation rate, following
  the direction toward a unified approach with RdTools. Retain median annual
  change as a separately labeled comparison estimate. The forced flat tail is
  removed under the separate decision in LFA-09. Use the first-to-last
  admitted-observation span as the default reporting interval, fixed before
  Monte Carlo subsampling. Use $P=365.2425$ days for annualization and retain the
  365-day comparison lag with the $P/365$ log-change adjustment. Nonlinear models
  reject fixed-rate input under LFA-09; these design decisions do not authorize
  source changes.
- **Headline reporting, 2026-11-11:** apply this overall-rate extractor to the
  accepted final fit for the headline `degradation_rate`, as decided in LFA-04.
  A median across sensitivity-draw chord rates is a separate reported quantity;
  it does not replace the final-fit headline when the two differ. Retain the
  within-fit median annual-change comparison below as a third, distinct result.
- **Postprocessing decision, 2026-10-22:** adopt cumulative-minimum
  postprocessing for nonlinear monotone degradation before rate extraction.
  For the fitted log trend $\ell$, form the reporting curve

  $$
  \widetilde\ell_i=\min_{0\leq j\leq i}\ell_j.
  $$

  Use the full-model-grid convention of the inspected RdTools extractor, then
  evaluate both rate summaries over the agreed reporting interval $[a,b]$.
  Apply the same procedure to each successful sensitivity draw and the final
  nonlinear fit. For the linear model set $\widetilde\ell=\ell$; do not impose
  monotonicity through this postprocessing on a linear model that permits gains.
  Retain the solver's fitted trend and distinguish it from this reporting curve.
  This decision changes rate extraction, not the fitted decomposition or its
  Shapley factors. Label any displayed postprocessed curve accordingly and
  retain diagnostics for its difference from the fitted curve. Validate solve
  status, finiteness, and numerical feasibility before extraction; cumulative
  minima do not turn an unsuccessful solve into a valid fit.
- **Overall estimand:** let $d_i>0$ be the fitted degradation factor,
  $\ell_i=\log d_i$, $\Delta>0$ the sample interval in days, and $P=365.2425$
  days per year. For declared endpoint indices $a<b$, with duration
  $H=(b-a)\Delta$, use the reporting curve defined above and report

  $$
  r_{\mathrm{overall}}(a,b)
  =100\operatorname{expm1}\left(\frac{P}{H}
  (\widetilde\ell_b-\widetilde\ell_a)\right).
  $$

  When both grid endpoints are admitted in a daily record of $N\geq2$ samples,
  $a=0$, $b=N-1$, and $\Delta=1$,
  this is $100\operatorname{expm1}(P[\widetilde\ell_{N-1}-\widetilde\ell_0]/(N-1))$.
  The duration is $N-1$ daily intervals, not $N$ days. A linear log trend reduces
  to the rate in LFA-01. The inspected RdTools implementation uses the
  algebraically equivalent `exp(slope * T) - 1`, with `T` in samples per year;
  here `T` corresponds to $P/\Delta$. `expm1` is the proposed numerically stable
  evaluation, not a different estimand.
- **Comparison estimate, retained by owner decision:** compute median annual
  change from the same reporting curve and declared reporting
  interval $[a,b]$, with a distinct name and label in outputs and reports. For
  a positive integer lag of $L$ samples, the annualized comparison is

  $$
  r_{\mathrm{annual,med}}
  =\operatorname{median}_{a\leq i\leq b-L}
  \left\{100\operatorname{expm1}\left(
  \frac{P}{L\Delta}(\widetilde\ell_{i+L}-\widetilde\ell_i)\right)\right\}.
  $$

  **Owner decision, 2026-10-22:** retain the 365-day lag ($L=365$ on the daily
  grid) and annualize each log change with the factor $P/365$, using
  $P=365.2425$ days. Changing only the denominator in the current percentage differences
  does not implement the chord. For nonlinear trends these two summaries can
  intentionally differ; neither should silently retain the other's label or
  overwrite its value. If the declared interval contains no complete lag pair,
  report the comparison as unavailable while retaining a valid overall chord.
- **Reporting-domain decision, 2026-10-22:** use the span from the first to the
  last admitted observation. Let $\Omega_{\mathrm{base}}$ be the finite
  observations admitted by the fit's base mask before Monte Carlo subsampling;
  set $a=\min\Omega_{\mathrm{base}}$ and $b=\max\Omega_{\mathrm{base}}$.
  Require $a<b$ and finite fitted values used by the reporting transform. Both
  summaries use the reporting curve over $[a,b]$, rather than raw measurements;
  missing observations within the interval do not shorten its elapsed duration.
  The model may still extend over the full grid, but leading or trailing grid
  points outside the admitted-observation span do not extend the reporting
  duration or supply additional annual-change pairs. With the full-grid
  cumulative minimum, a leading fitted value can affect the reporting curve
  inside $[a,b]$ if the raw fitted trend has an upward numerical excursion;
  document and test that consequence of the postprocessing convention.
  Record the endpoint dates and duration, and hold them fixed across all
  parameter/subsample draws for that estimation run, even when a draw omits
  an endpoint observation. Recompute the base span for a new estimation run
  with different admitted observations. The former 90% cutoff does not define
  a reporting boundary after the tail removal in LFA-09.
- **Cross-package boundary:** agreement of extraction on the same fitted log
  trajectory does not imply agreement of fitted trajectories from different
  losses, structural priors, or masks. The cumulative-minimum convention is now
  explicitly adopted for nonlinear rate reporting. Compare packages on the same
  transformed curve and reporting interval; no runtime dependency on RdTools is
  introduced merely to share that convention.
- **Proposed verification:** check both extractors against common reference
  trajectories: a known exponential trend, a flat trend, and a nonlinear trend
  where median annual change and chord intentionally differ. Verify invariance
  of extraction under $d\mapsto c d$ for $c>0$, durations other than an integer
  number of years, and explicit missing-endpoint/invalid-endpoint handling. Use
  analytic expected values rather than only agreement between implementations.
  Verify that both summaries are retained and labeled separately, use the same
  reporting curve and declared interval, and handle an interval too short for
  the annual-change comparison. Check unchanged output for an already monotone
  trend, the cumulative minimum of a known upward numerical excursion, and
  retention of the original fitted trend. Include a leading inadmissible grid
  segment when checking the full-grid cumulative-minimum convention. Check
  exclusion of leading/trailing grid points from the reporting duration,
  preservation of duration across internal gaps, and fixed endpoint
  dates across subsampling, including draws that omit endpoint observations.
  Combine with LFA-01 round trips and LFA-04
  repeated-estimation checks.

#### LFA-09 — Remove the forced flat tail and reject nonlinear fixed-rate input [model/API decision]

- **Location:** `make_problem`, approximately lines 483–534; final model
  construction after Monte Carlo estimation.
- **Finding:** for $m=\lfloor0.9N\rfloor$, the nonlinear basis forces the fitted
  trend to be constant from index $m-1$ through $N-1$. A full-record chord
  includes this imposed plateau in its duration. For the illustrative log trend
  $\ell_i=\beta\Delta\min(i,m-1)$, that chord is

  $$
  100\operatorname{expm1}\left(P\beta\frac{m-1}{N-1}\right).
  $$

  The reported quantity therefore depends directly on the tail assumption.
  This is an algebraic illustration, not a numerical result. In addition, the
  nonlinear branch currently ignores `deg_rate`; only the linear branch uses
  the existing fixed-rate mechanism.
- **Tail decision, 2026-10-22:** remove the forced flat tail. Represent the
  nonlinear log degradation trend over the full model grid, retaining its
  non-increasing constraint, squared second-difference smoothness penalty, and
  existing initial-value anchor. Remove the reduced basis that repeats the
  value at the 90% cutoff through the end of the record. This is an explicit
  model change, separate from the conversion repair in LFA-01 and extraction
  change in LFA-08. Rate extraction uses the admitted-observation span specified
  in LFA-08, independently of the full grid used for fitting.
- **RdTools alignment checked, 2026-10-22:** at revision
  `d414a758ea85ae60482a931eda3efc188c3980f3`, the unmodified
  `../rdtools/rdtools/signal_decomposition.py` implements `trend_type='monotone'`
  with `diff(x2) <= 0` and a squared second-difference penalty in `make_problem`
  (around lines 203–210). Its optional end-drop penalty is disabled by the
  `lam_end=0.0` default in the builder and public `degradation` entry point.
  Thus the agreed trend shape and smoothness family match that default monotone
  formulation. This does not assert identical weights, component-level offsets,
  residual losses, or fitted decompositions across packages.
- **Fixed-rate decision, 2026-10-22:** reject any supplied non-`None` `deg_rate`
  when `deg_type="nonlinear"`, with a clear argument error before solving.
  Nonlinear estimation with `deg_rate=None` remains supported. The final
  nonlinear fit is free under the agreed monotonicity, smoothness, and anchor
  constraints; its endpoint chord is extracted after fitting. Do not implicitly
  pass a Monte Carlo summary into `deg_rate` when constructing that final fit.
  Retain the summary and the final fitted rate separately (LFA-04). Document
  rejection of this formerly ignored input in the migration guide.
- **Proposed verification:** use an agreed trajectory that continues declining
  in the final 10% to verify removal of the forced plateau and preservation of
  monotonicity, the initial anchor, and the smoothness penalty over the full
  model grid. Verify the selected reporting interval. For linear models, verify
  exact rate preservation; for nonlinear models, verify rejection of supplied
  fixed rates and successful construction with `deg_rate=None`, including after
  Monte Carlo sampling and on repeated estimation. Keep Monte Carlo summaries
  and final fitted rates distinct and prevent inheritance of a prior fitted
  rate during repeated estimation.

#### LFA-10 — The sampling limit is exceeded and termination is not distinguished [P2]

- **Location:** `estimate_degradation_rate`, approximately lines 168–196.
- **Finding:** the loop requires twenty draws before checking `max_samples`,
  then checks `counter > max_samples`. If draws otherwise succeed,
  `max_samples=1` permits twenty draws; limits of at least twenty can be exceeded
  by one when stabilization has not occurred. Returned sampling metadata does
  not distinguish stabilization from budget exhaustion.
- **Owner decision, 2026-10-22:** allow positive budgets below twenty for
  diagnostic runs, with results explicitly marked incomplete. Do not reject
  such budgets or exceed them to satisfy the usual minimum draw count.
- **Failed-draw decision, 2026-10-22:** count every attempted sensitivity draw
  toward `max_samples`, including a numerical failure that prevents a valid fit.
  Record the failure, exclude it from statistics, and continue within the
  remaining attempt budget. Do not launch replacement draws beyond that budget.
- **Final-fit continuation decision, 2026-11-11:** fewer than twenty successful
  draws leave sensitivity incomplete, but do not by themselves prevent final
  loss estimation. With no supplied fixed rate, attempt the free-rate full-data
  linear fit specified in LFA-04. Report its outcome separately from sensitivity
  completeness; retain all draw diagnostics and do not promote provisional
  running statistics to a normal summary. This final fit is not an additional
  sensitivity draw and does not alter the sampling budget or counts.
- **Budget-exhaustion decision, 2026-11-11:** if the budget ends with at least
  twenty successful draws before stabilization, retain the sampled median and
  computed sensitivity percentiles with an explicit budget-exhausted and
  not-stabilized warning. Use that median in the final linear fit under LFA-04,
  absent an overriding caller rate. Do not relabel this termination as
  stabilized, exceed the budget, or discard the available sensitivity summary
  merely because its stopping criterion was not met.
- **Proposed correction:** validate a positive integer budget and enforce its
  bound before starting another attempt. Track attempted, successful, and failed
  draw counts separately, with failure reasons and sampled settings retained.
  Retain twenty successful draws as the
  minimum for the normal sampling summary and stabilization checks. With fewer
  draws, return the collected samples and an explicit incomplete-result status;
  any displayed running statistics are provisional diagnostics, not the normal
  summary. Report successful draw count and termination reason, and distinguish
  insufficient draws from budget exhaustion after the minimum was reached but
  before stabilization. Preserve LFA-04's separation of sampled summaries from
  fitted rates; an incomplete diagnostic result must not masquerade as a completed
  summary or silently inherit a previous run's estimate.
  Failed attempts do not enter the minimum successful-draw count, running
  quantiles, or stabilization window; repeated failures cannot count as unchanged
  statistics and trigger stabilization. Restore the base observation mask and
  configuration after each attempt (LFA-03), and never reuse stale solver values
  as a successful draw. If no draw succeeds, return the failure record and an
  incomplete result without fabricated numerical summaries.
  This continuation policy concerns numerical draw failures, not suppression of
  invalid configuration or programming errors. A valid preceding fit retained
  under LFA-06's reported refinement fallback is distinct from a draw with no
  valid fit; preserve the refinement diagnostics. The budget counts sensitivity
  draws, not each convex solve within selection or refinement (LFA-07).
  Stabilization and the twenty-draw minimum remain numerical procedure rules,
  not evidence of confidence coverage (LFA-02).
- **Proposed verification:** controlled sequences with budgets below, at, and
  above twenty; stabilization at the budget boundary; a sequence that never
  stabilizes; and failed draws under the selected policy. Assert both counts
  and reported reasons. Verify that a one-draw budget yields at most one draw,
  retains its diagnostic output, and reports incomplete results without a normal
  summary or stabilization claim. Verify the nineteen/twenty-draw boundary and
  fresh diagnostic state after an earlier full run. Include intermittent and
  all-failed draws; verify that attempted = successful + failed, attempted never
  exceeds the budget, only successful draws enter statistics, and failures
  cannot trigger stabilization or contaminate the next draw's state. For a
  budget-exhausted run with at least twenty successes and no stabilization,
  verify retained percentiles, the median-constrained final linear rate, and
  persistent warning/termination metadata. Coordinate
  configuration/restoration with LFA-03 and construction/solve accounting with
  LFA-07.

### `solardatatools/algorithms/quantile_estimation.py`

**Owner decision, 2026-10-22:** move the required `spcqe` quantile functionality
into native CVXPY code in SDT, shared with anomaly detection. PKG-03 records the
implementation boundary, required fitting/transform behavior, dependency
retirement, and verification. QUANT-01 remains a separate plotting correction.

#### QUANT-01 — Original-grid plotting uses the dilated-grid reshape [P2]

- **Location:** `plot_all_quantiles`, approximately lines 105–136.
- **Finding:** `dilated=False` selects original-grid arrays, but plotting still
  drops the first sample and reshapes using `nvals_dil`. The original grid does
  not have that initial dilated bin, so the dimensions fail.
- **Proposed correction:** select the slicing rule and samples-per-day count for
  the chosen grid.
- **Proposed verification:** plot both representations with distinct original and
  dilated resolutions, including one and several quantile levels.

### `solardatatools/algorithms/shade.py`

#### SHADE-01 — Resampling applies energy normalization to incompatible components [P2]

- **Locations:** ratio construction near line 199; annotation calls near lines
  237 and 280; component reconstruction near lines 96–106;
  `undo_batch_process` normalization near line 453.
- **Finding:** the dimensionless ratio $r=s/c$ is passed through a transformation
  that preserves column sums. A constant ratio on an $M$-point grid becomes
  $\widetilde r=(M/n_d)r$ on a day with $n_d$ daylight samples. For $M=256$ and
  $n_d=144$, a 20% ratio becomes 35.6%, crossing the default 25% threshold.
  The inverse transform also divides unconditionally by the interpolated column
  sum. An identically zero shade component produces $0/0$ on daylight entries.
  A nonzero signed residual with zero total can become undefined or be erased,
  depending on whether interpolation preserves its zero sum; near cancellation
  can make this rescaling unstable.
- **Owner decision, 2026-11-11:** replace per-component renormalization with one
  shared linear, energy-conserving resampling map for clear sky, shade, and
  residual, with validation. Retain the current fitted model initially and
  validate the map against its daylight-grid and corpus conventions.
  Dimensionless shading fractions require separate treatment.
- **Proposed correction:** for each day $d$, construct a map
  $R_d\in\mathbf{R}^{n_d\times M}$ determined by the source/target grids and
  their sampling measures, independent of component values. Apply the same
  map to all three components. Thus, for transformed clear sky $c$, shade $s$,
  and residual $e$,
  $$
  R_d(c-s+e)=R_dc-R_ds+R_de.
  $$
  Require a nonnegative map to preserve nonnegativity of clear sky and shade.
  If $a_d$ and $b_d$ are the source and target energy-integration weights,
  respectively, energy conservation is the linear identity
  $$
  b_d^T R_d=a_d^T.
  $$
  Specify those weights using the existing transformed-column energy
  convention, original sampling interval, and units; do not reinterpret the
  transformed entries as ordinary power samples without the corresponding
  measure. A bin-overlap map is an implementation candidate, subject to
  verification of the grid and endpoint conventions. No component-dependent
  normalization or division by a component sum is needed. An identically zero
  component maps to zero, and signed residuals with zero or nearly cancelling
  totals remain well-defined. Do not introduce arbitrary denominator floors
  or erase a residual merely because its total vanishes.
- **Empty target-day boundary:** if $n_d=0$, no map into the empty target
  grid can conserve nonzero source energy. Mark that day's reconstruction
  unavailable with its reason; do not fabricate zero output and claim energy
  conservation. Distinguish this from an empty day whose source energy is
  zero, and specify how each case enters exported totals and plots.
- **Shading-fraction decision, 2026-11-11:** compute original-grid shading
  fractions from the resampled components:
  $$
  \widetilde r_{d,t}=\frac{(R_ds)_t}{(R_dc)_t},
  $$
  restricted to entries with adequate resampled clear-sky power. Use this same
  fraction for the original-grid heatmap and polar annotations, so their
  thresholds refer to the reconstructed components. This supersedes the
  proposal to interpolate the normalized-grid ratio $s/c$. Do not apply energy
  normalization to a fraction. Specify and validate the low-clear-sky
  eligibility rule on the target grid; exclude ineligible entries from division
  rather than stabilize them with an arbitrary denominator floor.
- **Eligibility propagation, follow-up review:** carry the eligibility mask
  through thresholding, dataframe export, and polar aggregation. The current
  path turns low-clear-sky samples into zero fractions/`False` annotations,
  counting unassessed samples as unshaded. Keep them unavailable and exclude
  them from aggregation denominators. A Boolean display value alone cannot
  convey this distinction; retain assessment coverage separately.
- **Scope and remaining specification:** this changes reconstruction and
  fraction extraction, not the
  fitted objective, basis, corpus, or forward transform. Verify compatibility
  with those existing conventions before adoption; any necessary change to
  them requires a separately stated proposal. Linearity of reconstruction does
  not imply exact inversion of resampling or declination-bin aggregation.
- **Proposed verification:** check the energy identity and component additivity
  across day lengths and resolutions, using the declared integration weights.
  Include zero shade, signed zero-total and nearly cancelling residuals, and
  nonnegative power components. Check finite outputs, exact zero preservation,
  nonnegativity preservation, and daily/yearly energy consistency. Evaluate
  reconstructed profiles against the existing daylight-grid and corpus
  conventions without assuming pointwise equivalence to the old nonlinear
  normalization. Verify that $s=\alpha c$ gives the fraction $\alpha$ wherever
  the mapped clear-sky component is eligible. Include varying clear sky and
  shading fractions for which interpolation of $s/c$ differs from the adopted
  ratio of mapped components. Check low-clear-sky handling, common fractions
  through both annotation paths, and annotation thresholds with a consistent
  physical interpretation.
  Check that ineligible samples do not reduce reported shaded fractions during
  polar aggregation. Include days with no target daylight bins and both zero
  and nonzero source energy; verify unavailable results and coverage reporting.

#### SHADE-02 — Basic package import eagerly loads the shading corpus [P2]

- **Locations:** module-level corpus read near line 30; `make_class_parameters`
  near line 469. Import chain includes `solardatatools.__init__`, `DataHandler`,
  and `solardatatools.algorithms.__init__`.
- **Finding:** ordinary package import parses a 167,780,005-byte CSV from the
  compressed corpus and retains it, although only `make_class_parameters` uses
  it. Worker processes repeat this work. Archive size was inspected; import time
  and peak memory were not measured.
- **Proposed correction:** load/cache the corpus when that operation requires it.
- **Proposed verification:** ordinary import does not read the corpus; first-use
  loading produces unchanged class parameters. Any performance measurements are
  separate proposed work, not existing evidence.

#### SHADE-03 — Repeated runs ignore changed model settings [P2]

- **Location:** `run`, approximately lines 81–89; `has_run`, approximately
  lines 57–66.
- **Finding:** after the first run, `has_run` skips both construction and solving.
  New weights and other model settings therefore do not affect the returned
  components. The name `osd_problem` is historical: this implementation already
  builds a native CVXPY problem.
- **Proposed correction:** update parameters and solve for changes supported by
  the cached structure; rebuild for changes to grid, basis, dimensions, or other
  structural choices. Extract results only after validating the new solve. Use
  naming that reflects the retained native implementation.
- **Proposed verification:** repeated runs with changed weights agree with fresh
  instances, and structural changes trigger reconstruction. Coordinate with
  SD-06; merely retaining a problem object is not a complete re-solve mechanism.

### `solardatatools/algorithms/soiling.py`

**Owner decision, 2026-11-11:** retire the existing, separately implemented
standalone soiling algorithm and direct users to the main loss-factor analysis.
This supersedes the 2026-10-22 direction to provide a thin standalone interface.
Withdraw the standalone quantile-default question. LFA retains its configurable
quantile loss and default $\tau=0.9$.

**Agreed disposition; implementation pending:** retire `SoilingAnalysis`, `soiling_seperation`, and
`soiling_seperation_old` as separate public analysis entry points, and migrate
users to LFA's soiling component and selection/refinement workflow. Coordinate
removal of their exports in `solardatatools/algorithms/__init__.py` with callers,
examples, and API documentation. Do not require a replacement standalone
wrapper. The findings below remain evidence for retirement, migration, and
verification of the corresponding LFA behavior, rather than instructions to
repair and retain the additive estimator. Document changes in component units,
correction factors, defaults, and obsolete arguments. Explain that the old
standalone default was $\tau=0.85$ and that LFA accepts an explicitly configured
quantile, including 0.85; the same numerical quantile does not make the old
additive and new log-domain estimators equivalent. Agreement on retirement is
a planning decision, not authorization to remove source or public exports now.

#### SOIL-01 — An additive loss is applied as a fractional correction [P2]

- **Locations:** `SoilingAnalysis.run`, approximately lines 46–49; decomposition
  consistency constraint near line 151.
- **Finding:** the fitted model is $y=b+s+r$, with $b$ the seasonal-plus-degradation
  baseline and $s\leq0$ additive in normalized energy units. Dividing power by
  $1+s$ is inconsistent with that model. For $b=0.5$, $s=-0.1$, and $r=0$, the
  result is $0.4/0.9$, rather than the clean baseline $0.5$.
- **Disposition and proposed correction:** retire the standalone additive
  correction and document extraction from the log-soiling model in LFA-06.
  If $s_t\leq0$
  is its fitted log-soiling component, the multiplicative soiling factor is
  $\exp(s_t)$ and removing that factor gives

  $$
  p_{\mathrm{corrected}}(\tau)=p(\tau)\exp(-s_t),
  \qquad \tau\text{ in day }t.
  $$

  Apply the daily factor consistently to subdaily power. This removes only the
  fitted soiling factor; it preserves the other fitted factors and residual
  variation in the observed signal. Distinguish the log component, the soiling
  factor, and its inverse correction multiplier in results and plots. Preserve
  source arrays, zeros, and missing values when producing corrected arrays;
  update labels and configuration mapping for the changed model.
- **Proposed verification:** exercise the migration example using LFA's fitted
  soiling component, with explicit data, mask, and resolved settings. Verify the
  multiplicative correction against known log-soiling
  under a varying baseline, retention of other factors and residual variation,
  and finite factor extraction. Equality with the old additive fit is not required.

#### SOIL-02 — The advertised asymmetric slope penalty is symmetric [P2]

- **Location:** `soiling_seperation`, approximately lines 140–145.
- **Finding:** with $D$ the first-difference operator, the implemented term is

  $$
  0.9\sum_i(Ds)_i^+ +(1-0.1)\sum_i(Ds)_i^-
  =0.9\|Ds\|_1.
  $$

  Equal positive and negative increments therefore have equal cost.
- **Disposition:** supersede repair of the additive model's slope coefficients.
  Use LFA-06's direct log-soiling penalty: an unsquared L2 norm of downward first
  differences and an L1 magnitude term, with the agreed interval-weighted
  refinements. Do not carry the old symmetric term into the LFA formulation.
- **Proposed verification:** verify the downward-increment and magnitude terms
  separately against the LFA-06 specification. There is no requirement to
  construct a replacement standalone interface for this verification.

#### SOIL-03 — Short-record handling removes the constant baseline [P2]

- **Location:** `soiling_seperation`, approximately lines 150–163.
- **Finding:** for fewer than $0.75(365)$ samples, `s2 == 0` disables the entire
  seasonal component, including its constant level. With $T=100$, $y_t=1$,
  degradation disabled, and default positive weights, the remaining constraints
  and losses yield zero soiling, unit residual, and zero denoised signal. The
  positive constant baseline has nowhere else to enter the reconstruction.
- **Disposition and proposed correction:** retire the standalone seasonal-vector
  construction and its short-record branch. In the supported LFA model, preserve
  the baseline level when seasonal variation is disabled; do not translate
  `s2 == 0` into removal of that level during migration. Under the owner's
  2026-11-11 decision in LFA-06, short records produce an applicability warning
  and an attempted fit of the explicitly configured model, without automatic
  seasonal removal or a new one-year prerequisite. Retirement of the old path
  does not validate short-record soiling/degradation separation.
- **Proposed verification:** the constant-signal counterexample and a short
  record with a known baseline and loss; inspect components and reconstruction.
  This counterexample is algebraic evidence, not a completed numerical test.

#### SOIL-04 — Difference order and IRL1 scaling need an explicit definition

- **Location:** `soiling_seperation`, docstring around line 98, objective around
  line 138, and reweighting around lines 165–169.
- **Finding:** the docstring describes sparse first differences, while the
  reweighted penalty acts on second differences. The update is

  $$
  \frac{1}{0.01+100|(D_2s)_i|}
  =\frac{0.01}{10^{-4}+|(D_2s)_i|}.
  $$

  It therefore includes both an effective $\epsilon=10^{-4}$ and a penalty
  rescaling. This is a standard reciprocal form after accounting for scale;
  the inconsistency concerns its stated meaning and initialization contract.
- **Disposition and proposed correction:** retire the standalone second-
  difference reweighting loop. Use LFA-06's component-based selection and
  two cleaning-interval magnitude refinements. Their reference $\epsilon=0.01$
  is in log-soiling units; do not reuse the old effective $10^{-4}$ scale or
  apply SD-04's time-shift/capacity stopping rule to this separate procedure.
  Reuse the native parameterized model and solver/status handling (SD-06 and
  LFA-07). Retire the duplicate additive implementations, including
  `soiling_seperation_old`, with explicit migration to LFA. The previous proposal
  to retain compatibility entry points is not a requirement.
- **Proposed verification:** verify LFA path selection, no-soiling fallback,
  refinement counts and interval weights, reset behavior, and solver-option
  propagation. Coordinate with SOIL-01 through SOIL-03 to verify extraction and
  baseline behavior; do not preserve the retired second-difference objective
  through historical fixtures.

### `solardatatools/algorithms/sunrise_sunset_estimation.py`

#### SUN-01 — Solar-noon RMSE is computed from aggregate bias [P2]

- **Location:** `calculate_errors`, approximately lines 350–369.
- **Finding:** both events and all days are averaged before the residual is
  formed. Daily noon errors of $+1$ and $-1$ hours consequently report zero rather
  than an RMSE of one hour.
- **Proposed correction:** calculate noon separately for each day, then form daily
  residuals and RMSE. Define missing-event handling consistently.
- **Proposed verification:** the cancellation example and missing-event cases.
- **Related:** TIME-02 owns the sampling-grid construction issue in this file.

The threshold search in `run_optimizer` (approximately lines 184–293) is also a
caller of the reusable seasonal-quantile model proposed in SD-06. Retain its
final full-data fit and distinguish this observation-extraction threshold from
the Tier-3 structural weights discussed in SHIFT-01 and CAP-02. The time-shift
selection critique does not invalidate capacity's component-based screening or
every reconstruction holdout calculation.

### `solardatatools/algorithms/time_shifts.py`

#### SHIFT-01 — Reconstruction holdout does not identify the shift component [P2]

- **Owner scope clarification, 2026-10-21:** update time-shift detection to
  Tier-3 parameter selection, using the existing capacity selector as the
  archetype. Specify a simple component-informed screening rule and validate it
  before adoption. Capacity selection itself is preserved apart from focused
  reviewed corrections. The skill's instruction to judge the component does
  not prescribe abstention machinery, stability-based ranking, or refitting
  every candidate support; those proposed extensions are withdrawn.
- **Location:** `optimize_w1`, approximately lines 201–272; transition extraction
  around lines 120–135 and correction rounding around lines 158–181.
- **Finding:** the selection rule uses reconstruction holdout-error changes to
  choose a structural penalty. Jump counts, total variation, and seasonal RMS are
  calculated but do not enter that rule. Competing allocations between shift and
  seasonal components can have similar reconstruction error, so a score minimum
  or discontinuity does not establish the physical shift structure.
- **Proposed correction:** follow the capacity selector's pattern: evaluate a
  short weight grid, screen the recovered clock-shift component for meaningful
  jump amplitudes and a plausible number of jumps, and choose the smallest
  weight satisfying the agreed criteria. Use the agreed 2% in-sample
  fit-improvement screen alongside the component criteria; reconstruction
  holdout error does not determine the structural weight. The three screening
  criteria and fallback are specified below. Perform the agreed single
  fixed-support refit after selection.
- **Minimum-jump decision, 2026-10-21:** use five minutes ($1/12$ hour) as the
  minimum detectable clock shift. Use this physical threshold for meaningful
  jumps in candidate screening and for pooled support extraction, replacing
  the existing 0.05-hour (three-minute) support cutoff. Evaluate magnitudes
  before correction rounding. This specifies the acceptance threshold; actual
  detection performance remains subject to validation.
- **Jump-frequency decision, 2026-10-21:** use at most ten clock changes per
  year as the proposed candidate-screening bound, using the existing record-wide
  annualization convention. Very large counts indicate a likely decomposition
  failure. This is a domain screening rule, not an additional constraint on the
  convex subproblems. The owner regards ten as a likely safe bound; establish
  its adequacy through thorough testing at implementation time.
- **Fit-improvement decision, 2026-10-21:** require at least a 2% reduction in
  in-sample squared residual error relative to the constant-shift fit, subject
  to validation. For candidate weight $\lambda$, use
  $E_\lambda=\sum_{t\in\Omega}(y_t-c_t^{(\lambda)}-s_t^{(\lambda)})^2$ and
  accept this screen when $E_\lambda\leq0.98E_{\mathrm{const}}$, for positive
  reference error $E_{\mathrm{const}}$. Both errors use the same admitted
  observations and seasonal model. This compares residual loss, not the full
  regularized objective, using the IRL1 candidates before the single selected
  fixed-support refit. The reference can come from a sufficiently large grid
  weight yielding a constant shift component; it does not require a separate
  estimator. For zero reference error, use the constant-shift fallback below.
  This fit screen supplements the component criteria and does not replace them.
- **Endpoint decision, 2026-11-11:** apply CAP-02's endpoint verification and
  bounded upper-range extension to the automatic time-shift search. Keep the
  original grid if its endpoint is sufficient. Check numerical constancy over
  the full grid, not merely absence of pooled jumps exceeding five minutes.
  With the first-segment anchor, a constant shift component is zero within the
  numerical tolerance. Reuse the verified fit as the reference and report any
  added weights and extension termination. Preserve the agreed model, masks,
  recurrence option, candidate screens, and single selected-support refit.
- **Fallback decision, 2026-10-21:** if no candidate passes all three screens,
  return a constant shift component. Use the same outcome when the constant-shift
  reference fit has zero residual error. With the first segment fixed at zero,
  the result is $c=0$, empty jump support, and no relative clock correction.
  Retain the fitted solar-noon component. The empty-support refit follows SD-03;
  an already available equivalent constant-shift fit can be reused.
- **Numerical search failures, 2026-10-22:** continue over usable candidates and
  report failed weights as specified in SD-04. The preceding no-change fallback
  applies to screening usable fits against a usable reference. An unavailable
  reference or a search with no usable candidate is unsuccessful selection,
  not evidence for $c=0$. Distinguish a numerical fit failure from exhaustion
  of the bounded extension without a constant reference. The zero-error-reference
  fallback remains valid when its accepted constant-shift fit is available.
- **Correction-rounding decision, 2026-10-21:** round applied corrections
  directly to the nearest sample by default; retain whole-hour rounding as an
  explicit option. Apply this default consistently in `TimeShift.run`,
  `run_pipeline`, and `auto_fix_time_shifts`, which currently default to
  `round_shifts_to_hour=True`, and document the change in the migration guide.
  The five-minute detection threshold applies to the fitted component before
  rounding. The input cadence limits the resolution of the applied correction;
  retain the distinction between fitted shifts and applied corrections in
  outputs and documentation.
- **Conflicting correction suppression, follow-up review:** `TimeShift.run`
  (around lines 175–177) sets every correction to zero when the mean absolute
  sample roll is below 0.01. An accepted one-sample correction on one day of
  a record longer than 100 days is erased by this rule, irrespective of its
  physical magnitude. Retire that record-average suppression so the agreed
  selection and rounding determine the applied correction. Preserve the
  distinction between a detected shift that rounds to zero at the acquisition
  cadence and a shift suppressed by an unrelated duration-dependent rule.
  Verify a qualifying short-lived shift through public correction application
  on a long record, including nearest-sample and optional whole-hour rounding.
- **Recurrence decision, 2026-10-21:** retain exact 365-day recurrence of the
  shift component as explicitly opt-in, default off. Remove the automatic retry
  currently triggered by seasonal RMS exceeding 0.25 hours, which can enable
  recurrence even when `periodic_detector=False` was supplied. The recurrence
  assumption is distinct from annual seasonality and does not represent a
  general calendar-based clock-change rule. The accepted 365.2425-day Fourier
  seasonal default does not change the shift recurrence.
- **Recommendation-flag direction, 2026-10-21:** test a flag recommending that
  the user consider the recurrence option. It is advisory and must not enable
  recurrence, rerun correction, or alter the fitted result automatically. The
  trigger and its usefulness remain to be evaluated during implementation;
  the existing 0.25-hour seasonal-RMS threshold is not a validated flag rule.
  If seasonal RMS is used, remove the Fourier offset as specified in SD-01.
- **Reference-level decision, 2026-10-21:** set the first PWC segment to zero
  and interpret the Fourier component as solar noon in that initial clock
  regime. Retire the current nearest-noon fallback from relative correction
  extraction; coordinate the component-role and diagnostic translations in
  SD-01 with the advisory-only timezone checking in DH-04.
- **Proposed verification:** representative known shifts and confounded
  seasonal/shift examples, inspecting both components and the final rounded
  corrections. Include jumps below, at, and above five minutes and the effect
  of the agreed rounding policy at different input cadences. Include cases
  below, at, and above ten changes per year, with different record lengths,
  missing observations, and seasonal confounding. Verify the 2% residual-error
  comparison, its constant-shift reference, and zero-reference-error handling.
  Verify the no-qualifying-candidate fallback returns $c=0$ and leaves relative
  clock alignment unchanged. Verify that recurrence is off unless explicitly
  requested. Evaluate the recommendation flag on recurring and nonrecurring
  shifts and seasonal confounding, including false recommendations; verify that
  the flag leaves the selected model and corrections unchanged. Review whether
  the screening criteria select the
  intended component shapes. Coordinate support/refit semantics with SD-03,
  IRL1 with SD-04, and adoption of corrected data with DH-02.

## 4. Anomaly detection

### `anomalydetector/detector_model.py`

#### OUT-01 — Public `fit` forwards an unsupported argument [P1]

- **Locations:** `fit`, approximately lines 412–419; `fit_quantiles` signature,
  approximately lines 209–216.
- **Finding:** every `fit` call passes `client=client` to a method that does not
  accept it, including when the value is `None`.
- **Proposed correction:** remove obsolete forwarding or implement a consistent
  contract if client support is intended.
- **Proposed verification:** exercise the public orchestration method with
  lightweight stage substitutes and verify argument compatibility.

#### OUT-02 — Classifier selection disables or misnames supported models [P1]

- **Locations:** constructor dispatch, approximately lines 96–187;
  `pyproject.toml` optional dependencies and anomaly installation documentation.
- **Finding:** an `else` belongs only to the last of several independent `if`
  statements, overwriting most selected estimators, including the default
  ensemble and SVM, with `None`. The PCA+QDA and PCA+logisticRegression branches
  reference incorrectly named local variables instead. The default ensemble and
  explicit XGBoost selection also require `xgboost`, which is absent from the
  declared dependencies.
- **Owner decision:** provide XGBoost through an optional anomaly extra and
  retain the ensemble default.
- **Additional fitting defect, follow-up review:** the explicit XGBoost
  constructor enables `early_stopping_rounds=10`, but `fit_residuals` supplies
  no evaluation set. The inspected local XGBoost 3.0.0 callback requires at
  least one validation dataset for early stopping; the supplied assessment
  reports the same check in 3.0.2. Dependency declaration and constructor
  dispatch alone therefore do not establish a working fit path.
- **XGBoost fitting decision, 2026-11-11:** remove early stopping from the
  explicit XGBoost branch and train for its configured `n_estimators`
  (currently 100 boosting rounds). Retain the classifier validation split for
  scoring; it does not select the stopping iteration. No additional stopping
  split is introduced. Preserve the optional anomaly-extra and ensemble-default
  decisions above.
- **Proposed correction:** use mutually exclusive dispatch or a constructor
  mapping, assign each constructed estimator consistently, and reject unknown
  names. Declare `xgboost` in `solar-data-tools[anomaly]` and document that the
  default ensemble requires this extra. Keep its import conditional on selecting
  a classifier that needs it. Without the dependency, ensemble and XGBoost
  selection must give a clear error identifying the required extra; do not
  silently substitute another classifier or change the ensemble membership.
  Apply the agreed fixed-round XGBoost fitting configuration and document the
  training and classifier-validation roles.
- **Proposed verification:** instantiate every advertised selection and exercise
  its residual-fitting path; cover a caller-supplied estimator and unknown names.
  Verify that the base installation can import the anomaly API and use
  classifiers that do not require XGBoost, that missing-extra errors are explicit,
  and that installing the anomaly extra supports the default ensemble and
  explicit XGBoost paths. Coordinate installed-package checks with QA-02.
  Exercise an actual fit, not merely construction, and verify the configured
  boosting-round count and the documented roles of training and classifier
  validation observations. Validation data must not enter classifier training
  or determine a stopping iteration.

#### OUT-03 — Prediction classifies training residuals [P1]

- **Location:** `predict_residuals`, approximately lines 507–510.
- **Finding:** prediction concatenates `residuals_train` and `residuals_failure`,
  ignoring `residuals_test` produced by `predict_linear`. The output concerns the
  training cases rather than the supplied prediction dataset.
- **Proposed correction:** transform and classify prediction residuals using the
  stored preprocessing and classifier.
- **Proposed verification:** a different prediction-day count and distinguishable
  residual values; verify output length and data dependence.

#### OUT-04 — First prediction writes into uninitialized storage [P2]

- **Locations:** initialization near line 199; assignment near line 470.
- **Finding:** `quantile_test` is initialized to `None`, then indexed as a
  dictionary in `predict_quantiles`.
- **Proposed correction:** initialize prediction storage for each prediction call
  so stale keys cannot survive across datasets.
- **Proposed verification:** first and consecutive predictions with different
  datasets; coordinate with OUT-03 while retaining separate failure assertions.

#### OUT-06 — Fitting overrides leave dimension-dependent regression weights stale [P2]

- **Locations:** constructor defaults near lines 87–90; `fit_linear`,
  approximately lines 337–355; `utils.py`, `weighted_ridge_regression`.
- **Finding:** default weights are constructed for the initial lag count and
  basis dimension. `fit_linear(nlag=..., num_basis=...)` changes the design
  matrix but retains the old weights unless replacements are supplied. The
  augmented regression can then combine matrices with incompatible column counts.
- **Proposed correction:** resolve the feature/basis layout before constructing
  its default weights. Distinguish generated defaults from caller-supplied
  weights; validate explicit weights against the resolved coefficient count
  and ordering rather than silently replacing or resizing them. For $S$ sites,
  lag count $q$, and $B$ basis functions, the current design has
  $p=B((S-1)(2q+1)+1)$ coefficients. Validate the supported dimensions and the
  finite, nonnegative weight domain required by the regression formulation.
- **Proposed verification:** override each dimension alone and together, repeat
  fitting with changed dimensions, and supply matching and mismatched explicit
  weights. Compare a fit-time override with fresh construction at the same
  resolved configuration.

### `anomalydetector/multidata_handler.py` and `anomalydetector/utils.py`

#### OUT-05 — Split ordering and mask coordinates violate reconstruction assumptions [P2]

- **Locations:** `train_test_split`, approximately lines 244–277; `dilate`,
  approximately lines 99–105; `common_days`, `full_signal`, and
  `reconstruct_signal` in `utils.py`.
- **Findings:** randomized membership is retained in random chronological order,
  but reconstruction uses the first date as the origin and the last date as the
  extent. Earlier dates produce negative offsets; an interior maximum date can
  exceed the allocated array. Independently, the split uses indices into the
  common-day array to slice masks expressed in each raw handler's day coordinates.
  The resulting shortened masks are stored alongside the original full raw
  handlers. Subsequent dilation applies those masks to full-length matrices,
  which fails even with `shuffle=False`; sorting alone does not repair this.
- **Proposed correction:** randomize membership, then sort each partition
  chronologically and apply that ordering to every associated array. Retain
  full-length partition masks in each original handler's day coordinates,
  marking the raw positions corresponding to the selected common dates. Require
  each handler's masked dates to equal the partition's `common_days` in order;
  its raw-mask length need not equal the number of selected days. Preserve the
  parent object's masks and arrays.
- **Proposed verification:** use handlers with different date extents, excluded
  interior days, and distinct date/value sentinels. Exercise both shuffle modes
  and both split-then-dilate and dilate-then-split paths. Compare selected dates,
  values, mask lengths, and reconstruction, not only output shapes.
- **Related:** PKG-01 owns NumPy API compatibility in outage stages.

## 5. PV system profiling

### `pvsystemprofiler/algorithms/performance_model_estimation.py`

#### PROF-01 — Replace invalid annual-vector constructions with constrained Fourier components [P2]

- **Location:** `find_fit_costheta`, approximately lines 37–41.
- **Finding:** first differences are formed for singleton solstice-symmetry groups
  (including day 172 in a single year). The unconditional annual-lag equality
  also creates empty expressions for $T\leq365$.
- **Owner decision, 2026-11-11:** use Fourier representations with physical
  constraints for both smooth annual terms, with period $P=365.2425$ days, and
  validate the changes. Preserve the incidence term's solstice symmetry and
  both terms' nonpositivity. This supersedes the proposed repairs to the
  singleton and annual-lag equalities: retire those vector constructions.
- **Proposed formulation:** write the log-incidence component as
  $g=B_{\mathrm{sym}}a$, using a cosine basis about a fixed solstice phase, and
  the other annual log component as $s=B_{\mathrm{ann}}b$, using a sine/cosine
  basis. Include a constant coefficient in each annual component; do not import
  the capacity model's zero-DC convention into these nonpositive components.
  Evaluate both on physical elapsed time with the common annual period;
  enforce $g_t\leq0$ and $s_t\leq0$ over the entire model grid. Specify the
  solstice phase and calendar convention explicitly; a day-of-year reset must
  not introduce discontinuities into a representation with a fractional-year
  period. This proposal does not introduce a fitted phase parameter.
  Retain the unpenalized affine log trend and the current squared residual
  loss and normalization. All components remain defined on the full grid,
  with consistency imposed on admitted observations and the predefined
  residual zero outside that set, as specified in SD-01/SD-06.
- **Specifications to validate:** harmonic counts, solstice phase convention,
  and the regularization of the two annual components, including the resulting
  allocation of level between those components and the affine trend.
  The current second-difference and magnitude penalties can be applied to the
  evaluated Fourier components as a starting proposal. Do not silently equate
  those penalties with an unweighted coefficient norm or assume that their
  existing weights remain appropriate. A truncated basis and the physical
  annual period change the admissible curves; this is an explicit model change,
  not an output-equivalent repair. Coordinate the time-index and result
  contracts with PROF-02/PROF-03 and the native decomposition API.
- **Proposed verification:** cover partial-year, one-year, and multiyear
  records, leap/calendar boundaries, and missing observations. Check annual
  period, solstice symmetry, and full-grid nonpositivity; eliminate the empty
  difference/equality constructions. Assess component separation and downstream
  tilt/azimuth recovery using controlled geometry and representative records,
  including sensitivity to harmonic counts and regularization. Check compatible
  solver results and masked-data behavior under SD-06/QA-04. Define the
  statistically supported data regime; successful construction or a small
  residual alone does not establish orientation identifiability.

### `pvsystemprofiler/tilt_azimuth_study.py`

#### PROF-02 — Study initialization and decomposition calls violate the current contract [P2]

- **Locations:** constructor, approximately lines 97–100; `run`, approximately
  lines 170–172; downstream
  `algorithms/angle_of_incidence/curve_fitting.py`, approximately line 26.
- **Findings:** the constructor captures `filled_data_matrix` before implicitly
  running preprocessing. For an initially unprocessed handler, the study's
  stored `data_matrix` remains `None` after the handler is populated.
  `find_fit_costheta` is also called without required `doy`. Merely adding
  that argument is insufficient: the caller subsequently uses the returned
  one-dimensional envelope where a two-dimensional normalized data matrix is
  required.
- **Default-path solver finding, follow-up review:** when no explicit daytime
  threshold is supplied, the study's filtering path calls
  `algorithms/tilt_azimuth/daytime_threshold_quantile.py`, whose quantile solve
  hard-codes MOSEK and uses a vector with 365-day constraints. Bring this caller
  under the agreed native annual-Fourier and CLARABEL/default-options contract,
  preserving or explicitly specifying its loss normalization and physical
  constraints. A corrected constructor and main orientation builder would
  otherwise leave an undeclared MOSEK requirement on the default public path.
- **Owner decision:** require an already preprocessed `DataHandler` when
  constructing `TiltAzimuthStudy`; retire its automatic `run_pipeline` call.
- **Proposed correction:** check that the required processed matrix, day index,
  and clear-day flags are available and aligned before capturing the study's
  inputs. An unprocessed or incomplete handler must produce a clear prerequisite
  error directing the caller to run `run_pipeline` explicitly with the desired
  settings; `_ran_pipeline` alone is insufficient evidence of usable state.
  Update examples and migration guidance to show preprocessing before study
  construction. Update both arguments and interpretation of decomposition
  return values, consistent with the current `ConfigurationEstimator` call.
- **Proposed verification:** unprocessed and incomplete handlers fail clearly
  without launching preprocessing or mutating handler state. A prepared handler
  supports the public constructor-to-`run` path with supplied location. Verify
  the captured matrix and metadata and the array passed to incidence fitting,
  not only successful return.
  Include the default threshold-estimation path in the base installation,
  without MOSEK, and verify solver/options pass-through when explicitly supplied.

### `pvsystemprofiler/tilt_azimuth_study.py` and `pvsystemprofiler/estimator.py`

#### PROF-03 — Documented day selectors do not match their implementations [P2]

- **Locations:** `TiltAzimuthStudy.__init__`, approximately lines 60–95;
  `ConfigurationEstimator.estimate_orientation`, `estimate_all`, and
  `_cal_orientation_helper`, approximately lines 136–222.
- **Finding:** the study documents a numeric `[first day, last day]` range, but
  its constructor dispatch handles only `None`, `"full_year"`, and `"manual"`.
  A numeric list leaves `day_range_dict` uninitialized, while the `"manual"`
  branch stores that string as the interval. Conversely, the estimator
  documents `"all"`, `"clear"`, and `"cloudy"` selectors but compares day numbers
  with the first two characters of the supplied string as if they were numeric
  endpoints.
- **Owner decision:** use eligible clear-day observations across the full record
  in both orientation-estimation APIs. Retire the extra day selectors, including
  calendar intervals and all/cloudy modes.
- **Proposed correction:** remove `day_range` / `day_interval` selection and the
  study's seasonal-range sweeps. Preserve the applicable clear-day, daytime,
  clipping, and data-quality eligibility checks; using the full record does not
  admit every observation. Update signatures, examples, and docstrings, with an
  explicit migration for the retired named and numeric selector forms. Any
  compatibility entry point must identify an unsupported legacy request rather
  than silently ignore it. The earlier proposal to define interval inclusion,
  wrapping, and leap-day semantics is superseded by selector retirement.
- **Eligibility/state finding, follow-up review:** initialization excludes
  clipped days from the daily envelope, but `estimate_all` replaces `self.days`
  with clear-day flags alone and retains that changed mask on the reused
  estimator. Specify separate daily-envelope and sample-incidence masks:
  clipping can invalidate a day's maximum for the envelope while leaving
  other unclipped samples eligible for incidence fitting. Preserve the
  applicable quality checks for each quantity; full-record clear-day selection
  does not make their eligibility identical. Construct per-call masks without
  mutating the handler's flags or leaving a changed shared mask behind.
- **Proposed verification:** inspect the actual selected dates and final masks
  through both public APIs on multiyear records, including leap dates. Verify
  that eligible clear-day observations throughout the record can enter the fit,
  that the other eligibility checks remain effective, and that no former
  seasonal restriction remains. Cover explicit handling of retired selector
  arguments; a call that returns successfully is insufficient evidence of
  correct observation selection.
  Compare direct orientation estimation with `estimate_all` and repeated calls,
  including a day with some clipped samples. Verify the envelope mask and final
  incidence mask separately, together with preservation of caller-owned flags.

### `pvsystemprofiler/ground_truth_estimator.py`

The `np.float` failure and its suppression by a broad exception handler are
tracked under PKG-01. No duplicate implementation item is intended here.

## 6. Dask orchestration and data plugs

### `sdt_dask/dask_tool/runner.py`

#### DASK-01 — Caller pipeline options are discarded [P1]

- **Locations:** `set_up` accepts `**kwargs` near line 32; task construction near
  lines 186–188 supplies only hard-coded options.
- **Finding:** a specified `power_col`, solver, threshold, or `fix_shifts=False`
  never reaches the pipeline. A different channel can be analyzed from the one
  requested.
- **Proposed correction:** merge documented defaults with caller options and
  forward the resulting mapping, with caller values taking precedence.
- **Proposed verification:** a recording pipeline stub inside the delayed graph
  receives nondefault column, solver, and correction settings exactly.

#### DASK-02 — Result collection shuts down caller-owned infrastructure [P2]

- **Location:** `get_result`, approximately line 216.
- **Finding:** the runner accepts an existing client but unconditionally calls
  `client.shutdown()`, terminating the scheduler/workers rather than merely
  releasing its own work.
- **Owner decision:** the caller manages client and cluster cleanup. `Runner`
  never shuts down the supplied cluster, on success or failure; do not add a
  runner shutdown option.
- **Proposed correction:** remove the unconditional shutdown and document the
  caller's resource ownership. Result collection must leave the supplied client
  usable. Limit runner cleanup to resources associated with its own work, such
  as report files and references to completed computations. Update Dask examples
  and migration guidance to show explicit caller cleanup, including on failure,
  so users of dedicated cloud clusters know when to release them. The earlier
  proposal for configurable runner ownership/shutdown is superseded.
- **Proposed verification:** recording clients cover successful result collection
  and failures during computation or report writing. Verify that the runner
  neither closes the supplied client nor shuts down its cluster, and that it can
  be used for subsequent work. No live cloud infrastructure is required.

### `sdt_dask/dataplugs/S3Bucket_plug.py` and `sdt_dask/dataplugs/csv_plug.py`

#### S3-01 — CSV parsing does not implement the documented schema [P2]

- **Locations:** `_clean_data`, near line 49; `get_data` documentation near lines
  57–59; the related `LocalFiles.make_time_series` option and `_clean_data` in
  `csv_plug.py`; `sdt_dask/dataplugs/README.md` and Dask loader examples.
- **Finding:** a documented timestamp/power CSV is unconditionally sent through
  `make_time_series` with Cassandra-schema defaults (`ts`, `meas_val_f`,
  `meas_name`, `site`, `sensor`) and a default UTC−8 shift. Ordinary CSV input fails
  or receives an unintended time conversion.
- **Owner decision:** retire Cassandra-specific parsing from `S3Bucket`, the S3
  data plug used by the Dask runner. Support the documented timestamp/power CSV
  format; the earlier proposal to retain Cassandra parsing as an option is
  superseded.
- **Proposed correction:** replace the unconditional `make_time_series` call
  with explicit timestamp-column parsing and return a datetime-indexed power
  DataFrame suitable for the runner. Remove the Cassandra pivot and implicit
  UTC−8 shift. Specify accepted timestamp/timezone forms; any timezone conversion
  must be explicitly requested rather than inferred from the storage format.
  Update the loader documentation and examples, and provide migration guidance
  for legacy exports: convert them to the supported CSV format before ingestion
  or use a caller-supplied data plug. Reject unsupported schemas clearly rather
  than silently reinterpreting their columns. Apply the same simplification to
  the companion `LocalFiles` loader: retire its optional `make_time_series`
  path and Cassandra-specific description while preserving ordinary local CSV
  loading. Include that removed option in migration guidance. This cleanup
  concerns the built-in Dask data plugs; it does not propose removing the
  separately exported core `make_time_series` utility.
- **Proposed verification:** conventional timestamp/power CSV with explicit
  column configuration, exact timestamp and value preservation, no implicit
  eight-hour shift, and clear rejection of the retired Cassandra schema. Use
  stubbed S3 reads and local CSV fixtures; verify ordinary `LocalFiles` use and
  explicit handling of its retired option. No live cloud access is required.

### `sdt_dask/dataplugs/S3Bucket_plug.py`

#### S3-02 — Key enumeration stops after one response page [P2]

- **Location:** `_pull_keys`, approximately lines 79–85.
- **Finding:** one `list_objects_v2` call ignores continuation, silently omitting
  objects beyond the first page from the analysis cohort. The AWS contract permits
  at most 1,000 objects per response.
- **Proposed correction:** use pagination and retain the full intended key set.
- **Proposed verification:** stub multiple pages and an empty bucket; assert every
  key appears exactly once. No live AWS request is necessary.
- **Reference:** [AWS `list_objects_v2` contract](https://docs.aws.amazon.com/boto3/latest/reference/services/s3/client/list_objects_v2.html).

The missing `boto3` installation dependency is tracked under PKG-02.

## 7. Dependencies, distribution, and licensing

### `pyproject.toml` and affected NumPy call sites

#### PKG-01 — Shipped NumPy API usage is internally incompatible [P2]

- **Locations:** `solardatatools/sensor_identification.py:43`;
  `anomalydetector/detector_model.py:372` and other `np.concat` calls;
  `pvsystemprofiler/ground_truth_estimator.py:15–19,97–107`;
  NumPy dependency and Dask extra in `pyproject.toml`, and associated Docker pins.
- **Finding:** `np.alltrue` was removed in NumPy 2, `np.concat` was added in NumPy
  2, and `np.float` was removed in NumPy 1.24. Selecting one NumPy version cannot
  make all these paths work. The KML path suppresses the resulting exception and
  returns NaN summaries for valid polygons.
- **Owner decision, 2026-10-22:** require NumPy 2.x for the refactored package;
  do not retain NumPy 1.26 support. Exact version bounds remain subject to
  validation with the supported Python and dependency combinations.
- **Revised disposition:** `np.concat` is valid in the selected NumPy major
  version and no longer constitutes a compatibility defect. Replacing it with
  `np.concatenate` is not required by this issue. The removed `np.alltrue` and
  `np.float` uses still require correction.
- **Proposed correction:** use `np.all` and `float`; narrow KML exception handling
  and report when no usable polygons remain. Declare the validated NumPy 2.x
  range and document retirement of 1.x support. Reconcile the Dask extra's
  `numpy==2.0` pin and associated Docker requirements with the supported Python
  and dependency combinations, rather than assume that one exact NumPy release
  covers the entire advertised Python range. Align installation documentation,
  CI environments, and migration guidance with the resulting support matrix.
- **Proposed verification:** sensor construction, outage residual operations,
  and valid KML parsing under the declared NumPy 2.x boundaries. Verify resolution
  and representative imports for the base package and Dask extra on supported
  Python combinations. Confirm that package metadata enforces the chosen range;
  no NumPy 1.x compatibility suite is required. Fresh environments and dependency
  changes remain proposed work requiring scoped authorization.
- **References:** [NumPy 2 migration guide](https://numpy.org/doc/2.0/numpy_2_0_migration_guide.html),
  [NumPy 2 additions](https://numpy.org/doc/2.0/release/2.0.0-notes.html#array-api-compatible-functions-aliases),
  [NumPy 1.24 expired deprecations](https://numpy.org/doc/1.24/release/1.24.0-notes.html#expired-deprecations).

#### PKG-02 — The documented Dask extra omits an S3 import dependency [P2]

- **Locations:** optional dependencies around `pyproject.toml:77`;
  `sdt_dask/dataplugs/S3Bucket_plug.py:4`; `docker/Dockerfile`.
- **Finding:** the S3 module imports `boto3`, but the documented extra does not
  supply it. The Dockerfile installs it separately, masking the omission in that
  environment. Inspection of the dependency metadata did not establish a mandatory
  transitive installation through the existing extra.
- **Proposed correction:** explicitly declare the dependency in the appropriate
  supported extra and align installation documentation.
- **Proposed verification:** fresh installation of that extra followed by an S3
  module import, with no AWS calls.

#### PKG-03 — Native quantile functionality replaces spcqe alongside OSD retirement [refactoring decision]

- **Locations:** `pyproject.toml:33–37`; `_osd_signal_decompositions.py`;
  `algorithms/loss_factor_analysis.py`; `spcqe` imports in
  `_cvx_signal_decompositions.py`, `algorithms/quantile_estimation.py`, and
  `anomalydetector/detector_model.py`.
- **Finding:** removing the dispatch implementation alone leaves direct `gfosd`
  use in loss-factor analysis. Removing only the direct `sig-decomp` and `qss`
  requirements also leaves a dependency path through the inspected local
  `spcqe` 0.3.0: its metadata requires `sig-decomp`, and its solver imports include
  `gfosd`. Quantile estimation and anomaly detection use more of `spcqe` than
  the Fourier matrix helpers used by the capacity model.
- **Owner decision, 2026-10-22:** move the required quantile functionality into
  native CVXPY code inside SDT. Replace the `spcqe` dependency rather than wait
  for an upstream OSD-free release. This decision covers functionality used by
  this repository, not reproduction of the complete upstream package API.
- **Required functionality from current call sites:**
  - `PVQuantiles.estimate_quantiles` uses periodic quantile fitting with daily
    and annual Fourier terms, configured harmonic counts and regularization,
    `standing_wave=[True, False]`, `trend=False`, sequential fitting, and
    extraction of the fitted quantile matrix. Retain its dilation/undilation
    interfaces and explicit quantile ordering.
  - `OutagePipeline.fit_quantiles`, `predict_quantiles`, and `test_quantiles`
    additionally use the fitted quantile transform, including `extrapolate="solar"`
    and evaluation at explicit time indices relative to the training origin.
    Plotting also reads the fitted quantile matrix. Moving only the fitting
    optimization would leave these required transform and evaluation paths
    unresolved.
  - The capacity model uses `make_basis_matrix` and
    `make_regularization_matrix`. Supply the needed native Fourier and
    regularization helpers while preserving SD-05's level convention and
    effective weight scaling.
- **Upstream source baseline, inspected 2026-11-11:** the local `spcqe` 0.3.0
  implementation in `sdt-dev` establishes the following behavior; these are
  contracts to specify, not an instruction to redesign the quantile estimator.
  - `solve_cvx_sequential` solves the requested quantiles independently and
    sorts fitted values within each time row afterward. It does not impose
    coupled noncrossing constraints in that mode.
  - The Fourier/standing-wave basis includes pairwise products between the
    period-specific bases, including daily-by-annual interactions. Replacing
    it with additive daily and annual terms alone would change the model.
  - The estimator fits on standardized values by default, inverse-transforms
    the fitted quantiles, and least-squares refits basis coefficients to the
    sorted curves. Explicit-time prediction evaluates those coefficients and
    sorts the resulting rows. Stored fitted curves and explicit-time evaluation
    on the training grid need not coincide after that coefficient projection.
    Record both paths and their use by each SDT consumer before replacement.
- **Ordering and signed-time defects:** `PVQuantiles.estimate_quantiles`
  assigns sorted output columns to levels in caller order; unsorted requested
  levels can therefore be mislabeled. Canonicalize level/column associations
  explicitly. An anomaly test partition can begin before the training origin,
  yielding negative offsets from `full_signal`. Upstream explicit-time
  evaluation indexes an extended basis array, so some negative offsets wrap
  into its end instead of evaluating earlier physical times. Evaluate the
  native basis at signed time coordinates relative to the fixed training
  origin; do not silently move that origin or use negative array indexing as
  extrapolation. Preserve the intended daily and annual phases on either side.
- **Proposed correction:** use one shared native implementation for PV quantiles
  and anomaly detection, with the small-helper organization in SD-06. Preserve
  the current mathematical fitting and transformation semantics except for
  explicitly agreed changes; dependency removal does not authorize a new
  quantile estimator. Complete the specification of loss/regularization
  normalization, sequential fitting and rearrangement, coefficient projection,
  transform interpolation and tail extrapolation, and tied-quantile handling
  using the source baseline above. A joint noncrossing fit is not a consequence
  of dependency removal and would require a separate model proposal.
  Apply the agreed annual period $P=365.2425$ in dilated coordinates; document
  the correction from the current call sites' `365.24225` multiplier. Preserve
  observation masks and calendar origins across fitting and later evaluation.
  Support appropriate CVXPY solvers and reusable parameterized problems; no
  solver selection may silently select another quantile model.
  Coordinate SD-01 and LFA-06 with removal of `spcqe`, `sig-decomp`, and `qss`
  dependencies and obsolete imports. Verify the resolved dependency graph.
  Document replacements for exposed `spq_object`/`spqs` objects and obsolete
  configuration in the migration guide. Retain applicable attribution/licenses
  for any incorporated upstream code. Update API pages, examples, `CLAUDE.md`,
  and solver guidance when the implementation changes.
- **Saved-model migration decision, 2026-11-11:** the public module-level `save`/`load`
  functions in `anomalydetector/detector_model.py` (around line 888) serialize
  whole fitted pipelines, including `spcqe` instances. Existing artifacts
  therefore depend on the old modules and their fitted-state semantics.
  Require retraining for the native SDT version; retain the old environments
  for existing artifacts. Document this migration and explicit artifact/version
  provenance.
  Do not promise automatic cross-version unpickling or retain the obsolete
  runtime dependency solely to imply compatibility. An artifact converter is
  outside the agreed port; proposing one later would require separate scope
  and validation.
- **Proposed verification:** inspect resolved metadata and imports for the agreed
  distribution, then exercise the native decomposition, loss-factor, quantile,
  and anomaly entry points in an authorized installed-package check. Preserve
  optional MOSEK installation and support for other appropriate CVXPY solvers.
  Independently check native objectives and basis/weight semantics against the
  recorded specifications, using reference-output comparisons only as additional
  evidence. Cover multiple quantile levels, masks/gaps, daily and annual phases,
  transform interpolation/extrapolation, and later evaluation at the training
  calendar origin. Check both consumers against the same native implementation,
  fresh builds versus parameter updates, and the complete anomaly fit/predict
  lifecycle. Keep the QUANT-01 plotting fix and existing anomaly findings in
  scope at their owning locations; migration must not preserve their defects.
  Include unsorted requested quantiles, interaction-basis terms, row sorting
  and coefficient projection, and evaluation before and after the training
  origin. Compare stored-curve and explicit-time paths according to the specified
  contract. Through the installed public outage API, fit, save, load, and
  predict with a new native artifact; check prediction values, masks, and
  provenance. Legacy-artifact handling must follow the agreed migration policy.

### `solardatatools/circular_statistics.py`, `third-party-licenses/`, and packaging metadata

#### LICENSE-01 — Incorporated Astropy code lacks its upstream redistribution notice [P2]

- **Locations:** `circular_statistics.py:17–18`; `third-party-licenses/LICENSE`;
  license inclusion in distribution metadata.
- **Finding:** the source identifies incorporated Astropy BSD-3 code, but the
  repository review did not find the corresponding upstream copyright/license
  notice. The existing third-party license file contains SLAC's notice.
- **Proposed correction:** establish the incorporated source's provenance and
  restore its applicable notice/license, preserving source attribution. Configure
  inclusion in both source and wheel distributions.
- **Proposed verification:** inspect built archive contents for the applicable
  upstream notice; do not infer inclusion merely from its presence in the checkout.
- **Reference:** [Astropy upstream license](https://raw.githubusercontent.com/astropy/astropy/main/LICENSE.rst).

## 8. Verification and documentation follow-through

These items distinguish defects in verification configuration from evidence that
has not yet been collected. They are not additional authorization to run builds,
tests, installations, or experiments.

### `.github/workflows/test.yml`

#### QA-01 — Coverage targets a nonexistent source directory

- **Location:** approximately line 60, `--cov=src`.
- **Finding:** the repository contains four top-level packages and no `src`
  directory. The configured target does not establish their coverage.
- **Proposed correction:** measure the actual packages and detect a no-data
  coverage result. Do not update a published coverage percentage without a
  corresponding measurement.
- **Proposed verification:** inspect the coverage report's measured files and
  ensure all intended packages are represented.

### Distribution validation

#### QA-02 — Installed-wheel behavior remains unverified

- **Evidence gap:** building distributions does not establish that the resulting
  wheel imports and serves public entry points outside the source checkout.
  Missing package assets were not established by this review.
- **Proposed verification:** when authorized, install the built wheel in an
  isolated environment, run outside the checkout, and exercise representative
  imports, packaged resources, optional extras, and license contents. Coordinate
  with PKG-01, PKG-02, OUT-02, SHADE-02, and LICENSE-01.
  Include PKG-03's fitted outage-model save/load/predict lifecycle with the
  native quantile implementation, and PROF-02's default threshold-estimation
  path on the base installation. Source-checkout imports and constructor-only
  checks do not exercise these contracts.

### `tests/` and public documentation

#### QA-03 — Focused contract coverage is missing for the reported failures

- Existing anomaly tests exercise quantile transformation and linear coefficients,
  but not the full fitting/prediction lifecycle or exported split.
- Include numerical failure-result propagation, independent-stage continuation,
  dependency skips, and partial `DataHandler`/Dask reports (DH-07). A returning
  pipeline call must not be treated as evidence that every stage succeeded.
- Bundt tests exercise the lower-level renderer rather than the public handler
  wrapper's calendar aggregation and initialization (DH-05/DH-06). Include
  explicit preprocessing prerequisites and full-record clear-day eligibility
  checks for profiling (PROF-02/PROF-03), and
  unavailable-subset/failed-fold results for sensor identification (SENSOR-02).
- Historical numerical fixtures do not independently establish objective
  semantics, observation-mask preservation, unit invariance, correct state
  transformations, or plotting purity. The follow-up checks include timezone
  warning behavior and preservation of alignment across views (DH-04),
  raw-handler mask coordinates after splitting
  (OUT-05), dimension-changing regression overrides (OUT-06), quality-score
  semantics (QUALITY-01), and zero/signed shading components (SHADE-01).
- No Dask test suite was identified in the review. Option forwarding, pagination,
  schema parsing, and cluster ownership can be tested with local substitutes;
  live cloud execution is not required for those contracts.
- The proposed checks attached to each issue are the starting point for an
  approved verification scope, not a request for exhaustive testing or replacement
  of all existing fixtures.
- Update user-facing documentation alongside any approved contract change:
  units/cadence, solver-option support, uncertainty terminology, minimum data
  duration, optional dependencies, S3 schema/timezone handling, cluster
  ownership, orientation-selector retirement, and sampling-budget/termination
  semantics (LFA-10).
  Resolve the intended behavior before revising examples.

### `tests/solardatatools/test_signal_decompositions.py`, `test_cvx_signal_decompositions.py`, and algorithm tests

#### QA-04 — Native decomposition verification must establish mathematical contracts

- **Evidence gap:** the existing decomposition tests largely compare stored
  numerical outputs. They do not establish that solver selection preserves the
  model, that parameter updates trigger correct re-solves, or that structural
  choices recover the intended components. Tests mentioning cardinality or
  alternate backends can preserve obsolete behavior even when those options are
  silently ignored.
- **Proposed verification scope:**
  - Compare objective values and feasibility for the same convex formulation
    across CLARABEL, MOSEK when available, and other solvers within the agreed
    support scope. Do not require identical nonunique component solutions.
  - Verify that an omitted solver selects CLARABEL without hidden solver-setting
    overrides, and that explicit solver selection and solver-specific options
    reach every applicable solve unchanged. Cover repeated solves, refinement,
    candidate search, sensitivity sampling, and higher-level pipeline calls.
  - Check public weight effects, observation exclusion, finite data handling,
    role-based extraction, and solver-status handling independently of fixtures.
  - Verify DCP/DPP for reusable builders and compare updated-parameter solves
    with fresh constructions of the same model, including changing masks.
  - Evaluate the fixed IRL1 merit function independently, check decrease within
    numerical tolerance, and verify initialization/reset behavior across a
    weight path. Do not infer support correctness from subproblem termination.
  - Check first/interior/last breakpoint indices, empty-support/constant-component
    behavior, and retention of the residual loss during one fixed-support refit.
    An extra null-model solve is needed only where the agreed selector requires it.
  - Verify constant-reference endpoint checks and bounded weight extension for
    capacity and time shifts, including reference-error provenance and failure
    outcomes, as specified in CAP-02/SHIFT-01.
  - Include the short-record soiling baseline case and repeated clipping/shading
    fits. Review representative structural ambiguities at the component level,
    including the actual time corrections and capacity labels produced.
  - Verify the degradation conversion, chosen chord estimand, reporting interval,
    and nonlinear fixed-rate/tail policies in LFA-01, LFA-08, and LFA-09. Use
    analytic reference trajectories shared with the RdTools comparison, retain
    separately named annual-change diagnostics, and preserve distinct Monte Carlo
    and final fitted-rate results across repeated calls (LFA-04).
- **Compatibility decision:** historical fixtures may remain useful evidence,
  but accepting changed outputs requires an agreed model change. Replacing
  fixture values is not independent validation. Remove tests that exist solely
  to preserve retired OSD/QSS behavior after defining public retirement semantics.
- **Performance evidence:** the source-derived counts in SD-06 identify repeated
  work, not measured savings. Benchmark the agreed models only after correctness
  checks, recording graph construction/canonicalization separately from solving.

## 9. Issue decision and completion record

Active issues and corrections remain **proposed; implementation unapproved**
unless an issue-specific record below states otherwise. SD-02's standalone OSD
repair is superseded by the recorded retirement direction; its indexing example
remains relevant to the replacement. Use the following record for a selected
item, adding entries only as decisions are made. The status of one issue does
not alter another issue's authorization.

```text
Issue ID(s):
Date and current revision:
Owner disposition: pending / accepted / revised / rejected / deferred
Agreed mathematical or API contract:
Authorized work and boundaries:
Open questions:
Changes made, if authorized:
Verification actually performed and results:
Remaining limitations:
Human review outcome:
```

Completion of an individual item requires the agreed correction (if any), the
agreed evidence, and human review. Rejected or deferred findings remain recorded
with their rationale rather than being silently removed. Closing an item or
finishing this plan does not certify excluded behavior or a higher assurance
register.

### 2026-10-21 — Focused decomposition review incorporated

- **Owner direction:** native CVXPY implementations in this project; retire OSD;
  support appropriate solvers including CLARABEL and MOSEK; emphasize
  simplification, IRL1, Tier-3 selection, and reusable parameterized problems.
- **Plan disposition:** revise SD-01 and SD-03, supersede the standalone SD-02
  repair, and incorporate the location-specific findings and proposals from the
  focused re-review. Mathematical choices such as the seasonal representation,
  penalty normalization, and structural selection rules remain open.
- **Authorized work completed:** creation and subsequent revision of this working
  plan. The reviews used source inspection and algebraic reasoning. No package
  implementation, test execution, numerical solve, benchmark, build, dependency,
  or infrastructure change is recorded as completed here.
- **Implementation boundary:** issue-specific direction and substantial human
  input remain required. Updating this document does not approve its proposed
  corrections or verification work.

### 2026-10-21 — Degradation-rate alignment with RdTools incorporated

- **Owner direction:** move toward a unified degradation-rate approach with the
  adjacent RdTools project. The supplied review was checked against both local
  sources; the RdTools revision and inspection limits are recorded above.
- **Plan disposition:** narrow LFA-01 to consistent annual-rate conversion and
  add LFA-08 for the proposed compound endpoint chord as the overall rate. Median
  annual change may remain as a separately named diagnostic. The proposed common
  year is 365.2425 days. Record the estimand and reporting-domain choice before
  implementation; this plan update does not itself finalize those choices.
- **Open decisions:** the nonlinear flat-tail assumption, reporting interval and
  missing-endpoint policy, nonlinear fixed-rate behavior, and public exposure of
  Monte Carlo summaries versus final fitted rates. LFA-09 and the revised LFA-04
  make these separate from the conversion fix. LFA-02 sensitivity terminology and
  LFA-03/LFA-04 repeatability corrections remain applicable.
- **Work completed:** plan revision and read-only source comparison. No source
  edits in either package, test execution, or numerical solves were performed
  for this update. Implementation and verification remain subject to
  issue-specific direction and substantial human input.

### 2026-10-21 — Follow-up public-API review incorporated

- **Evidence:** checked the supplied review against the recorded repository
  revision and the relevant existing tests. Findings are based on source
  inspection, not completed runtime reproductions.
- **New issues:** DH-04 for final timezone-correction consistency; DH-05/DH-06
  for Bundt calendar aggregation and initialization; QUALITY-01 for the dataset
  score contract; SENSOR-02 for unavailable subsets and failed folds; OUT-06
  for dimension-dependent regression weights; PROF-03 for day selectors; and
  LFA-10 for the sampling-budget and termination contract.
- **Refinements:** extend TIME-02 to exported `make_2d`, OUT-05 to original
  handler mask coordinates, SHADE-01 to zero and signed components, and PROF-02
  to initially unprocessed handlers. Strengthen SD-01/SD-03 with component-level
  argument meanings and preservation of structural constraints during refitting.
  Cross-reference LFA-10 from LFA-03/LFA-07 and expand QA-03 accordingly.
- **Disposition:** retain these as proposed issues and corrections, with API and
  modeling choices identified explicitly. Existing degradation-rate alignment
  proposals remain intact. Only this working document was edited; no package
  source, tests, dependencies, or sibling-project files were changed, and no
  tests, numerical solves, installations, or builds were run for this update.
  Implementation continues to require issue-specific direction and substantial
  human input.

### 2026-10-21 — First decomposition design decisions recorded

- **Issue IDs:** SD-01, SD-04, SD-05, and SHIFT-01; related refitting work in SD-03.
- **Owner disposition:** accept Fourier seasonality with physical period
  $P=365.2425$ days as the common default, allowing explicit model-specific
  exceptions. Harmonic counts, regularization, and intercept allocation remain
  to be specified.
- **Owner disposition:** retain exact 365-day recurrence of the clock-shift
  component as an explicit optional model assumption, separate from seasonal
  periodicity. Selection policy and public compatibility details remain open.
- **Owner disposition:** use IRL1 as the standard step-selection procedure for
  both time shifts and capacity changes. The convex-subproblem/nonconvex-procedure
  distinction in SD-04 remains part of the mathematical specification.
- **Remaining questions:** objective and penalty normalization, seasonal level
  allocation, harmonic count/regularization, IRL1 initialization and smoothing
  scale, stopping criteria, structural-weight/support selection, fixed-support
  refitting, and public API compatibility.
- **Authorized work and outcome:** record these design choices in the working
  plan and continue resolving questions. No implementation, tests, numerical
  solves, or dependency changes were authorized or performed by this exchange.

### 2026-10-21 — Component roles, retained normalization, and refitting agreed

- **Issue IDs:** SD-01, SD-03 through SD-06, and SHIFT-01; downstream time-shift
  diagnostics in `DataHandler` and capacity-component extraction are affected.
- **Owner disposition:** for time shifts, the Fourier seasonal component carries
  the offset and is the solar-noon estimator. For capacity changes, the PWC
  component carries the offset and is the local-capacity estimator. The remaining
  time-shift reference-level convention must be resolved explicitly.
- **Owner disposition:** retain existing model choices and document them;
  broader model updates are future work. In particular, preserve the current
  residual loss families and normalization rather than adopting a new common
  scaling convention. This does not rescind the explicit design choices already
  accepted in this record.
- **Owner disposition:** use one convex fixed-support refit for both time shifts
  and capacity changes, removing the jump penalty while retaining the remaining
  objective and constraints. Support selection remains to be specified.
- **Integration consequences:** translate time-shift extraction and diagnostics
  that assume an absolute PWC level or a zero-offset seasonal deviation. Record
  the extra capacity refit in reuse/performance planning. Preserve effective
  penalty coefficients during algebraic factoring, including existing squared
  weight dependence.
- **Authorized work and outcome:** only the working plan was revised. No source
  edits, tests, numerical solves, or dependency changes were authorized or
  performed by this exchange.

### 2026-10-21 — Time-shift reference and breaking API direction agreed

- **Issue IDs:** SD-01, SD-03, SD-06, and SHIFT-01; DH-04 remains a separate
  overall timezone-correction issue.
- **Owner disposition:** set the first time-shift PWC segment to zero. The
  Fourier component carries the solar-noon offset relative to that initial
  clock regime. This replaces the previous reference-clock heuristic for
  relative corrections and applies during discovery and fixed-support refitting.
- **Owner disposition:** allow a cleaner breaking public API with an explicit
  migration guide. Specify replacement calls, outputs and component roles,
  solver options, and retirement of obsolete arguments before implementation.
- **Remaining questions:** IRL1 numerical policy, jump-support selection,
  structural-parameter selection, and concrete API/result design. Broader
  unapproved model changes remain future work.
- **Authorized work and outcome:** only this working document was edited.
  No source changes, tests, numerical solves, or dependency work were authorized
  or performed by this exchange.

### 2026-10-21 — IRL1 stopping, capacity support, and Tier-3 design scope agreed

- **Issue IDs:** SD-03/SD-04, CAP-01/CAP-02, and SHIFT-01.
- **Owner disposition:** retain each estimator's existing $\epsilon=0.1$ and
  use tolerance-based stopping on the fixed IRL1 log-sum merit function with a
  hard iteration cap. Specify the test, tolerances, cap, and termination reporting
  before implementation.
- **Owner disposition:** define capacity support by final IRL1 first differences
  exceeding 0.001 log units; use that support for the convex refit and form
  capacity labels afterward. Label bins and physical/model-selection thresholds
  have separate meanings.
- **Scope interpretation subsequently corrected:** this entry originally
  extended replacement selection design to both estimators. The owner clarified
  that capacity selection is the intended archetype and should be preserved;
  the replacement Tier-3 design applies to time shifts. The clarification below
  supersedes that broader interpretation.
- **Authorized work and outcome:** record decisions and continue specification.
  Only this working plan was edited. Required future validation is not an
  authorization to run tests or experiments now; no code, dependency, or
  numerical-execution changes were made.

### 2026-10-21 — Tier-3 scope corrected to follow the capacity archetype

- **Owner clarification:** the capacity change detector is well designed and
  supplies the archetype for the skill's Tier-3 discussion. It may be reviewed
  for focused issues; sweeping selection changes are outside the requested scope.
  Time-shift detection is the estimator requiring an update to Tier-3 selection.
- **Plan correction:** retain capacity's component-based screening and smallest
  qualifying-weight choice. Narrow CAP-01 to metric naming and unused split
  cleanup, and CAP-02 to the reference/fallback description and behavior. Use
  this design as the starting point for time-shift screening, with its physical
  thresholds still requiring owner input and validation.
- **Withdrawn proposals:** the new abstention/ambiguity machinery, stability as
  a primary selection framework, and a separate refit of every candidate support.
  None was adopted. The earlier single final fixed-support refit, IRL1 numerical
  decisions, component roles, and API migration direction remain in effect.
- **Evidence:** reread the skill's Tier-3 guidance and inspected the capacity
  screening code. The skill calls for judging the component rather than tuning
  its structural parameter by reconstruction holdout error. The withdrawn
  framework was an additional proposal, not a requirement of that guidance.
- **Authorized work and outcome:** only the working plan was corrected. No
  source changes, tests, numerical solves, installations, or builds were performed.

### 2026-10-21 — Finite TV threshold and capacity-grid assumption clarified

- **Owner correction:** there is a finite critical TV weight above which the
  capacity component has no changes. The current implementation does not
  calculate the threshold and assumes its weight grid extends far enough; the
  owner expects that it does.
- **Plan correction:** CAP-02 now concerns checking or certifying that endpoint,
  not requiring a separately constrained no-change model. No inadequate grid
  endpoint has been demonstrated. Record the KKT certificate and a conservative
  inexpensive bound with the current objective normalization; their numerical
  verification and any implementation remain future, issue-specific work.
- **Scope:** preserve the capacity selector and the previously agreed algorithmic
  changes. Only this working document was revised; no tests or numerical solves
  were performed.

### 2026-10-21 — Minimum detectable clock shift and correction rounding specified

- **Owner decision:** five minutes ($1/12$ hour) is the minimum detectable clock
  shift. Record this in SHIFT-01 and SD-03 for meaningful-jump screening and
  pooled support extraction, superseding the existing three-minute cutoff.
- **Owner decision:** default to nearest-sample correction rounding and keep
  whole-hour rounding optional. Apply the default consistently across public
  entry points and document it in the migration guide. Detection uses the
  unrounded fitted component; the applied correction is limited by sample
  cadence.
- **Scope:** only this working plan was edited. No source changes, tests, or
  numerical solves were performed.

### 2026-10-21 — Clock-change frequency bound specified for validation

- **Owner direction:** very large clock-change counts usually indicate a
  problem. Use ten changes per year as a proposed safe screening bound in
  SHIFT-01, subject to thorough testing at implementation time.
- **Validation scope:** all algorithm changes require thorough implementation
  testing. Recording the proposed bound does not establish its adequacy or
  authorize implementation or test execution in this planning discussion.
- **Scope:** only this working plan was edited; no code changes or numerical
  experiments were performed.

### 2026-10-21 — Time-shift fit-improvement screen and fallback agreed

- **Owner decision:** use a 2% minimum in-sample fit-improvement screen relative
  to a constant-shift fit, measured with the existing squared residual loss and
  subject to validation. Together with the five-minute minimum jump and the
  proposed ten-changes-per-year bound, this defines the candidate screens for
  choosing the smallest qualifying weight.
- **Owner decision:** if no candidate qualifies, return a constant shift
  component, hence $c=0$ and no relative clock correction under the agreed
  first-segment reference. Apply the same fallback when the constant-shift
  reference fit has zero residual error.
- **Scope:** only this working plan was edited. No implementation or numerical
  validation was performed.

### 2026-10-21 — Shift recurrence made opt-in; recommendation flag proposed

- **Owner decision:** exact 365-day clock-shift recurrence is explicitly opt-in,
  default off. Remove the automatic periodic retry in the pipeline. Revise
  DH-02 accordingly instead of preserving and repairing that retry path.
- **Owner direction:** test a recommendation flag during implementation. The
  flag advises consideration of recurrence without enabling it automatically;
  its trigger and effectiveness require evaluation. Any seasonal-RMS diagnostic
  must exclude the Fourier offset.
- **Scope:** only this working plan was edited. No implementation, tests, or
  numerical experiments were performed.

### 2026-10-22 — IRL1 iteration-limit behavior agreed

- **Owner decision:** when successful convex solves reach the IRL1 iteration
  cap without meeting the stopping tolerance, use the last iterate through
  selection/refitting and report that the cap was reached. Preserve this
  termination information in the final result; do not report convergence.
- **Remaining numerical specification:** stopping tolerances, the cap value,
  iteration counting, and treatment of numerical merit increases still require
  specification and implementation-time validation.
- **Scope:** only this working plan was edited. No code changes, tests, or
  numerical experiments were performed.

### 2026-10-22 — Degradation endpoint chord adopted; median comparison retained

- **Owner decision:** report the compound endpoint chord as the overall
  degradation rate and retain median annual change as a separately labeled
  comparison. Compute both from the same fitted component and reporting interval;
  their difference for nonlinear trajectories is meaningful, not an extraction
  inconsistency.
- **Related decisions:** the separate decisions below remove the forced flat
  tail, select the admitted-observation span for rate reporting, reject fixed-rate
  input for nonlinear models, and set the common annualization convention.
  Changing the estimand is distinct from repairing the inverse rate conversion
  in LFA-01.
- **Scope:** only this working plan was edited. No source changes, tests, or
  numerical experiments were performed.

### 2026-10-22 — Nonlinear degradation tail constraint removed from the design

- **Owner decision:** remove the forced flat tail while retaining monotonicity
  and smoothness. Preserve the existing initial-value anchor; the nonlinear
  trend extends over the full grid. The reporting interval is specified in the
  subsequent decision below; the later fixed-rate decision rejects that input
  for nonlinear models.
- **Source check:** the current RdTools monotone trend uses a non-increasing
  constraint and squared second-difference penalty, with its optional end-drop
  penalty off by default. Record that component-level alignment in LFA-09;
  agreement of complete fitted decompositions is not implied.
- **Scope:** only this working plan was edited. The RdTools source was inspected
  read-only. No code changes, tests, or numerical experiments were performed.

### 2026-10-22 — Degradation reporting interval specified

- **Owner decision:** use the first-to-last admitted-observation span for
  degradation-rate reporting. Determine its endpoints from the base fit mask
  before Monte Carlo subsampling and hold them fixed across draws.
- **Extraction contract:** both the overall chord and median annual-change
  comparison use fitted trend values over that span. Leading/trailing grid
  points outside the span are excluded; internal gaps retain their elapsed time.
  Report the interval and duration explicitly.
- **Scope:** only this working plan was edited. No source changes, tests, or
  numerical experiments were performed.

### 2026-10-22 — Nonlinear fixed-rate input rejected

- **Owner decision:** reject supplied fixed-rate input for nonlinear degradation
  models. Continue to support freely fitted nonlinear trends with `deg_rate=None`.
- **Final-fit consequence:** a Monte Carlo summary must not be implicitly
  supplied as a fixed rate to the final nonlinear fit. Preserve the summary and
  final fitted chord as distinct results, alongside the separate within-fit
  median annual-change comparison.
- **Scope:** only this working plan was edited. No source changes, tests, or
  numerical experiments were performed.

### 2026-10-22 — Degradation annualization convention agreed

- **Owner decision:** use $P=365.2425$ days consistently in fixed-rate
  conversion and degradation-rate extraction and reporting. Retain the
  median comparison's 365-day lag, annualizing each log change by $P/365$
  before conversion to a percentage change.
- **Consistency:** apply the same convention to the overall chord, the median
  annual-change comparison, and all Monte Carlo rate draws and summaries.
  Agreement of these summaries for linear log trends is a verification case;
  nonlinear summaries can legitimately differ.
- **Scope:** only this working plan was edited. No source changes, tests, or
  numerical experiments were performed.

### 2026-10-22 — Direct RdTools soiling formulation adopted for loss-factor analysis

- **Owner decision:** adopt the improved, lower-memory RdTools soiling convention
  with appropriate testing. Replace the sawtooth dictionary with a nonpositive
  log-soiling signal, an unsquared L2 penalty on downward increments, and an L1
  magnitude penalty, as specified in LFA-06.
- **Validation:** evaluate component recovery and attribution, parameterized
  solve reuse, and measured peak memory and runtime at implementation time.
  This is a model change; equality with dictionary-model outputs is not required.
- **Subsequent decision:** adopt RdTools' associated component-based weight
  selection and post-selection IRL1 refinement in place of random stiffness
  perturbations, preserving SDT's quantile loss. Select once per estimation run
  and hold the model and selected weight fixed across Monte Carlo draws.
- **Scope:** only this working plan was edited. RdTools source was inspected
  read-only; no implementation, tests, or memory/runtime experiments were run.

### 2026-10-22 — RdTools soiling workflow adopted with SDT quantile loss preserved

- **Owner decision:** adopt RdTools' component-based soiling weight selection
  and two cleaning-interval IRL1 refinement solves, with LFA validation, in place
  of random soiling-stiffness perturbations.
- **Owner requirement:** retain the quantile loss configuration in SDT. Preserve
  its configured quantile level, loss normalization, and existing quantile-level
  perturbations; validate transferred soiling weights and screening criteria
  under that loss.
- **Subsequent decision:** run soiling selection once and hold the selected
  model and weight fixed across sensitivity draws. The two post-selection
  refinements remain part of each selected-soiling draw.
- **Scope:** only this working plan was edited. Source inspection was read-only;
  no implementation, tests, or numerical experiments were performed.

### 2026-10-22 — Soiling selection held fixed across sensitivity draws

- **Owner decision:** select the soiling model and weight once at the configured
  quantile and base observation mask, then hold them fixed across the run's
  sensitivity draws. Repeat the two refinement solves within each draw that
  includes soiling, recomputing its interval weights from its own fitted signal.
- **Integration:** retain SDT's quantile/subsample sensitivity procedure. The
  resulting summaries are conditional on the selected model and weight. A new
  independent estimation run performs fresh selection; problem reuse must not
  introduce stale selection or adaptive weights.
- **Scope:** only this working plan was edited. No implementation, tests, or
  numerical experiments were performed.

### 2026-10-22 — Missing-energy outage-accounting convention clarified

- **Owner clarification:** the established treatment counts missing-energy gaps
  as energy not produced relative to baseline and includes them in the outage
  factor used for Shapley attribution. Preserve that convention in LFA-05.
- **Separation of roles:** retain baseline energy on gap dates and account for
  zero production there, while excluding missing observations from the log fit.
  Preserve the distinction between missing and measured-zero inputs and the
  existing Shapley allocation of interactions. Document the accounting meaning
  of the outage bucket; temporary sampling exclusions are not physical gaps.
- **Scope:** only this working plan was edited. No source changes, tests, or
  numerical experiments were performed.

### 2026-10-22 — Positive production retained on outage-flagged days

- **Owner decision:** retain positive reported energy in totals for days marked
  with an outage flag, while excluding those dates from fitting. Keep this
  separate from the missing-energy convention, which accounts for zero
  production on genuinely unobserved dates.
- **Attribution consequence:** replace binary deletion for partially productive
  flagged dates with a factor construction that reproduces their retained
  production and preserves Shapley additivity. The subsequent decision below
  resolves the factor construction and treatment of ratios above one in LFA-05.
- **Scope:** only this working plan was edited. No source changes, tests, or
  numerical experiments were performed.

### 2026-10-22 — Bounded outage factors with residual excess production

- **Owner decision:** on flagged days use $o_t=\min\{1,E_t/q_t\}$ and
  $w_t=\max\{1,E_t/q_t\}$, where $q_t>0$ is fitted production before weather
  and outages. Retain $o_t=0$, $w_t=1$ on missing days.
- **Accounting:** the combined factors reproduce reported production, preserve
  Shapley additivity, and keep outage contributions nonpositive. Shortfalls on
  flagged dates enter the outage factor by convention; this does not identify
  their physical cause. Flagged observations remain excluded from fitting.
- **Scope:** only this working plan was edited. No implementation, tests, or
  numerical experiments were performed.

### 2026-10-22 — Negative production is a data-validity failure

- **Owner clarification:** negative daily production is nonphysical in this
  application and belongs to upstream SDT data validation and processing.
  Remove its treatment from the open accounting questions in LFA-05; the
  suggested conversion to missing energy or outage accounting is withdrawn.
- **Scope:** only this working plan was edited. No implementation or tests
  were performed.

### 2026-10-22 — Standalone soiling shares the native LFA formulation

- **Owner decision:** share the agreed native LFA formulation and soiling
  selection/refinement workflow through a thin standalone interface, retaining
  SDT's quantile loss.
- **Disposition:** SOIL-01 through SOIL-04 now specify replacement of the
  separate additive estimator. Use the multiplicative factor extracted from
  log-soiling for correction, share model construction and solve logic, and
  document the model/API migration. Separate repairs of the old slope penalty
  and second-difference IRL1 loop are superseded.
- **Subsequent clarification, 2026-11-11:** LFA is the supported soiling
  interface. The thin-standalone-interface requirement in this entry is
  superseded by the retirement/migration proposal at SOIL-01 through SOIL-04.
- **Scope:** only this working plan was edited. No implementation, tests, or
  numerical experiments were performed.

### 2026-10-22 — Required quantile functionality moves into SDT

- **Owner decision:** move the required `spcqe` functionality into native CVXPY
  code in SDT, replacing the dependency. The proposed upstream-release path is
  superseded.
- **Disposition:** PKG-03 now includes shared periodic quantile fitting and
  transformation for PVQuantiles and anomaly detection, native Fourier helpers
  for capacity, and removal of the remaining dependency path to OSD. Preserve
  model and weight semantics except where changes are already agreed, and
  document the public-object and configuration migration.
- **Scope:** only this working plan was edited. Call sites were inspected
  read-only; no implementation, tests, or numerical experiments were performed.

### 2026-10-22 — Explicit unit handling adopted

- **Owner decision:** remove magnitude-based unit inference; honor supplied
  units and retain the existing W default.
- **Disposition:** DH-03 now specifies unit propagation to daily energy and
  loss accounting, and numerical conversion at requested display boundaries.
  No automatic-inference mode is retained.
- **Scope:** only this working plan was edited. No implementation, tests, or
  numerical experiments were performed.

### 2026-10-22 — Closest observation retained for timestamp collisions

- **Owner decision:** when observations collide during timestamp snapping,
  retain the observation closest to the grid timestamp and report the collision.
- **Disposition:** TIME-01 separates collision resolution from the requirement
  that each source observation populate at most one grid entry. Preserve empty
  bins as missing and verify the selected source row and observation mask.
- **Scope:** only this working plan was edited. No implementation, tests, or
  numerical experiments were performed.

### 2026-10-22 — Timezone warnings replace automatic offset correction

- **Owner direction:** provide a warning when significant production near
  midnight suggests the timezone is substantially wrong, replacing automatic
  timezone adjustment.
- **Disposition:** DH-04 now retires both inferred timezone-correction paths
  and specifies an advisory diagnostic. The proposed boundary-extension versus
  truncation choice is superseded. Preserve separately requested relative
  clock-shift correction and explicit timezone utilities; update migration and
  reporting guidance accordingly.
- **Scope:** only this working plan was edited. No implementation, tests, or
  numerical experiments were performed.

### 2026-10-22 — Bundt plotting requires explicit preprocessing

- **Owner decision:** require preprocessing before `plot_bundt`; do not launch
  it automatically from the plotting method.
- **Disposition:** DH-06 specifies explicit prerequisite checks and a clear
  error directing the caller to `run_pipeline`, including for incomplete
  analysis state.
- **Scope:** only this working plan was edited. No implementation, tests, or
  numerical experiments were performed.

### 2026-10-22 — Dataset quality uses the documented passing fraction

- **Owner decision:** the standalone `dataset_quality_score` reports the
  documented fraction of passing days. Continuous density ratios remain daily
  diagnostics.
- **Disposition:** QUALITY-01 now specifies Boolean aggregation and separates
  baseline-fitting thresholds from daily acceptance criteria. Preserve the
  handler's existing overall-quality definition based on `no_errors` flags.
- **Scope:** only this working plan was edited. No implementation, tests, or
  numerical experiments were performed.

### 2026-10-22 — Sensor identification retains usable subset rankings

- **Owner decision:** return rankings for usable subsets and flag incomplete
  coverage when another subset lacks sufficient validation data.
- **Disposition:** SENSOR-02 preserves available results without claiming
  agreement across untested conditions. Report observed disagreement and
  incomplete coverage distinctly.
- **Scope:** only this working plan was edited. No implementation, tests, or
  numerical experiments were performed.

### 2026-10-22 — Small degradation-sampling budgets remain available

- **Owner decision:** allow smaller diagnostic sampling runs with incomplete
  results, rather than reject budgets below twenty.
- **Disposition:** LFA-10 enforces the requested maximum, returns the collected
  diagnostic draws, and distinguishes insufficient draws from a normal summary
  and from stabilization. The minimum cannot cause the budget to be exceeded.
- **Scope:** only this working plan was edited. No implementation, tests, or
  numerical experiments were performed.

### 2026-10-22 — Failed sensitivity draws consume the attempt budget

- **Owner decision:** record numerical failures, count them toward `max_samples`,
  exclude them from statistics, and continue within the remaining budget.
- **Disposition:** LFA-10 distinguishes attempted, successful, and failed draws.
  Only successful draws enter the summary and stabilization checks; exhausted
  budgets return the collected results with the appropriate completion status.
- **Scope:** only this working plan was edited. No implementation, tests, or
  numerical experiments were performed.

### 2026-10-22 — Cumulative-minimum postprocessing adopted for nonlinear rates

- **Owner decision:** adopt cumulative-minimum postprocessing for nonlinear
  degradation-rate extraction, following the inspected RdTools convention.
- **Disposition:** LFA-08 applies it to both the overall chord and median annual
  change, in sensitivity draws and final reporting. Retain the raw fitted trend
  separately, keep the agreed reporting interval, and distinguish the reporting
  transformation from a change to the fitted decomposition or loss accounting.
- **Scope:** only this working plan was edited. No implementation, tests, or
  numerical experiments were performed.

### 2026-10-22 — Scalar quantile smoothing retains annual Fourier mode only

- **Owner decision:** retire the scalar quantile smoother's nonperiodic option
  and support annual Fourier mode only.
- **Disposition:** SD-01 now specifies one annual model across solvers and an
  explicit migration for the removed periodicity toggle. This does not retire
  the separate opt-in recurrence assumption for clock shifts.
- **Scope:** only this working plan was edited. No implementation, tests, or
  numerical experiments were performed.

### 2026-10-22 — Orientation estimation uses full-record clear-day eligibility

- **Owner decision:** use eligible clear days across the full record and retire
  the extra selectors in both orientation-estimation APIs.
- **Disposition:** PROF-03 replaces calendar-range and all/cloudy selector
  repairs with their explicit retirement. Preserve applicable observation
  eligibility checks, remove seasonal-range sweeps, and document the migration.
- **Scope:** only this working plan was edited. No implementation, tests, or
  numerical experiments were performed.

### 2026-10-22 — XGBoost remains optional and the ensemble remains the default

- **Owner decision:** provide XGBoost through an optional anomaly extra while
  retaining the ensemble default.
- **Disposition:** OUT-02 specifies the extra, conditional imports, and clear
  installation errors for classifiers that require it. Preserve classifier
  choices and ensemble membership, and verify the base and extra installations.
- **Scope:** only this working plan was edited. No implementation, tests, or
  numerical experiments were performed.

### 2026-10-22 — TiltAzimuthStudy requires explicit preprocessing

- **Owner decision:** require an already preprocessed `DataHandler` rather than
  automatically invoking `run_pipeline` during study construction.
- **Disposition:** PROF-02 specifies prerequisite checks, a clear error for
  unprocessed or incomplete state, and migration guidance. Retain the separate
  corrections to the decomposition call and use of its outputs.
- **Scope:** only this working plan was edited. No implementation, tests, or
  numerical experiments were performed.

### 2026-10-22 — Dask CSV ingestion retires Cassandra-specific parsing

- **Owner decision:** retire the Cassandra-specific path after confirming that
  S3-01 concerns input loading for the Dask runner.
- **Disposition:** S3-01 now specifies timestamp/power CSV input, removal of the
  Cassandra pivot and implicit UTC−8 shift, and migration of legacy exports.
  The proposed optional Cassandra mode is superseded. The related cleanup also
  covers the companion `LocalFiles` Cassandra option; ordinary local CSV loading
  remains supported.
- **Scope:** only this working plan was edited. No implementation, tests, or
  numerical experiments were performed.

### 2026-10-22 — Failed IRL1 refinements retain the last successful iterate

- **Owner decision:** for time-shift and capacity IRL1, retain the last successful
  iterate after a numerical refinement failure and report the failure. An initial
  fit failure leaves no usable candidate.
- **Disposition:** SD-04 specifies retention within the same candidate, continued
  selection/refitting, and separate failed-refinement termination. SD-06
  distinguishes this deliberate recovery from accidental reuse of stale solver
  values. Failed solver output must never supply the retained estimate.
- **Scope:** only this working plan was edited. No implementation, tests, or
  numerical experiments were performed.

### 2026-10-22 — Failed fixed-support refits return the selected IRL1 estimate

- **Owner decision:** return the selected IRL1 estimate if the final
  fixed-support refit fails, and flag that failure.
- **Disposition:** SD-03 preserves the penalized candidate and distinguishes it
  from a successful fixed-support refit. Labels, corrections, and plots use the
  returned components consistently; selected support and IRL1 termination remain
  separate diagnostics. CAP-02 records the corresponding label fallback.
- **Scope:** only this working plan was edited. No implementation, tests, or
  numerical experiments were performed.

### 2026-10-22 — Inaccurate optimal solves receive qualified acceptance

- **Owner decision:** accept `optimal_inaccurate` with numerical checks and an
  explicit warning.
- **Disposition:** SD-06 requires finite values and passing numerical feasibility
  checks, retains the solver status, and propagates the qualification with
  accepted estimates. Unusable results follow the relevant stage's failure
  policy; feasibility alone is not an optimality certificate.
- **Scope:** only this working plan was edited. No implementation, tests, or
  numerical experiments were performed.

### 2026-10-22 — Dask cleanup remains with the caller

- **Owner decision:** the caller manages cleanup; `Runner` never shuts down the
  supplied cluster.
- **Disposition:** DASK-02 preserves the supplied client and cluster on success
  and failure, removes the proposed runner shutdown option, and assigns explicit
  cleanup to caller code in examples and migration guidance.
- **Scope:** only this working plan was edited. No implementation, tests, or
  numerical experiments were performed.

### 2026-10-22 — Native decomposition functions return one result object

- **Owner decision:** return one lightweight object with named components and
  diagnostics from each low-level native decomposition function.
- **Disposition:** SD-01 replaces positional tuples and `return_all` variants
  with consistent role-based access and explicit solve/refinement outcomes.
  SD-06 separates returned result values from the mutable, reusable solver
  workspace. Exact names and migration examples remain implementation
  specifications to settle before source changes.
- **Scope:** only this working plan was edited. No implementation, tests, or
  numerical experiments were performed.

### 2026-10-22 — Weight searches continue over usable candidates

- **Owner decision:** continue time-shift and capacity selection over usable
  candidates and report failed weights.
- **Disposition:** SD-04 excludes numerically failed initial fits from screening
  and records incomplete paths. CAP-02 and SHIFT-01 require usable references
  and distinguish numerical failure from the existing no-qualifying-candidate
  fallback. No screening thresholds or ranking rules are replaced.
- **Scope:** only this working plan was edited. No implementation, tests, or
  numerical experiments were performed.

### 2026-10-22 — The refactored package requires NumPy 2.x

- **Owner decision:** require NumPy 2.x and retire NumPy 1.x support.
- **Disposition:** PKG-01 retains the removed-API corrections and withdraws
  `np.concat` as a compatibility defect under the chosen target. Validate exact
  bounds and align the base package, Dask extra, Docker requirements, CI, and
  migration documentation with supported Python/dependency combinations.
- **Scope:** only this working plan was edited. No implementation, tests, or
  numerical experiments were performed.

### 2026-11-11 — Provisional IRL1 numerical defaults specified

- **Owner decision:** use one ordinary-l1 initialization, at most twenty
  reweighted solves, and a relative fixed-merit decrease tolerance of $10^{-4}$
  as configurable starting values, with validation before adoption. Exactly
  zero merit stops immediately; increases are not convergence.
- **Disposition:** SD-04 records the stopping inequality, solve-count convention,
  and numerical/runtime comparisons needed to evaluate these defaults.
  Treatment of numerical merit increases remains to be specified separately.
- **Scope:** only this working plan was edited. No implementation, tests, or
  numerical experiments were performed.

### 2026-11-11 — IRL1 rejects excessive merit increases

- **Owner decision:** reject a step whose fixed log-sum merit increase exceeds
  numerical tolerance, stop reweighting, retain the preceding iterate, and
  report the increase.
- **Disposition:** SD-04 defines a separate merit-increase termination outcome
  while retaining the candidate for selection/refitting. The rejected solve's
  status and merit values remain diagnostic information. The numerical allowance
  for merit increases remains to be specified and validated.
- **Scope:** only this working plan was edited. No implementation, tests, or
  numerical experiments were performed.

### 2026-11-11 — Provisional IRL1 merit-increase allowance specified

- **Owner decision:** use a configurable relative allowance of $10^{-6}$ times
  the preceding merit, with no absolute term, subject to validation.
- **Disposition:** SD-04 accepts and records smaller positive increases without
  declaring convergence, rejects larger increases under the preceding decision,
  and evaluates the allowance alongside the $10^{-4}$ stopping tolerance and
  twenty-reweighting cap before adoption.
- **Scope:** only this working plan was edited. No implementation, tests, or
  numerical experiments were performed.

### 2026-11-11 — Full-grid component construction clarified

- **Owner clarification:** every component is defined over the entire time
  index. For the predefined residual costs, the residual is zero outside the
  known-set support; this is an aspect of the method's construction.
- **Disposition:** the common mathematical convention, SD-01 result contract,
  and SD-06 masking/parameterization guidance now state this invariant. Withdraw
  the proposed NaN residual output. Preserve the input mask and raw missingness
  separately from the full-grid component estimates and imputed signal.
- **Source check:** the skill's `src/signaldecomp/decompose.py` applies the
  residual cost to the full-length residual and masks the consistency equality;
  its predefined costs in `data_fidelity.py` are separable and minimized at zero.
  This confirms the construction without introducing a runtime `signaldecomp`
  dependency or changing SDT's agreed loss normalization.
- **Scope:** only this working plan was edited. No implementation, tests, or
  numerical experiments were performed.

### 2026-11-11 — Timestamp snapping defaults to half the sampling interval

- **Owner decision:** use a default tolerance of $\Delta/2$ and allow a caller
  override.
- **Disposition:** TIME-01 records the physical-time tolerance and retains the
  agreed one-use and closest-collision rules. Tie handling remains a documented
  implementation proposal; jitter, tolerance boundaries, and cadence cases are
  included in the verification scope.
- **Scope:** only this working plan was edited. No implementation, tests, or
  numerical experiments were performed.

### 2026-11-11 — Numerical failures support pipeline continuation

- **Owner direction:** favor documented failure and continuation over requiring
  downstream pipelines to catch a special numerical exception. Preserve failure
  information in reports during continuation.
- **Disposition:** SD-01/SD-06 use a failure result when no accepted fit exists;
  the exception-first proposal is withdrawn. Usable retained-fit fallbacks remain
  unchanged. Added DH-07 for the source-backed loss of failure information in
  handler/Dask reports and the proposed per-stage continuation contract.
- **Scope:** only this working plan was edited. No implementation, tests, or
  numerical experiments were performed.

### 2026-11-11 — Orientation decomposition uses constrained Fourier terms

- **Owner decision:** use Fourier representations with physical constraints
  for both annual terms in `find_fit_costheta`, and validate the changes.
- **Disposition:** PROF-01 adopts the common 365.2425-day period while retaining
  solstice symmetry for the incidence term and full-grid nonpositivity for both
  annual terms. Retire the earlier singleton/annual-lag equality repairs with
  the vector constructions they would have patched. Harmonic counts, phase,
  and regularization remain specifications to validate;
  verification includes downstream tilt/azimuth recovery.
- **Scope:** only this working plan was edited. No implementation, tests, or
  numerical experiments were performed.

### 2026-11-11 — Shading reconstruction uses a shared conservative map

- **Owner decision:** adopt a shared linear, energy-conserving resampling map
  for clear sky, shade, and residual, with validation.
- **Disposition:** SHADE-01 specifies common component mapping, additive
  reconstruction, and conservation under explicit integration weights. Retain
  the existing fitted model initially and verify daylight-grid/corpus
  compatibility. Dimensionless shading fractions remain a separate
  specification; the decision does not authorize a model or corpus redesign.
- **Scope:** only this working plan was edited. No implementation, tests, or
  numerical experiments were performed.

### 2026-11-11 — Shading fractions use resampled components

- **Owner decision:** calculate original-grid shading fractions as resampled
  shade divided by resampled clear sky.
- **Disposition:** SHADE-01 uses that ratio for heatmap and polar annotations,
  restricted to adequate clear-sky levels. The proposal to interpolate the
  normalized-grid fraction is superseded. Verification includes proportional
  components and varying profiles that distinguish the two procedures.
- **Scope:** only this working plan was edited. No implementation, tests, or
  numerical experiments were performed.

### 2026-11-11 — Automatic weight searches verify their constant reference

- **Owner decision:** verify the upper endpoint and extend the weight range
  only if needed, with a bounded extension, for capacity and time-shift searches.
- **Disposition:** CAP-02/SHIFT-01 retain sufficient existing grids and specify
  reporting of added weights and the accepted constant-component reference.
  Use that reference's residual error for the 2% screen; distinguish numerical
  constancy from meaningful-jump thresholds. Preserve the existing component
  criteria and report unsuccessful selection if the bounded procedure cannot
  establish a usable reference. No failure of the existing capacity endpoint
  has been demonstrated.
- **Scope:** only this working plan was edited. No implementation, tests, or
  numerical experiments were performed.

### 2026-11-11 — CLARABEL defaults and CVXPY pass-through agreed

- **Owner decision:** use CLARABEL with its default settings as the default;
  allow pass-through to CVXPY for other solvers and solver-specific settings.
- **Disposition:** SD-01/SD-06 specify the default, option forwarding through
  all applicable solve stages, and retirement of silent solver substitution.
  Keep solve configuration separate from model selection, numerical acceptance,
  and IRL1 termination. Include migration examples and forwarding verification.
- **Scope:** only this working plan was edited. No implementation, tests, or
  numerical experiments were performed.

### 2026-11-11 — Final linear loss fitting uses the sampled median rate

- **Owner decision:** after a completed sensitivity run and absent an overriding
  caller-supplied fixed rate, fix the final full-data linear degradation fit to
  the median of the sampled endpoint-chord rates.
- **Disposition:** LFA-04 makes this an explicit final-phase constraint using
  LFA-01's exact annual-rate conversion. Keep draw rates freely estimated,
  restore the base mask and configured quantile for final fitting, and retain
  the sampled summary separately. LFA-07 follows the same phase distinction;
  nonlinear final fits remain unconstrained by the sampled summary.
- **Scope:** only this working plan was edited. No implementation, tests, or
  numerical experiments were performed.

### 2026-11-11 — Incomplete sensitivity permits a free-rate final fit

- **Owner decision:** with fewer than twenty successful sensitivity draws and
  no supplied fixed rate, attempt a full-data fit with a freely estimated
  linear rate and flag sensitivity as incomplete.
- **Disposition:** LFA-04/LFA-10 separate the final-fit outcome from sensitivity
  completeness. A successful fit can provide loss attribution and a labeled
  freely estimated rate without creating a completed sensitivity summary.
  Preserve samples and failure diagnostics, restore the base model settings,
  and keep the final fit outside the sensitivity-draw budget and counts.
- **Scope:** only this working plan was edited. No implementation, tests, or
  numerical experiments were performed.

### 2026-11-11 — Unstabilized sampling retains its median after twenty successes

- **Owner decision:** when the sampling budget is exhausted with at least
  twenty successful draws, use the sampled median for the final linear fit
  and flag lack of stabilization.
- **Disposition:** LFA-04/LFA-10 retain the sensitivity percentiles and explicit
  budget-exhausted/not-stabilized warning. LFA-07 preserves that qualification
  during final fitting. The free-rate fallback remains specific to fewer than
  twenty successful draws; a consolidated outcome table distinguishes the
  three cases without changing the sampling budget or nonlinear model.
- **Scope:** only this working plan was edited. No implementation, tests, or
  numerical experiments were performed.

### 2026-11-11 — Supported soiling analysis is the main LFA

- **Owner clarification:** the supported soiling analysis is the main
  loss-factor analysis; a standalone soiling analysis may not survive.
- **Disposition:** remove the requirement for a thin standalone wrapper from
  LFA-06 and SOIL-01 through SOIL-04. Propose retirement of the legacy standalone
  entry points and their exports, with migration to LFA. Retain the defect
  evidence and applicable mathematical verification without requiring a second
  public estimator. Withdraw the standalone quantile-default question; LFA's
  existing configurable quantile loss and default remain in effect.
- **Scope:** only this working plan was edited. No implementation, tests, or
  numerical experiments were performed; public API removal remains proposed.

### 2026-11-11 — Standalone soiling retirement confirmed

- **Owner decision:** retire the existing standalone soiling algorithm and
  direct users to LFA.
- **Disposition:** LFA-06 and SOIL-01 through SOIL-04 now record retirement as
  the agreed direction, superseding the earlier tentative retirement proposal
  and thin-wrapper requirement. Migrate the standalone public entry points,
  exports, examples, and documentation to LFA, with explicit changed-model and
  component-extraction guidance. Preserve the legacy findings as evidence;
  implementing their separate repairs is no longer the planned work.
- **Scope:** only this working plan was edited. No implementation, tests, or
  numerical experiments were performed. Source removal requires issue-specific
  implementation direction.

### 2026-11-11 — The final-fit rate is the reporting headline

- **Owner decision:** use the final-fit rate as the headline degradation rate.
- **Disposition:** LFA-04/LFA-08 bind `degradation_rate` to the accepted final
  fit's overall endpoint chord under the agreed reporting transform. Retain
  the sensitivity median and percentiles separately from the headline and from
  the within-fit median annual-change comparison. LFA-02 clarifies the
  sensitivity labels and legacy bound-attribute migration. An unavailable final
  fit leaves the headline unavailable without discarding a valid sensitivity
  summary.
- **Scope:** only this working plan was edited. No implementation, tests, or
  numerical experiments were performed.

### 2026-11-11 — Short-record LFA warns and attempts the configured model

- **Owner decision:** warn and attempt the configured LFA model on records
  shorter than one year.
- **Disposition:** LFA-06/SOIL-03 retain warning-and-continuation with a recorded
  applicability warning, without adopting the retiring standalone algorithm's
  automatic removal of seasonality. Validate component recovery and usable
  record lengths; retain ordinary configuration and numerical acceptance checks.
- **Decision-pass assessment:** the major owner-level choices identified in
  this review are now recorded. The remaining specification and evidence work
  is consolidated near the beginning of this document by code location. This
  assessment does not mark the issues implemented, validate the algorithms, or
  authorize source changes or experiments.
- **Scope:** only this working plan was edited. No implementation, tests, or
  numerical experiments were performed.

### 2026-11-11 — Follow-up assessment exposes interactions and public callers

- **Review disposition:** incorporate the recurrence/pooling and clear-sky
  counterexamples, clock-coordinate mismatch in dilation, Dask duration gate,
  LFA provenance and numerical-search gaps, and quantile evaluation/artifact
  contracts into their existing issue entries. Add the related cadence,
  eligibility, dependency-gate, plotting, and XGBoost fitting findings at their
  owning locations. Retain all 62 issue IDs and settled model policies.
- **Evidence:** current local source, local upstream dependency source, and
  algebraic reasoning support these additions. Runtime, component-recovery,
  performance, and installed-wheel verification remain proposed.
- **Planning correction:** the earlier question-pass closure was premature.
  The remaining-specifications table now names the public verification paths
  and distinguishes remaining choices from already agreed policies.
- **Owner decisions in this follow-up:** optimize clear-sky paths independently
  within contiguous daylight segments; account for observed production with
  missing sampling intervals contributing zero; retain filled daily energy
  for fitting on days admitted by the existing quality/outage flags. Keep the
  energy vectors, coverage, and accounting assumptions explicit.
- **Further owner decision:** for an admitted partial day use
  $o_t=E_t^{\mathrm{obs}}/E_t^{\mathrm{fill}}$ and
  $w_t=E_t^{\mathrm{fill}}/q_t$, with positive denominators and filling that
  preserves nonnegative observed samples. Keep the previously agreed flagged-
  and fully missing-day rules.
- **Measured-zero decision:** exclude genuinely measured-zero days from the
  log fit and use $o_t=0$, $w_t=1$, while retaining their distinct input status.
- **Admission decision:** remove the unit-dependent `0.1` near-zero cutoff;
  use finite, strictly positive fitting energy and the existing quality/outage
  flags. Keep positive reported production in accounting independently of
  fit admission; no additional low-energy threshold is adopted.
- **Saved-model migration decision:** require retraining for the native SDT
  version and retain old environments for existing saved outage models.
  Document the incompatibility and artifact provenance; no converter or
  automatic cross-version loading is included in the agreed port.
- **Incomplete-day decision:** `make_2d` pads partial calendar days with NaN
  by default, preserving time-of-day positions and one date per output column.
  Retain explicit trimming and the supported cadence/grid-alignment checks.
- **XGBoost fitting decision:** train the explicit XGBoost classifier for its
  configured boosting-round count without early stopping. Retain the classifier
  validation split for scoring; no separate stopping split is introduced.
- **Scope:** only this working plan was edited. No implementation, tests,
  numerical solves, installs, builds, or experiments were performed. New
  proposals and validation plans still require issue-specific direction.

### 2026-11-11 — DH-01 completed

- **Owner disposition:** accepted and committed on `cleanup` in
  `dacc3037ea6068253ff7e5d769f472a28b84e774`.
- **Agreed contract:** plotting leaves the stored matrix intact. The existing
  W-to-kW scaling conditions and explicit `units` override apply.
- **Authorized work and changes:** use out-of-place division in
  `DataHandler.plot_heatmap` and add one regression for repeated extra-matrix
  plots. Both plots show the expected kW values and label, and the stored watt
  values and missing entry are preserved.
- **Verification:** the initial regression reproduced double scaling on the
  second plot before the fix. The final run of
  `pytest -v tests/solardatatools/test_data_handler.py` passed both tests with
  Python 3.13.11 in `sdt-dev` and `MPLBACKEND=Agg`. `git diff --check` passed.
- **Review and remaining scope:** independent public-tool review inspected the
  diff and plotting call path and reported no findings. The reviewer did not
  rerun the tests. The owner accepted the result. DIL-01 remains open.
