"""Result types shared by native signal decomposition models."""

from dataclasses import dataclass, field

import numpy as np


@dataclass(frozen=True)
class SolveDiagnostics:
    """Solver settings, backend status, and numerical checks for one attempt.

    Constraint errors are keyed by the model's constraint names. An empty
    mapping indicates that the errors could not be evaluated.
    """

    solver: str
    solver_options: dict
    solver_status: str | None = None
    objective_value: float | None = None
    constraint_errors: dict[str, float] = field(default_factory=dict)
    warnings: tuple[str, ...] = ()
    failure_reason: str | None = None


@dataclass(frozen=True)
class DecompositionResult:
    """Snapshot of the components by role and their admitted observations."""

    observation_mask: np.ndarray
    components: dict[str, np.ndarray] | None
    diagnostics: SolveDiagnostics

    @property
    def has_fit(self):
        return self.components is not None
