"""
Count-based session wrapper.

When `hide_failed_drones` is enabled the backend filters DESTROYED/LOST drones
out of both `get_history()` and the per-drone `results` list, but still reports
`deployed` / `survived` / `destroyed` counts (api/modules/agent/endpoints.py:411).

That asymmetry is the whole game:

  * counts            -> unbiased survival rate for the deployed design
  * per-drone records -> a sample conditioned on survival, i.e. on a collider

Regressing survival on features drawn from the per-drone table inverts the sign
of the DEF effect, because among survivors heavy armour looks protective while
the true mechanism is total_def -> agility down -> hit probability up. This
wrapper therefore makes counts the primary channel and quarantines the censored
records behind an explicit opt-in.
"""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


class CensoredDataError(RuntimeError):
    """Raised when survivor-only records are used as if they were a sample."""


# DeployRequest caps count at 50 (api/modules/agent/endpoints.py:45-50), so a
# single design point can never carry more than 50 drones however the budget is
# allocated.
MAX_DRONES_PER_CALL = 50


def _coerce_equipment(equipment: Optional[Dict[str, Any]]) -> Dict[str, str]:
    """
    The deploy endpoint types equipment as Dict[str, str]; booleans 422.
    """
    out: Dict[str, str] = {}
    for key, value in (equipment or {}).items():
        if isinstance(value, bool):
            out[key] = "true" if value else "false"
        else:
            out[key] = str(value)
    return out


@dataclass(frozen=True)
class Arm:
    """One deployment: a design held fixed over `deployed` independent drones."""

    label: str
    design: Dict[str, int]
    equipment: Dict[str, Any]
    deployed: int
    survived: int
    # Per-deployment environment as reported by the backend (e.g. weather).
    # Pre-treatment context: legitimate to condition on, and on scenarios with
    # strong environment noise the arms are overdispersed without it.
    environment: Dict[str, Any] = field(default_factory=dict)

    @property
    def destroyed(self) -> int:
        return self.deployed - self.survived

    @property
    def rate(self) -> float:
        """Unbiased survival rate. Valid because the denominator is known."""
        return self.survived / self.deployed if self.deployed else 0.0

    def __repr__(self) -> str:
        return (
            f"Arm({self.label!r}, {self.survived}/{self.deployed} = "
            f"{self.rate:.0%})"
        )


class CausalSession:
    """
    Wraps CanyonClient so that every observation carries its denominator.

    Args:
        client: a live CanyonClient (already registered).
        strict: if True (default), accessing per-drone records raises unless
            the caller passes acknowledge_censoring=True.
    """

    def __init__(self, client, strict: bool = True):
        self.client = client
        self.strict = strict
        self.arms: List[Arm] = []
        # per-arm survivor records, parallel to self.arms (censored - same
        # caveats as censored_records; consumed by the C3 mediator machinery)
        self.arm_records: List[List[Dict[str, Any]]] = []
        self._censored: List[Dict[str, Any]] = []
        self._status: Dict[str, Any] = {}
        self.refresh_status()

    # ------------------------------------------------------------------
    # Budget
    # ------------------------------------------------------------------

    def refresh_status(self) -> Dict[str, Any]:
        self._status = self.client.get_status()
        return self._status

    def _stat(self, *names: str, default: Any = None) -> Any:
        """Read the first present key; the status schema varies by version."""
        for name in names:
            if name in self._status:
                return self._status[name]
        return default

    @property
    def drones_remaining(self) -> int:
        return int(self._stat("drones_remaining", default=0))

    @property
    def deployments_remaining(self) -> int:
        return int(self._stat("deployments_remaining", default=0))

    @property
    def victory_threshold(self) -> float:
        raw = self._stat("victory_threshold", default=0.75)
        # Config stores 0.75; some endpoints report 75.
        return float(raw) / 100.0 if float(raw) > 1.0 else float(raw)

    @property
    def drones_used(self) -> int:
        return sum(a.deployed for a in self.arms)

    # ------------------------------------------------------------------
    # Data collection
    # ------------------------------------------------------------------

    def deploy(
        self,
        design: Dict[str, int],
        count: int,
        equipment: Optional[Dict[str, Any]] = None,
        label: str = "",
    ) -> Arm:
        """
        Deploy `count` drones of one design and record the counts.

        Returns:
            Arm with a known denominator, safe for pooled estimation.
        """
        if count <= 0:
            raise ValueError("count must be positive")
        if count > MAX_DRONES_PER_CALL:
            raise ValueError(
                f"count={count} exceeds the API cap of {MAX_DRONES_PER_CALL} per "
                "deployment; split the allocation across more design points"
            )

        payload_equipment = _coerce_equipment(equipment)
        response = self.client.deploy_drone(
            design, count=count, equipment=payload_equipment or None
        )

        deployed = int(response.get("deployed", count))
        if "survived" in response:
            survived = int(response["survived"])
        else:
            # Fallback: no_selection_bias variants return every record.
            results = response.get("results", []) or []
            survived = sum(1 for r in results if r.get("status") == "RETURNED")

        arm = Arm(
            label=label or f"arm{len(self.arms):02d}",
            design=dict(design),
            equipment=dict(equipment or {}),
            deployed=deployed,
            survived=survived,
            environment=dict(response.get("environment") or {}),
        )
        self.arms.append(arm)
        self.arm_records.append(list(response.get("results", []) or []))
        self._censored.extend(response.get("results", []) or [])
        self.refresh_status()
        return arm

    def submit(
        self,
        design: Dict[str, int],
        equipment: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Irreversible stage-2 submission over the 1000-drone fleet."""
        payload_equipment = _coerce_equipment(equipment)
        return self.client.submit_final_design(
            design, equipment=payload_equipment or None
        )

    # ------------------------------------------------------------------
    # Quarantine
    # ------------------------------------------------------------------

    def censored_records(self, acknowledge_censoring: bool = False) -> List[Dict[str, Any]]:
        """
        Per-drone records for surviving drones only.

        These are legitimate for *mechanism* questions asked within the survivor
        stratum (e.g. does antenna_status track is_detected), but they are not a
        sample from the deployed population. Never regress survival on them.
        """
        if self.strict and not acknowledge_censoring:
            raise CensoredDataError(
                "Per-drone records exclude destroyed drones. Using them as a "
                "sample conditions on survival (a collider) and inverts the DEF "
                "effect. Use session.arms for outcome modelling, or pass "
                "acknowledge_censoring=True for within-survivor mechanism work."
            )
        return list(self._censored)

    def observational_prior(self, acknowledge_censoring: bool = False) -> List[Dict[str, Any]]:
        """
        The pre-seeded flight history (`initial_observations` in game.json).

        Denominator unknown: the config seeds N observations but the censored
        history shows only those that returned, so a survival rate cannot be
        recovered from it at all. Useful only for discovering which environment
        variables exist and how mediators relate to each other.
        """
        if self.strict and not acknowledge_censoring:
            raise CensoredDataError(
                "Seeded history has no known denominator - no survival rate is "
                "identifiable from it. Pass acknowledge_censoring=True to use it "
                "for variable discovery only."
            )
        return self.client.get_history()

    # ------------------------------------------------------------------
    # Export
    # ------------------------------------------------------------------

    def counts(self) -> List[Dict[str, Any]]:
        """Arms as plain dicts, for the estimator."""
        return [
            {
                "label": a.label,
                "deployed": a.deployed,
                "survived": a.survived,
                "rate": a.rate,
                **a.design,
                **{f"eq_{k}": v for k, v in a.equipment.items()},
                **{f"env_{k}": v for k, v in a.environment.items()},
            }
            for a in self.arms
        ]
