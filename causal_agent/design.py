"""
Experiment design over the 10 available deployments.

The binding constraint is arithmetic, not statistical sophistication: 200 drones
across at most 10 design points. A pairwise contrast on a subtle mechanism
(delta ~ 0.14 around p ~ 0.78) needs ~70 drones per arm for 2 sigma, so
one-factor-at-a-time burns the entire budget on a single question. A factorial
layout with pooled estimation spends every drone on every coefficient instead.
"""

import itertools
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np


# ----------------------------------------------------------------------
# Factors
# ----------------------------------------------------------------------

@dataclass(frozen=True)
class Factor:
    """
    A two-level (or k-level) experimental factor.

    Each level carries the design/equipment fragment it applies, so a factor may
    move several DEF keys at once (e.g. an overall "critical armour" level).
    """

    name: str
    levels: Tuple[Any, ...]
    settings: Tuple[Dict[str, Any], ...]
    target: str = "design"  # "design" or "equipment"

    def __post_init__(self):
        if len(self.levels) != len(self.settings):
            raise ValueError(f"factor {self.name}: levels/settings length mismatch")
        if self.target not in ("design", "equipment"):
            raise ValueError(f"factor {self.name}: bad target {self.target!r}")


# ----------------------------------------------------------------------
# Design space
# ----------------------------------------------------------------------

class DesignSpace:
    """
    Action space read from the API, with client-side constraint checking.

    Args:
        action_space: payload from client.get_action_space().
    """

    def __init__(self, action_space: Dict[str, Any]):
        # /api/v2/action_space wraps the payload; accept either shape.
        if "action_space" in action_space and "numerical" not in action_space:
            self.envelope = action_space
            action_space = action_space["action_space"] or {}
        else:
            self.envelope = {}
        self.raw = action_space
        self.numerical: Dict[str, Dict[str, Any]] = action_space.get("numerical", {}) or {}
        self.discrete: Dict[str, Dict[str, Any]] = action_space.get("discrete", {}) or {}
        self.boolean: Dict[str, Dict[str, Any]] = action_space.get("boolean", {}) or {}
        constraints = action_space.get("constraints", {}) or {}
        self.total_def_budget: Optional[float] = constraints.get("total_def_budget")

    @property
    def def_keys(self) -> List[str]:
        return list(self.numerical.keys())

    def default_design(self) -> Dict[str, int]:
        return {k: int(v.get("default", 0)) for k, v in self.numerical.items()}

    def default_equipment(self) -> Dict[str, Any]:
        eq: Dict[str, Any] = {}
        for k, v in self.discrete.items():
            if "default" in v:
                eq[k] = v["default"]
        for k, v in self.boolean.items():
            if "default" in v:
                eq[k] = v["default"]
        return eq

    def clip(self, design: Dict[str, int]) -> Dict[str, int]:
        out = {}
        for k, v in design.items():
            spec = self.numerical.get(k, {})
            lo = spec.get("min", 0)
            hi = spec.get("max", 50)
            out[k] = int(min(max(v, lo), hi))
        return out

    def total_def(self, design: Dict[str, int]) -> int:
        return sum(int(v) for k, v in design.items() if k in self.numerical)

    def violations(self, design: Dict[str, int]) -> List[str]:
        """Constraint failures, empty if the design is submittable."""
        problems = []
        for k, v in design.items():
            spec = self.numerical.get(k)
            if spec is None:
                problems.append(f"unknown design key {k!r}")
                continue
            if not (spec.get("min", 0) <= v <= spec.get("max", 50)):
                problems.append(f"{k}={v} outside [{spec.get('min')}, {spec.get('max')}]")
        if self.total_def_budget is not None:
            total = self.total_def(design)
            if total > self.total_def_budget:
                problems.append(f"total_def {total} exceeds budget {self.total_def_budget}")
        return problems

    def materialize(
        self,
        factors: Sequence[Factor],
        level_idx: Sequence[int],
        base_design: Optional[Dict[str, int]] = None,
        base_equipment: Optional[Dict[str, Any]] = None,
    ) -> Tuple[Dict[str, int], Dict[str, Any]]:
        """Apply one row of a factorial layout to the base design."""
        design = dict(base_design if base_design is not None else self.default_design())
        equipment = dict(base_equipment if base_equipment is not None else self.default_equipment())
        for factor, idx in zip(factors, level_idx):
            fragment = factor.settings[idx]
            if factor.target == "design":
                design.update(fragment)
            else:
                equipment.update(fragment)
        return self.clip(design), equipment


# ----------------------------------------------------------------------
# Factorial layouts
# ----------------------------------------------------------------------

def full_factorial(k: int) -> np.ndarray:
    """All 2^k level-index rows."""
    return np.array(list(itertools.product([0, 1], repeat=k)), dtype=int)


def fractional_factorial(k: int, runs: Optional[int] = None) -> np.ndarray:
    """
    Two-level layout in `runs` rows.

    Supports the full 2^k design and the half fraction 2^(k-1) generated by
    setting the last factor to the product of the others. For k=4 in 8 runs that
    is the standard resolution-IV design: main effects are clear of two-factor
    interactions, which is what we need since only main effects are affordable.

    Returns:
        (runs, k) array of level indices in {0, 1}.
    """
    if runs is None or runs >= 2 ** k:
        return full_factorial(k)
    if runs != 2 ** (k - 1):
        raise ValueError(
            f"only full (2^{k}) and half ({2 ** (k - 1)}) fractions supported, got {runs}"
        )
    base = full_factorial(k - 1)
    coded = np.where(base == 0, -1, 1)
    generated = np.prod(coded, axis=1)
    last = np.where(generated < 0, 0, 1).reshape(-1, 1)
    return np.hstack([base, last])


def allocate(total_drones: int, n_arms: int, weights: Optional[Sequence[float]] = None) -> List[int]:
    """
    Split the drone budget across arms, largest-remainder rounding.

    Args:
        weights: relative effort per arm; defaults to equal. Give confirmatory
            arms more weight when the contrast they resolve is small.
    """
    if n_arms <= 0:
        return []
    w = np.ones(n_arms) if weights is None else np.asarray(weights, dtype=float)
    if w.sum() <= 0:
        raise ValueError("weights must sum to a positive number")
    w = w / w.sum()
    exact = w * total_drones
    counts = np.floor(exact).astype(int)
    counts = np.maximum(counts, 1)
    # Give back or claw back the rounding drift.
    drift = total_drones - int(counts.sum())
    order = np.argsort(-(exact - np.floor(exact)))
    i = 0
    while drift != 0:
        j = order[i % n_arms]
        if drift > 0:
            counts[j] += 1
            drift -= 1
        elif counts[j] > 1:
            counts[j] -= 1
            drift += 1
        i += 1
    return [int(c) for c in counts]


# ----------------------------------------------------------------------
# Feature maps
# ----------------------------------------------------------------------

class FeatureMap:
    """
    Turns (design, equipment) into the regressor row used by the outcome model.

    The choice of basis IS the causal hypothesis. `linear()` is the
    assumption-light default; threshold bases (antenna alive vs destroyed,
    camera above/below its floor) encode a specific mechanism and must be
    *derived from data* by the causal core, never assumed up front - assuming
    them is how you accidentally hard-code the answer key.
    """

    def __init__(self, terms: Dict[str, Callable[[Dict[str, Any], Dict[str, Any]], float]]):
        self.terms = dict(terms)

    @property
    def names(self) -> List[str]:
        return ["intercept"] + list(self.terms.keys())

    def row(self, design: Dict[str, Any], equipment: Dict[str, Any]) -> np.ndarray:
        return np.array(
            [1.0] + [f(design, equipment) for f in self.terms.values()], dtype=float
        )

    def matrix(self, arms: Sequence[Any]) -> np.ndarray:
        # Environment variables ride along under an env_ prefix so terms that
        # condition on per-deployment context can see them; terms that don't
        # reference them are unaffected.
        rows = []
        for a in arms:
            env = getattr(a, "environment", None) or {}
            merged = {**a.equipment, **{f"env_{k}": v for k, v in env.items()}}
            rows.append(self.row(a.design, merged))
        return np.vstack(rows)

    # -- constructors ---------------------------------------------------

    @classmethod
    def linear(cls, def_keys: Sequence[str], scale: float = 10.0) -> "FeatureMap":
        """One coefficient per DEF key, plus a quadratic in total DEF."""
        terms: Dict[str, Callable] = {}
        for key in def_keys:
            terms[key] = (lambda k: (lambda d, e: d.get(k, 0) / scale))(key)
        keys = list(def_keys)
        terms["total_def"] = lambda d, e: sum(d.get(k, 0) for k in keys) / scale
        terms["total_def_sq"] = lambda d, e: (sum(d.get(k, 0) for k in keys) / scale) ** 2
        return cls(terms)

    @classmethod
    def from_terms(cls, terms: Dict[str, Callable]) -> "FeatureMap":
        """Hypothesis-driven basis, e.g. proposed by the causal core."""
        return cls(terms)


# ----------------------------------------------------------------------
# Information-driven arm selection
# ----------------------------------------------------------------------

def d_optimal_subset(
    candidates: np.ndarray,
    k: int,
    seed_rows: Optional[np.ndarray] = None,
    ridge: float = 1e-6,
) -> List[int]:
    """
    Greedily pick `k` candidate rows maximising log det(X'X + ridge*I).

    Args:
        candidates: (n, p) feature rows to choose from.
        k: how many to pick.
        seed_rows: rows already committed (previous deployments).

    Returns:
        Indices into `candidates`, in selection order.
    """
    candidates = np.atleast_2d(candidates)
    p = candidates.shape[1]
    info = ridge * np.eye(p)
    if seed_rows is not None and len(seed_rows):
        seed = np.atleast_2d(seed_rows)
        info = info + seed.T @ seed

    chosen: List[int] = []
    for _ in range(min(k, len(candidates))):
        best_idx, best_score = None, -np.inf
        for i, row in enumerate(candidates):
            if i in chosen:
                continue
            sign, logdet = np.linalg.slogdet(info + np.outer(row, row))
            score = logdet if sign > 0 else -np.inf
            if score > best_score:
                best_idx, best_score = i, score
        if best_idx is None:
            break
        chosen.append(best_idx)
        info = info + np.outer(candidates[best_idx], candidates[best_idx])
    return chosen
