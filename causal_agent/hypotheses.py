"""
C1 + C2 of the causal core: mechanism hypotheses and evidence weighting.

A *hypothesis* is a small set of candidate mechanisms, each a declarative Term
(linear / threshold / saturating / interaction / equipment indicator) with an
optional sign prior. Hypotheses compile to design.FeatureMap, so the existing
estimator and decision rule work unchanged.

The declarative form is deliberate: it is the interface at which an LLM (or a
heuristic library) proposes candidate causal features as data, never as code.
Whether a proposed mechanism is *real* is decided here, by Laplace log-evidence
over the deployment counts - the LLM names features, the evidence proves them.

Scoring is model *averaging*, not selection: at 8-12 arms no single hypothesis
is identifiable, and argmax over hypotheses would be the same winner's-curse
mistake as argmax over noisy arms (design invariant 4).
"""

import json
import math
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np

from .design import DesignSpace, FeatureMap
from .estimator import BinomialLogit, sigmoid

# All features are normalised to [0, 1] using the action space (per-key max for
# linear/interaction terms, the total budget for total terms). This is what
# keeps a weakly-informative coefficient prior meaningful *everywhere in the
# design space*: with unbounded features a fitted logit extrapolates to p ~ 1
# in unexplored corners, the decision rule chases the mirage, and the
# information gain of every real experiment rounds to zero.

TERM_KINDS = (
    "linear", "threshold", "saturating", "interaction",
    "total", "total_sq", "equipment", "group_total", "min",
    "env_interaction", "custom",
)

# (min_keys, max_keys) per kind. group_total and min take a *set* of design
# keys - the vocabulary for joint mechanisms ("these components matter
# together", "the weakest of these components dominates") that no
# one-knob-at-a-time basis can express.
_ARITY = {
    "linear": (1, 1), "threshold": (1, 1), "saturating": (1, 1),
    "interaction": (2, 2), "total": (0, 0), "total_sq": (0, 0),
    "equipment": (1, 1), "group_total": (2, 6), "min": (2, 6),
    "env_interaction": (1, 1), "custom": (0, 0),
}


@dataclass(frozen=True)
class Term:
    """
    One candidate mechanism, declaratively.

    kind:
        linear       d[key] / max_key
        threshold    1 if d[key] >= param else 0
        saturating   1 - exp(-d[key] / param), param > 0
        interaction  (d[k1]/max_k1) * (d[k2]/max_k2)
        total        sum(d) / budget       (all numerical keys)
        total_sq     (sum(d) / budget)^2
        equipment    1 if str(e[key]) == option else 0
        group_total  sum over listed keys / sum of their maxes
        min          weakest link: min over listed keys of d[k]/default_k,
                     capped at 1.5 and rescaled to [0, 1]
        env_interaction  (d[key]/max_key) * normalised(env var named in
                     `option`) - a design effect *moderated by* observed
                     per-deployment environment. This is how a proposer edits
                     the causal model beyond design-only structure. Needs env
                     stats at compile time; without them the term is 0.
    sign: prior belief about the coefficient's direction (+1 protective,
        -1 harmful, 0 unknown). Enters as a small prior mean, never a constraint.
    """

    kind: str
    keys: Tuple[str, ...] = ()
    param: float = 0.0
    option: str = ""
    sign: int = 0

    def __post_init__(self):
        if self.kind not in TERM_KINDS:
            raise ValueError(f"unknown term kind {self.kind!r}")
        lo, hi = _ARITY[self.kind]
        if not (lo <= len(self.keys) <= hi):
            raise ValueError(
                f"term {self.kind!r} takes {lo}-{hi} key(s), got {self.keys!r}"
            )
        if len(set(self.keys)) != len(self.keys):
            raise ValueError(f"term {self.kind!r} has duplicate keys {self.keys!r}")
        if self.kind == "saturating" and self.param <= 0:
            raise ValueError("saturating term needs param > 0")
        if self.kind == "equipment" and not self.option:
            raise ValueError("equipment term needs an option value")
        if self.kind == "env_interaction" and not self.option:
            raise ValueError("env_interaction term needs the env variable name "
                             "in 'option'")
        if self.kind == "custom":
            if not self.option:
                raise ValueError("custom term needs an expression in 'option'")
            from .exprterm import compile_expr  # validates syntax eagerly
            compile_expr(self.option)
        if self.sign not in (-1, 0, 1):
            raise ValueError(f"sign must be -1, 0 or 1, got {self.sign!r}")

    @property
    def name(self) -> str:
        if self.kind == "linear":
            return f"lin({self.keys[0]})"
        if self.kind == "threshold":
            return f"thr({self.keys[0]}>={self.param:g})"
        if self.kind == "saturating":
            return f"sat({self.keys[0]}/{self.param:g})"
        if self.kind == "interaction":
            return f"int({self.keys[0]}*{self.keys[1]})"
        if self.kind == "equipment":
            return f"eq({self.keys[0]}={self.option})"
        if self.kind == "group_total":
            return "grp(" + "+".join(self.keys) + ")"
        if self.kind == "min":
            return "min(" + ",".join(self.keys) + ")"
        if self.kind == "env_interaction":
            return f"envx({self.keys[0]}*{self.option})"
        if self.kind == "custom":
            return f"expr({self.option[:48]})"
        return self.kind

    def compile(self, space: DesignSpace,
                env_stats: Optional[Dict[str, Tuple[float, float, float]]] = None
                ) -> Callable[[Dict, Dict], float]:
        keys = tuple(space.def_keys)

        def key_scale(k: str) -> float:
            return float(max(space.numerical.get(k, {}).get("max", 50), 1))

        budget = float(space.total_def_budget
                       or sum(key_scale(k) for k in keys) or 1.0)
        if self.kind == "linear":
            k, s = self.keys[0], key_scale(self.keys[0])
            return lambda d, e, k=k, s=s: d.get(k, 0) / s
        if self.kind == "threshold":
            k, t = self.keys[0], self.param
            return lambda d, e, k=k, t=t: 1.0 if d.get(k, 0) >= t else 0.0
        if self.kind == "saturating":
            k, s = self.keys[0], self.param
            return lambda d, e, k=k, s=s: 1.0 - math.exp(-d.get(k, 0) / s)
        if self.kind == "interaction":
            k1, k2 = self.keys
            s1, s2 = key_scale(k1), key_scale(k2)
            return lambda d, e, k1=k1, k2=k2, s1=s1, s2=s2: \
                (d.get(k1, 0) / s1) * (d.get(k2, 0) / s2)
        if self.kind == "total":
            return lambda d, e, keys=keys, b=budget: \
                sum(d.get(k, 0) for k in keys) / b
        if self.kind == "total_sq":
            return lambda d, e, keys=keys, b=budget: \
                (sum(d.get(k, 0) for k in keys) / b) ** 2
        if self.kind == "group_total":
            gk = self.keys
            denom = max(sum(key_scale(k) for k in gk), 1.0)
            return lambda d, e, gk=gk, denom=denom: \
                sum(d.get(k, 0) for k in gk) / denom
        if self.kind == "min":
            gk = self.keys
            refs = {k: float(max(space.numerical.get(k, {}).get("default", 0), 1))
                    for k in gk}
            return lambda d, e, gk=gk, refs=refs: \
                min(min(d.get(k, 0) / refs[k], 1.5) for k in gk) / 1.5
        if self.kind == "env_interaction":
            k, s = self.keys[0], key_scale(self.keys[0])
            stats = (env_stats or {}).get(self.option)
            if stats is None:
                # Env variable never observed: dead term, mild Occam penalty.
                return lambda d, e: 0.0
            lo, hi, mean = stats
            span = max(hi - lo, 1e-9)
            default = (mean - lo) / span

            def f(d, e, k=k, s=s, name=self.option, lo=lo, span=span,
                  default=default):
                v = e.get(f"env_{name}")
                envn = default if not isinstance(v, (int, float)) \
                    else (float(v) - lo) / span
                return (d.get(k, 0) / s) * envn

            return f
        if self.kind == "custom":
            from .exprterm import compile_expr
            return compile_expr(self.option)
        # equipment
        k, opt = self.keys[0], self.option
        return lambda d, e, k=k, opt=opt: 1.0 if str(e.get(k, "")).lower() == opt.lower() else 0.0

    # -- JSON interface (what the LLM speaks) ---------------------------

    def to_dict(self) -> Dict[str, Any]:
        out: Dict[str, Any] = {"kind": self.kind}
        if self.keys:
            out["keys"] = list(self.keys)
        if self.kind in ("threshold", "saturating"):
            out["param"] = self.param
        if self.kind == "equipment":
            out["option"] = self.option
        if self.sign:
            out["sign"] = self.sign
        return out

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "Term":
        if not isinstance(d, dict):
            raise ValueError(f"term must be an object, got {type(d).__name__}")
        keys = d.get("keys", [])
        if isinstance(keys, str):
            keys = [keys]
        return cls(
            kind=str(d.get("kind", "")),
            keys=tuple(str(k) for k in keys),
            param=float(d.get("param", 0.0)),
            option=str(d.get("option", "")),
            sign=int(d.get("sign", 0)),
        )


@dataclass(frozen=True)
class Hypothesis:
    """A named bundle of candidate mechanisms. Compiles to a FeatureMap."""

    name: str
    terms: Tuple[Term, ...]
    prior_weight: float = 1.0
    source: str = "heuristic"  # "heuristic" | "llm" | "evolved"
    rationale: str = ""

    MAX_TERMS = 8

    def feature_map(self, space: DesignSpace,
                    extra_terms: Optional[Dict[str, Callable]] = None,
                    env_stats: Optional[Dict[str, Tuple[float, float, float]]] = None
                    ) -> FeatureMap:
        terms: Dict[str, Callable] = {}
        for t in self.terms:
            name = t.name
            # Disambiguate accidental duplicates rather than silently dropping.
            while name in terms:
                name += "'"
            terms[name] = t.compile(space, env_stats=env_stats)
        for name, fn in (extra_terms or {}).items():
            if name not in terms:
                terms[name] = fn
        return FeatureMap.from_terms(terms)

    def prior_mean(self, sign_scale: float = 0.5, n_extra: int = 0) -> np.ndarray:
        return np.array([0.0] + [t.sign * sign_scale for t in self.terms]
                        + [0.0] * n_extra)

    def dedupe_key(self) -> Tuple[str, ...]:
        return tuple(sorted(t.name for t in self.terms))

    def validate(self, space: DesignSpace) -> List[str]:
        """Problems that make this hypothesis unusable on this action space."""
        problems: List[str] = []
        if not self.terms:
            problems.append("no terms")
        if len(self.terms) > self.MAX_TERMS:
            problems.append(f"too many terms ({len(self.terms)} > {self.MAX_TERMS})")
        for t in self.terms:
            if t.kind in ("linear", "threshold", "saturating", "interaction",
                          "group_total", "min", "env_interaction"):
                for k in t.keys:
                    if k not in space.numerical:
                        problems.append(f"{t.name}: unknown design key {k!r}")
            if t.kind == "threshold":
                spec = space.numerical.get(t.keys[0], {})
                lo, hi = spec.get("min", 0), spec.get("max", 50)
                if not (lo < t.param <= hi):
                    problems.append(f"{t.name}: threshold outside ({lo}, {hi}]")
            if t.kind == "equipment":
                k = t.keys[0]
                if k not in space.discrete and k not in space.boolean:
                    problems.append(f"{t.name}: unknown equipment key {k!r}")
        return problems

    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "terms": [t.to_dict() for t in self.terms],
            "rationale": self.rationale,
        }

    @classmethod
    def from_dict(cls, d: Dict[str, Any], source: str = "llm") -> "Hypothesis":
        if not isinstance(d, dict):
            raise ValueError(f"hypothesis must be an object, got {type(d).__name__}")
        raw_terms = d.get("terms", [])
        if not isinstance(raw_terms, list):
            raise ValueError("hypothesis 'terms' must be a list")
        terms = tuple(Term.from_dict(t) for t in raw_terms)
        return cls(
            name=str(d.get("name", "unnamed"))[:64],
            terms=terms,
            source=source,
            rationale=str(d.get("rationale", ""))[:500],
        )


def dedupe(hypotheses: Sequence[Hypothesis]) -> List[Hypothesis]:
    """Drop hypotheses whose term sets are identical, keeping the first."""
    seen, out = set(), []
    for h in hypotheses:
        key = h.dedupe_key()
        if key in seen:
            continue
        seen.add(key)
        out.append(h)
    return out


# ----------------------------------------------------------------------
# Environment conditioning (the observational start of C5)
# ----------------------------------------------------------------------

def collect_env_stats(arms: Sequence[Any]) -> Dict[str, Tuple[float, float, float]]:
    """(lo, hi, mean) per numeric env variable observed across >= 2 arms."""
    values: Dict[str, List[float]] = {}
    for a in arms:
        for k, v in (getattr(a, "environment", None) or {}).items():
            if isinstance(v, (int, float)) and not isinstance(v, bool):
                values.setdefault(k, []).append(float(v))
    return {k: (min(vs), max(vs), float(np.mean(vs)))
            for k, vs in values.items() if len(vs) >= 2 and max(vs) > min(vs)}


def env_nuisance_terms(arms: Sequence[Any], max_terms: int = 6) -> Dict[str, Callable]:
    """
    Covariate terms for per-deployment environment variables that varied
    across the observed arms.

    The environment is pre-treatment context reported with each deployment
    (design invariant 3: it confounds mediators, not the randomised design),
    so conditioning on it is valid and removes between-deployment noise that
    otherwise masquerades as a design effect. Continuous variables are
    min-max scaled over the observed range; when a term is evaluated on a
    *candidate* design (no environment yet), it falls back to the observed
    mean, which marginalises the prediction over the environment the stage-2
    fleet will actually face - to first order in the logit.
    """
    values: Dict[str, List[Any]] = {}
    for a in arms:
        for k, v in (getattr(a, "environment", None) or {}).items():
            values.setdefault(k, []).append(v)

    terms: Dict[str, Callable] = {}
    for k in sorted(values):
        if len(terms) >= max_terms:
            break
        vs = values[k]
        if len(vs) < 2:
            continue
        numeric = all(isinstance(v, (int, float)) and not isinstance(v, bool)
                      for v in vs)
        if numeric:
            lo, hi = float(min(vs)), float(max(vs))
            if hi <= lo:
                continue
            default = (float(np.mean(vs)) - lo) / (hi - lo)

            def f(d, e, k=k, lo=lo, hi=hi, default=default):
                v = e.get(f"env_{k}")
                if v is None or not isinstance(v, (int, float)):
                    return default
                return (float(v) - lo) / (hi - lo)

            terms[f"env({k})"] = f
        else:
            counts: Dict[str, int] = {}
            for v in vs:
                counts[str(v)] = counts.get(str(v), 0) + 1
            ref = max(counts, key=counts.get)
            for level, c in sorted(counts.items()):
                if level == ref or len(terms) >= max_terms:
                    continue
                freq = c / len(vs)

                def f(d, e, k=k, level=level, freq=freq):
                    v = e.get(f"env_{k}")
                    if v is None:
                        return freq
                    return 1.0 if str(v) == level else 0.0

                terms[f"env({k}={level})"] = f
    return terms


# ----------------------------------------------------------------------
# C2: evidence-weighted ensemble
# ----------------------------------------------------------------------

@dataclass
class FittedHypothesis:
    hypothesis: Hypothesis
    fmap: FeatureMap
    model: BinomialLogit
    log_evidence: float
    weight: float = 0.0


class HypothesisEnsemble:
    """
    Fit every hypothesis on the same counts, weight by predictive evidence,
    and predict by model averaging.

    Evidence modes:
      * "prequential" (default): sleeping-experts prequential scoring. Each
        hypothesis is scored ONLY on arms observed after it entered the
        population, by one-step-ahead posterior predictive probability
        (fit on arms[:t], predict arm t). For hypotheses present from the
        start this telescopes into the marginal likelihood; a hypothesis
        *bred from the data mid-session* earns nothing from the arms it was
        bred to explain - it wakes at the population's uniform share and must
        predict FUTURE arms to gain weight. This is the defence against
        adaptive hypothesis fishing.
      * "laplace": closed-form Laplace evidence on all arms (legacy; still
        computed in both modes for diagnostics).

    Args:
        hypotheses: candidate mechanism bundles (validated, deduped by caller
            or here).
        space: DesignSpace, for def_keys and validation.
        sign_scale: prior-mean magnitude a sign belief contributes.
    """

    def __init__(self, hypotheses: Sequence[Hypothesis], space: DesignSpace,
                 prior_sd: float = 2.5, sign_scale: float = 0.5,
                 rng: Any = None, evidence_mode: str = "prequential",
                 select: str = "average"):
        if evidence_mode not in ("prequential", "laplace"):
            raise ValueError(f"unknown evidence_mode {evidence_mode!r}")
        if select not in ("average", "map"):
            raise ValueError(f"unknown select {select!r}")
        self.select = select
        self.space = space
        self.def_keys = space.def_keys
        self.prior_sd = prior_sd
        self.sign_scale = sign_scale
        self.evidence_mode = evidence_mode
        self.rng = rng if rng is not None else np.random.default_rng(0)
        self.fitted: List[FittedHypothesis] = []
        self.hypotheses: List[Hypothesis] = []
        self.rejected: List[Tuple[Hypothesis, List[str]]] = []
        self.birth: Dict[Tuple[str, ...], int] = {}
        self._arms_seen = 0
        self._preq_cache: Dict[Tuple[Tuple[str, ...], int], float] = {}
        self.add(hypotheses)

    def add(self, hypotheses: Sequence[Hypothesis]) -> List[Hypothesis]:
        """
        Validate, dedupe against current set, and register. Returns accepted.

        Hypotheses added after data has been observed are stamped with a birth
        index; prequential scoring only credits them for later arms.
        """
        accepted = []
        seen = {h.dedupe_key() for h in self.hypotheses}
        for h in hypotheses:
            problems = h.validate(self.space)
            if problems:
                self.rejected.append((h, problems))
                continue
            key = h.dedupe_key()
            if key in seen:
                continue
            seen.add(key)
            self.hypotheses.append(h)
            self.birth[key] = self._arms_seen
            accepted.append(h)
        return accepted

    def _prefix_evidence(self, h: Hypothesis, fmap: FeatureMap,
                         arms: Sequence[Any], upto: int, n_extra: int) -> float:
        """Laplace log-evidence of arms[:upto] under h (0.0 when upto == 0)."""
        if upto <= 0:
            return 0.0
        key = h.dedupe_key()
        arm = arms[upto - 1]
        cache_key = (key, upto, arm.label, arm.survived, arm.deployed)
        if cache_key not in self._preq_cache:
            model = BinomialLogit(
                prior_sd=self.prior_sd,
                rng=np.random.default_rng(self.rng.integers(2 ** 31)),
                prior_mean=h.prior_mean(self.sign_scale, n_extra=n_extra),
            ).fit(fmap.matrix(arms[:upto]),
                  [a.survived for a in arms[:upto]],
                  [a.deployed for a in arms[:upto]], names=fmap.names)
            self._preq_cache[cache_key] = float(model.log_evidence)
        return self._preq_cache[cache_key]

    def _prequential_weights(self, arms: Sequence[Any],
                             fmaps: Dict[Tuple[str, ...], FeatureMap],
                             n_extra: int) -> Dict[Tuple[str, ...], float]:
        """
        Sleeping-experts weights with closed-form conditional evidence.

        A hypothesis born at arm index b is credited with
            log p(arms[b:] | arms[:b], h) = E_h(T) - E_h(b)
        (both Laplace log-evidences), i.e. only the data that arrived after it
        existed - the anti-fishing correction. It additionally wakes at the
        population's mean weight at its birth time, so a newborn neither
        inherits credit nor starts handicapped. For hypotheses present from
        the start (b = 0) this reduces exactly to the full Laplace evidence,
        so the correction costs nothing when there is nothing to correct.
        """
        T = len(arms)
        entries = []  # (birth, key, h)
        for h in self.hypotheses:
            key = h.dedupe_key()
            entries.append((min(self.birth.get(key, 0), T), key, h))
        entries.sort(key=lambda e: e[0])

        offset: Dict[Tuple[str, ...], float] = {}
        birth_of: Dict[Tuple[str, ...], int] = {}
        for birth, key, h in entries:
            if offset:
                # population weights evaluated at this birth time
                vals = np.array([
                    offset[k2] + self._prefix_evidence(
                        h2, fmaps[k2], arms, birth, n_extra)
                    - self._prefix_evidence(h2, fmaps[k2], arms, birth_of[k2], n_extra)
                    for _, k2, h2 in entries if k2 in offset
                ])
                m = vals.max()
                wake = m + np.log(np.mean(np.exp(vals - m)))
            else:
                wake = 0.0
            offset[key] = wake + np.log(max(h.prior_weight, 1e-12))
            birth_of[key] = birth

        lw: Dict[Tuple[str, ...], float] = {}
        for birth, key, h in entries:
            lw[key] = (offset[key]
                       + self._prefix_evidence(h, fmaps[key], arms, T, n_extra)
                       - self._prefix_evidence(h, fmaps[key], arms, birth, n_extra))
        return lw

    def fit(self, arms: Sequence[Any]) -> "HypothesisEnsemble":
        y = [a.survived for a in arms]
        n = [a.deployed for a in arms]
        extra = env_nuisance_terms(arms)
        env_stats = collect_env_stats(arms)
        self.env_terms = list(extra)
        self.env_stats = env_stats
        self._arms_seen = len(arms)
        self.fitted = []
        fmaps: Dict[Tuple[str, ...], FeatureMap] = {}
        for h in self.hypotheses:
            fmap = h.feature_map(self.space, extra_terms=extra, env_stats=env_stats)
            fmaps[h.dedupe_key()] = fmap
            model = BinomialLogit(
                prior_sd=self.prior_sd,
                rng=np.random.default_rng(self.rng.integers(2 ** 31)),
                prior_mean=h.prior_mean(self.sign_scale, n_extra=len(extra)),
            ).fit(fmap.matrix(arms), y, n, names=fmap.names)
            self.fitted.append(FittedHypothesis(h, fmap, model, model.log_evidence))

        if self.evidence_mode == "prequential" and arms:
            lw = self._prequential_weights(arms, fmaps, n_extra=len(extra))
            log_ev = np.array([lw[f.hypothesis.dedupe_key()] for f in self.fitted])
        else:
            log_ev = np.array([f.log_evidence + np.log(f.hypothesis.prior_weight)
                               for f in self.fitted])
        log_ev = np.where(np.isfinite(log_ev), log_ev, -1e30)
        w = np.exp(log_ev - log_ev.max())
        w = w / w.sum()
        if self.select == "map":
            # Ablation mode: winner-takes-all over hypotheses (the winner's-
            # curse variant model averaging exists to avoid).
            w = np.where(np.arange(len(w)) == int(np.argmax(w)), 1.0, 0.0)
        for f, wi in zip(self.fitted, w):
            f.weight = float(wi)
        return self

    # -- accessors ------------------------------------------------------

    @property
    def weights(self) -> np.ndarray:
        return np.array([f.weight for f in self.fitted])

    def ranking(self, top: int = 10) -> List[FittedHypothesis]:
        return sorted(self.fitted, key=lambda f: f.weight, reverse=True)[:top]

    def prune(self, min_weight: float = 1e-3, keep_at_least: int = 12) -> List[Hypothesis]:
        """
        Selection step of the evolve loop: drop hypotheses the evidence has
        effectively killed, keeping at least `keep_at_least` (by weight) so a
        noisy round cannot wipe out the population. Returns the removed
        hypotheses (the "eliminated" list handed back to the proposer, so
        evolution does not re-propose them).
        """
        self._check_fitted()
        ranked = sorted(self.fitted, key=lambda f: f.weight, reverse=True)
        keep = {id(f.hypothesis) for f in ranked[:keep_at_least]}
        removed = [f.hypothesis for f in self.fitted
                   if id(f.hypothesis) not in keep and f.weight < min_weight]
        if removed:
            gone = {id(h) for h in removed}
            self.hypotheses = [h for h in self.hypotheses if id(h) not in gone]
            self.fitted = [f for f in self.fitted if id(f.hypothesis) not in gone]
            w = self.weights
            for f, wi in zip(self.fitted, w / w.sum()):
                f.weight = float(wi)
        return removed

    def _check_fitted(self):
        if not self.fitted:
            raise RuntimeError("call fit() first")

    # -- prediction -----------------------------------------------------

    def predict_mean_matrix(
        self, candidates: Sequence[Tuple[Dict[str, Any], Dict[str, Any]]]
    ) -> np.ndarray:
        """Plug-in survival probability per hypothesis: (H, C) matrix."""
        self._check_fitted()
        rows = []
        for f in self.fitted:
            X = np.vstack([f.fmap.row(d, e) for d, e in candidates])
            rows.append(f.model.predict(X))
        return np.vstack(rows)

    def predictive_samples(
        self,
        candidates: Sequence[Tuple[Dict[str, Any], Dict[str, Any]]],
        draws: int = 4000,
    ) -> np.ndarray:
        """
        Mixture posterior draws of survival probability: (draws, C).

        Draws are allocated to hypotheses by posterior weight, so the sample
        reflects both within-hypothesis parameter uncertainty and
        between-hypothesis structural uncertainty.
        """
        self._check_fitted()
        counts = self.rng.multinomial(draws, self.weights)
        chunks = []
        for f, k in zip(self.fitted, counts):
            if k == 0:
                continue
            X = np.vstack([f.fmap.row(d, e) for d, e in candidates])
            chunks.append(f.model.predictive_samples(X, draws=int(k)))
        samples = np.vstack(chunks)
        self.rng.shuffle(samples, axis=0)
        return samples

    def coefficient_summary(self, top: int = 5) -> List[Dict[str, Any]]:
        """Compact fit summary, e.g. for the LLM's evolve step."""
        out = []
        for f in self.ranking(top):
            out.append({
                "hypothesis": f.hypothesis.name,
                "weight": round(f.weight, 4),
                "log_evidence": round(f.log_evidence, 2),
                "coefficients": {
                    k: {"mean": round(v["mean"], 3), "z": round(v["z"], 2)}
                    for k, v in f.model.coefficients().items()
                },
            })
        return out
