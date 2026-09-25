"""
Hypothesis proposers: where the LLM attaches (C1), and only here.

A proposer turns (action space, observations so far) into candidate Hypothesis
objects. Two implementations:

  * HeuristicProposer - LLM-free library over the action space. This is the
    control condition: whatever the LLM proposer scores is measured against it
    on the same drone budget.
  * LLMProposer - prompts a language model for mechanisms in the declarative
    JSON grammar of hypotheses.Term, parses strictly, and validates against
    the action space. Anything malformed is dropped, never repaired into code.
    The LLM proposes and evolves *features*; the evidence weighting in
    hypotheses.HypothesisEnsemble decides what is causal.

Both expose the same two calls:
    propose(space, ctx)  - initial hypothesis set
    evolve(space, ctx)   - refinements after data has arrived (ctx carries the
                           arms observed and the current hypothesis ranking)

Transports for LLMProposer are plain `prompt -> text` callables, so tests stub
them and the same proposer runs against any OpenAI-compatible endpoint or the
local `claude` CLI.
"""

import json
import re
import subprocess
import tempfile
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Sequence

from .design import DesignSpace
from .hypotheses import Hypothesis, Term, dedupe


@dataclass
class ProposalContext:
    """What a proposer is allowed to see. No per-drone records, no file paths."""

    victory_threshold: float = 0.75
    drones_remaining: int = 200
    deployments_remaining: int = 10
    arms: List[Dict[str, Any]] = field(default_factory=list)  # counts only
    ranking: List[Dict[str, Any]] = field(default_factory=list)  # ensemble summary
    eliminated: List[str] = field(default_factory=list)  # pruned hypothesis names
    history_variables: List[str] = field(default_factory=list)  # names only


# ----------------------------------------------------------------------
# Heuristic (LLM-free control)
# ----------------------------------------------------------------------

class HeuristicProposer:
    """
    Assumption-light library: for every design knob, the data gets to choose
    between "irrelevant", "linear", "threshold somewhere", "saturating"; plus
    an equipment indicator per non-default option. Every hypothesis carries the
    total-DEF terms so the armour->agility channel is never aliased into a
    component effect (see runner.py's factor D note).
    """

    def __init__(self, thresholds=(0.25, 0.5, 0.75), max_equipment: int = 6):
        # Equipment/discrete-choice hypotheses are safe to propose since the
        # candidate pool materialises referenced options only as one-knob
        # contrasts on default-proportional backgrounds (no blind crossing).
        self.thresholds = thresholds
        self.max_equipment = max_equipment

    @staticmethod
    def _base_terms() -> List[Term]:
        return [Term("total"), Term("total_sq", sign=-1)]

    def propose(self, space: DesignSpace, ctx: ProposalContext) -> List[Hypothesis]:
        out: List[Hypothesis] = []
        base = self._base_terms()

        out.append(Hypothesis("baseline_total", tuple(base), source="heuristic",
                              rationale="armour only matters in aggregate"))
        linear_terms = [Term("linear", (k,)) for k in space.def_keys]
        # One joint linear hypothesis mirrors the scaffold's default basis.
        out.append(Hypothesis(
            "linear_all", tuple(linear_terms[:6] + base), source="heuristic",
            rationale="independent linear effect per component"))

        # Joint-structure hypotheses. The only prior used is the action
        # space's own default allocation: keys with above-median defaults are
        # a candidate "load-bearing" group. That is metadata the backend hands
        # every agent, not scenario ground truth.
        defaults = {k: space.numerical[k].get("default", 0) for k in space.def_keys}
        if len(defaults) >= 3:
            med = sorted(defaults.values())[len(defaults) // 2]
            core = tuple(k for k, v in defaults.items() if v >= med)[:6]
            rest = tuple(k for k in space.def_keys if k not in core)
            if 2 <= len(core):
                out.append(Hypothesis(
                    "core_spread",
                    (Term("group_total", core, sign=1), Term("total_sq", sign=-1)),
                    source="heuristic",
                    rationale="components with big default allocations carry survival"))
                out.append(Hypothesis(
                    "core_weakest_link",
                    (Term("min", core, sign=1), *base),
                    source="heuristic",
                    rationale="survival tracks the least-armoured core component"))
            for k in rest:
                if 2 <= len(core) and len(core) + 1 <= 6:
                    out.append(Hypothesis(
                        f"core_not_{k}",
                        (Term("group_total", core, sign=1),
                         Term("linear", (k,)), Term("total_sq", sign=-1)),
                        source="heuristic",
                        rationale=f"core armour helps; {k} has its own effect"))
        out.append(Hypothesis(
            "balance_all",
            (Term("min", tuple(space.def_keys[:6]), sign=1), *base),
            source="heuristic",
            rationale="survival tracks the weakest component overall"))

        for k in space.def_keys:
            spec = space.numerical.get(k, {})
            lo, hi = spec.get("min", 0), spec.get("max", 50)
            out.append(Hypothesis(
                f"lin_{k}", (Term("linear", (k,)), *base), source="heuristic"))
            for frac in self.thresholds:
                t = lo + (hi - lo) * frac
                if t <= lo:
                    continue
                out.append(Hypothesis(
                    f"thr_{k}_{t:g}",
                    (Term("threshold", (k,), param=float(t)), *base),
                    source="heuristic"))
            sat_scale = max((hi - lo) / 3.0, 1.0)
            out.append(Hypothesis(
                f"sat_{k}", (Term("saturating", (k,), param=sat_scale), *base),
                source="heuristic"))

        eq_added = 0
        for key, spec in list(space.boolean.items()) + list(space.discrete.items()):
            options = spec.get("options")
            if options is None:
                options = ["true", "false"] if key in space.boolean else []
            default = str(spec.get("default", "")).lower()
            for opt in options:
                if str(opt).lower() == default or eq_added >= self.max_equipment:
                    continue
                out.append(Hypothesis(
                    f"eq_{key}_{opt}",
                    (Term("equipment", (key,), option=str(opt)), *self._base_terms()),
                    source="heuristic"))
                eq_added += 1
        return dedupe(out)

    def evolve(self, space: DesignSpace, ctx: ProposalContext) -> List[Hypothesis]:
        """
        Data-driven refinement: for threshold mechanisms that lead the
        posterior, add half-step neighbours so the breakpoint can localise.
        """
        out: List[Hypothesis] = []
        for entry in ctx.ranking[:3]:
            name = entry.get("hypothesis", "")
            m = re.match(r"thr_(.+)_([0-9.]+)$", name)
            if not m:
                continue
            key, t = m.group(1), float(m.group(2))
            spec = space.numerical.get(key, {})
            lo, hi = spec.get("min", 0), spec.get("max", 50)
            step = (hi - lo) * 0.125
            for t2 in (t - step, t + step):
                if lo < t2 <= hi:
                    out.append(Hypothesis(
                        f"thr_{key}_{t2:g}",
                        (Term("threshold", (key,), param=float(t2)),
                         *self._base_terms()),
                        source="evolved"))
        return dedupe(out)


# ----------------------------------------------------------------------
# LLM proposer
# ----------------------------------------------------------------------

_GRAMMAR = """Each hypothesis is a JSON object:
  {"name": "<short_snake_case>", "rationale": "<one line>", "terms": [<term>, ...]}
A term is one of (2-4 terms per hypothesis; "sign" is optional: 1 if the term
should RAISE survival, -1 if it should LOWER it, omit if unknown):
  {"kind": "linear",      "keys": ["<design_key>"], "sign": 1}
  {"kind": "threshold",   "keys": ["<design_key>"], "param": <number>, "sign": 1}
  {"kind": "saturating",  "keys": ["<design_key>"], "param": <number > 0>, "sign": 1}
  {"kind": "interaction", "keys": ["<key1>", "<key2>"]}
  {"kind": "total"}                       (sum of all design points)
  {"kind": "total_sq", "sign": -1}        (its square, e.g. weight penalty)
  {"kind": "group_total", "keys": ["<k1>", "<k2>", ...], "sign": 1}
      (2-6 keys: armour summed over a set of components that matter jointly,
       e.g. the structurally vital parts of the drone)
  {"kind": "min", "keys": ["<k1>", "<k2>", ...], "sign": 1}
      (2-6 keys: weakest-link - survival tracks the LEAST armoured of the set)
  {"kind": "equipment",   "keys": ["<equipment_key>"], "option": "<value>"}
  {"kind": "custom", "option": "<math expression>", "sign": 1}
      (define a NEW feature the grammar lacks, as a bounded math expression
       over d['<design_key>'], e['<equipment_key>'], env['<env_var>'] using
       + - * / min max abs exp log sqrt sigmoid step and comparisons, e.g.
       "sigmoid((d['camera_def'] - 15) / 5) * step(d['engine_def'] - 10)".
       Keep values roughly in [-1, 1]; unbounded expressions are rejected.)
  {"kind": "env_interaction", "keys": ["<design_key>"], "option": "<env_variable>"}
      (a design effect MODERATED by an observed per-deployment environment
       variable, e.g. armour that only matters in high wind. Use only env
       variable names listed as observed. Pair it with the plain term for the
       same key so main effect and moderation separate.)"""

_PROPOSE_PROMPT = """You are the hypothesis-generation module of a causal-discovery agent.

The agent designs a drone, deploys small batches into a hostile canyon, observes
only (deployed, survived) counts per batch, and must submit one final design
that clears a {threshold:.0%} survival threshold over 1000 drones. Budget:
{drones} drones across {deploys} deployments. The agent chooses designs freely,
so treatment is randomised; your job is ONLY to name candidate MECHANISMS by
which design choices could causally affect survival.

Action space (design knobs and equipment):
{space}

{observations}

Think about what each name plausibly does physically (e.g. an antenna transmits
- could make drones detectable; heavy total armour could reduce agility; a
camera might need a minimum level to function). Propose {n} DIVERSE hypotheses
covering different mechanisms, functional forms and components - including ones
that would surprise a naive designer. Do not assume any specific number is
special; propose thresholds only at round plausible values within each knob's
range.

{grammar}

Reply with ONLY a JSON array of hypothesis objects, no other text."""

_EVOLVE_PROMPT = """You are the evolution module of a causal-discovery agent (same setting as
before: only (deployed, survived) counts are observable, final design must
clear {threshold:.0%} over 1000 drones; {drones} drones and {deploys}
deployments remain).

Action space:
{space}

Deployments so far (design -> survived/deployed):
{arms}

The hypothesis population is under selection by Bayesian model evidence.
Current survivors, best first (coefficients are log-odds effects, z is signal
strength):
{ranking}

Eliminated by the evidence (do NOT re-propose these or trivial variants):
{eliminated}

Breed the next generation: up to {n} NEW hypotheses. Favour
  * mutations of the leaders (shift a breakpoint, swap linear<->saturating,
    add or drop one term),
  * recombinations of two leaders that both show signal,
  * and, if the leaders all explain the data poorly (low survival everywhere),
    fresh mechanisms unlike anything above.

{grammar}

Reply with ONLY a JSON array of hypothesis objects, no other text."""


def _space_summary(space: DesignSpace) -> str:
    parts = {"design_points": {
        k: {kk: v[kk] for kk in ("min", "max", "default") if kk in v}
        for k, v in space.numerical.items()}}
    if space.total_def_budget is not None:
        parts["total_design_budget"] = space.total_def_budget
    if space.discrete:
        parts["discrete_equipment"] = space.discrete
    if space.boolean:
        parts["boolean_equipment"] = space.boolean
    return json.dumps(parts, indent=1, default=str)


def _arms_summary(ctx: ProposalContext) -> str:
    if not ctx.arms:
        return ""
    lines = []
    for a in ctx.arms:
        design = {k: v for k, v in a.items()
                  if k not in ("label", "deployed", "survived", "rate")
                  and not k.startswith("eq_") and not k.startswith("env_")}
        env = {k[4:]: (round(v, 1) if isinstance(v, float) else v)
               for k, v in a.items() if k.startswith("env_")}
        env_note = f"  env={env}" if env else ""
        lines.append(f"  {design} -> {a['survived']}/{a['deployed']}{env_note}")
    if ctx.history_variables:
        lines.append("Observed environment variables (reported per deployment, "
                     "not controllable): " + ", ".join(ctx.history_variables))
    return "\n".join(lines)


def extract_json_array(text: str) -> List[Any]:
    """First parseable JSON array in the reply; [] if none."""
    # Strip fenced blocks first, then fall back to bracket matching.
    fenced = re.findall(r"```(?:json)?\s*(\[.*?\])\s*```", text, re.DOTALL)
    candidates = fenced + re.findall(r"\[.*\]", text, re.DOTALL)
    for cand in candidates:
        try:
            parsed = json.loads(cand)
            if isinstance(parsed, list):
                return parsed
        except json.JSONDecodeError:
            continue
    return []


class LLMProposer:
    """
    Args:
        complete: prompt -> reply text. See openai_chat_complete / claude_cli_complete.
        n_propose / n_evolve: how many hypotheses to request.
        verbose: print parse/validation failures.
    """

    def __init__(self, complete: Callable[[str], str], n_propose: int = 12,
                 n_evolve: int = 6, verbose: bool = False):
        self.complete = complete
        self.n_propose = n_propose
        self.n_evolve = n_evolve
        self.verbose = verbose
        self.transcript: List[Dict[str, str]] = []

    def _ask(self, prompt: str, source: str) -> List[Hypothesis]:
        try:
            reply = self.complete(prompt)
        except Exception as exc:  # transport failure must never kill a session
            if self.verbose:
                print(f"[llm] transport error: {exc}")
            return []
        self.transcript.append({"prompt": prompt, "reply": reply})
        out: List[Hypothesis] = []
        for item in extract_json_array(reply):
            try:
                h = Hypothesis.from_dict(item, source=source)
            except (ValueError, TypeError, KeyError) as exc:
                if self.verbose:
                    print(f"[llm] dropped malformed hypothesis: {exc}")
                continue
            # Every hypothesis carries a total-DEF control: without it, "this
            # component matters" is aliased with "more armour overall", and the
            # armour->agility channel gets credited to whatever the LLM named.
            kinds = {t.kind for t in h.terms}
            if not kinds & {"total", "total_sq", "group_total"} \
                    and len(h.terms) <= Hypothesis.MAX_TERMS - 2:
                h = Hypothesis(h.name,
                               h.terms + (Term("total"), Term("total_sq", sign=-1)),
                               source=h.source, rationale=h.rationale)
            out.append(h)
        return dedupe(out)

    def propose(self, space: DesignSpace, ctx: ProposalContext) -> List[Hypothesis]:
        obs = _arms_summary(ctx)
        prompt = _PROPOSE_PROMPT.format(
            threshold=ctx.victory_threshold, drones=ctx.drones_remaining,
            deploys=ctx.deployments_remaining, space=_space_summary(space),
            observations=(f"Deployments so far:\n{obs}" if obs else
                          "No deployments yet."),
            n=self.n_propose, grammar=_GRAMMAR)
        return self._ask(prompt, source="llm")

    def evolve(self, space: DesignSpace, ctx: ProposalContext) -> List[Hypothesis]:
        prompt = _EVOLVE_PROMPT.format(
            threshold=ctx.victory_threshold, drones=ctx.drones_remaining,
            deploys=ctx.deployments_remaining, space=_space_summary(space),
            arms=_arms_summary(ctx) or "  (none)",
            ranking=json.dumps(ctx.ranking, indent=1, default=str),
            eliminated=", ".join(ctx.eliminated) or "(none yet)",
            n=self.n_evolve, grammar=_GRAMMAR)
        return self._ask(prompt, source="llm-evolved")


# ----------------------------------------------------------------------
# Transports
# ----------------------------------------------------------------------

def openai_chat_complete(base_url: str, api_key: str, model: str,
                         temperature: float = 0.7, timeout: int = 120,
                         **extra: Any) -> Callable[[str], str]:
    """OpenAI-compatible /chat/completions transport (works with micuapi)."""
    import requests

    url = base_url.rstrip("/") + "/chat/completions"

    def complete(prompt: str) -> str:
        payload = {"model": model, "temperature": temperature,
                   "messages": [{"role": "user", "content": prompt}], **extra}
        resp = requests.post(
            url, json=payload, timeout=timeout,
            headers={"Authorization": f"Bearer {api_key}"})
        resp.raise_for_status()
        return resp.json()["choices"][0]["message"]["content"]

    return complete


def claude_cli_complete(model: str = "claude-haiku-4-5-20251001",
                        timeout: int = 300) -> Callable[[str], str]:
    """
    Local `claude -p` transport. Runs from an empty temp directory so the
    model has no line of sight to this repo (experiments/ holds the answer
    key - design invariant 6).
    """

    def complete(prompt: str) -> str:
        workdir = tempfile.mkdtemp(prefix="causal-llm-")
        result = subprocess.run(
            ["claude", "-p", "--model", model],
            input=prompt, capture_output=True, text=True,
            timeout=timeout, cwd=workdir)
        if result.returncode != 0:
            raise RuntimeError(f"claude CLI failed: {result.stderr[:500]}")
        return result.stdout

    return complete
