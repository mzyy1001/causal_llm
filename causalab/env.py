"""
CausaLab-style environment, reimplemented strictly from the published
specification of arXiv:2605.26029 (the official repo was unavailable at
implementation time). FROZEN once hashed - no agent-informed tuning.

Spec elements implemented (quotes paraphrased from the paper):
  * "sample a DAG G over V = O ∪ {Y}, assign root nodes from their exogenous
    sources, then compute non-root variables in topological order"
  * "exactly two structural-equation families: linear and quadratic"
  * for k-node graphs: observation budget 2, intervention budget 4(k-1)
  * actions: set one controllable non-frequency property on the manipulator
    crystal; the environment recomputes that crystal's measurement
  * task: predict the resonance frequency (Y) of a held-out reactor crystal
    governed by the same SCM
  * metrics: frequency accuracy (exact-match rate within tolerance), graph
    precision/recall/F1, SHD, coefficient F1, root identification

Design decisions where the paper is silent (fixed before any run):
  * property values ~ Uniform(-3, 3) at roots; interventions clamp to [-5, 5]
  * coefficients sampled from {-2, -1, 1, 2} (discrete, so coefficient-F1 is
    well defined); quadratic terms use coefficient/2 * parent^2
  * deterministic mechanisms (exact prediction is meaningful); tolerance 0.5
  * reactor crystal: fresh exogenous draw; its non-frequency properties are
    visible, Y hidden
"""

import hashlib
import json
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np


@dataclass
class Episode:
    k: int
    parents: Dict[int, List[int]]           # node -> parent list (Y = k-1)
    families: Dict[int, str]                # node -> "linear" | "quadratic"
    coefs: Dict[int, Dict[int, float]]      # node -> {parent: coefficient}
    intercepts: Dict[int, float]
    roots: List[int]


class CausaLabEnv:
    """
    One episode = one hidden SCM. Property nodes 0..k-2 ("non-frequency
    properties"), node k-1 = Y (resonance frequency).
    """

    def __init__(self, k: int = 6, seed: int = 0, edge_prob: float = 0.5,
                 tol: float = 0.5):
        self.k = k
        self.tol = tol
        self.rng = np.random.default_rng(seed)
        self.episode = self._sample_scm(edge_prob)
        self.obs_budget = 2
        self.int_budget = 4 * (k - 1)
        self.obs_used = 0
        self.int_used = 0
        self._manip = self._draw_exogenous()
        self._reactor = self._draw_exogenous()
        self._final: Optional[Dict[str, Any]] = None

    # ---------------- SCM ---------------------------------------------
    def _sample_scm(self, edge_prob: float) -> Episode:
        k = self.k
        order = list(self.rng.permutation(k - 1)) + [k - 1]  # Y last
        parents: Dict[int, List[int]] = {v: [] for v in range(k)}
        for i, v in enumerate(order):
            for u in order[:i]:
                if self.rng.random() < edge_prob:
                    parents[v].append(int(u))
        # Y must have at least one parent
        if not parents[k - 1]:
            parents[k - 1].append(int(order[self.rng.integers(k - 1)]))
        families = {v: ("linear" if self.rng.random() < 0.5 else "quadratic")
                    for v in range(k) if parents[v]}
        coefs = {v: {int(u): float(self.rng.choice([-2, -1, 1, 2]))
                     for u in parents[v]} for v in range(k) if parents[v]}
        intercepts = {v: float(np.round(self.rng.uniform(-1, 1), 2))
                      for v in range(k) if parents[v]}
        roots = [v for v in range(k) if not parents[v]]
        self._order = order
        return Episode(k, parents, families, coefs, intercepts, roots)

    def _draw_exogenous(self) -> Dict[int, float]:
        return {v: float(np.round(self.rng.uniform(-3, 3), 2))
                for v in self.episode.roots}

    def _evaluate(self, exogenous: Dict[int, float],
                  do: Optional[Dict[int, float]] = None) -> Dict[int, float]:
        do = do or {}
        vals: Dict[int, float] = {}
        for v in self._order:
            if v in do:
                vals[v] = float(do[v])
            elif v in self.episode.roots:
                vals[v] = float(exogenous[v])
            else:
                total = self.episode.intercepts[v]
                for u, c in self.episode.coefs[v].items():
                    x = vals[u]
                    total += c * x if self.episode.families[v] == "linear" \
                        else (c / 2.0) * x * x
                vals[v] = float(np.round(total, 3))
        return vals

    # ---------------- agent API ----------------------------------------
    def observe(self) -> Optional[Dict[str, Any]]:
        """One observational sample (fresh exogenous draw), all values."""
        if self.obs_used >= self.obs_budget:
            return None
        self.obs_used += 1
        vals = self._evaluate(self._draw_exogenous())
        return {"properties": {v: vals[v] for v in range(self.k - 1)},
                "frequency": vals[self.k - 1]}

    def intervene(self, prop: int, value: float) -> Optional[Dict[str, Any]]:
        """Set one manipulator property; recompute its measurement."""
        if self.int_used >= self.int_budget:
            return None
        if not (0 <= prop < self.k - 1):
            raise ValueError("property index out of range")
        self.int_used += 1
        value = float(np.clip(value, -5, 5))
        vals = self._evaluate(self._manip, do={prop: value})
        return {"do": {prop: value},
                "properties": {v: vals[v] for v in range(self.k - 1)},
                "frequency": vals[self.k - 1]}

    def reactor_properties(self) -> Dict[int, float]:
        vals = self._evaluate(self._reactor)
        return {v: vals[v] for v in range(self.k - 1)}

    def submit(self, frequency_prediction: float,
               graph_edges: Optional[List[Tuple[int, int]]] = None,
               coef_estimates: Optional[Dict[str, float]] = None
               ) -> Dict[str, Any]:
        """Final prediction (once); returns full metric set."""
        if self._final is not None:
            return self._final
        truth = self._evaluate(self._reactor)[self.k - 1]
        acc = int(abs(frequency_prediction - truth) <= self.tol)
        true_edges = {(u, v) for v, ps in self.episode.parents.items()
                      for u in ps}
        pred_edges = set(map(tuple, graph_edges or []))
        tp = len(true_edges & pred_edges)
        prec = tp / len(pred_edges) if pred_edges else 0.0
        rec = tp / len(true_edges) if true_edges else 0.0
        f1 = 2 * prec * rec / (prec + rec) if prec + rec else 0.0
        shd = len(true_edges ^ pred_edges)
        coef_tp = 0
        if coef_estimates:
            for key, est in coef_estimates.items():
                u, v = map(int, key.split("->"))
                true_c = self.episode.coefs.get(v, {}).get(u)
                if true_c is not None and abs(est - true_c) <= 0.25:
                    coef_tp += 1
        n_true_coefs = sum(len(c) for c in self.episode.coefs.values())
        coef_f1 = (2 * coef_tp / (len(coef_estimates or {}) + n_true_coefs)
                   if (coef_estimates or n_true_coefs) else 0.0)
        self._final = {
            "frequency_pred": frequency_prediction, "frequency_true": truth,
            "accuracy": acc, "graph_precision": round(prec, 3),
            "graph_recall": round(rec, 3), "graph_f1": round(f1, 3),
            "shd": shd, "coef_f1": round(coef_f1, 3),
        }
        return self._final


def env_spec_hash() -> str:
    """Hash of this file - freeze marker recorded before any agent run."""
    with open(__file__, "rb") as f:
        return hashlib.sha256(f.read()).hexdigest()[:16]
