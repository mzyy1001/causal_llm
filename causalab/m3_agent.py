"""
M3: 3-factor factorial on the frozen CausaLab env (hash 1fd9817e1c6860dd).

Factors:
  acq  in {llm, engine}  - who selects the intervention PLAN (2 stages x 10)
  sub  in {llm, engine}  - who selects the final frequency prediction
  comp in {screen, neutral} - the identification component: balanced
        one-property-at-a-time screening menu vs matched-size random menu

Shared across all cells: the fitted model class (per-target regression over
linear+quadratic parent-subset hypotheses, BIC-weighted), all information
shown, budgets (2 obs + 4(k-1) interventions), stateless LLM prompts,
menu-only actions. ITT: invalid controller output after one retry ->
deterministic default (first menu option) recorded as controller failure.
"""

import itertools
import json
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from causalab.env import CausaLabEnv
from causal_agent.propose import openai_chat_complete


def make_llm(model):
    from mpi_transport import completion
    return completion(model, temperature=0.4)


def llm_indices(complete, prompt, n_opts, n_pick):
    for _ in range(2):
        try:
            r = complete(prompt)
            js = json.loads(r[r.find("{"):r.rfind("}") + 1])
            idx = [int(i) for i in js["menu_indices"]][:n_pick]
            if all(0 <= i < n_opts for i in idx) and len(idx) == n_pick:
                return idx, "ok"
        except Exception:
            continue
    return None, "controller"


# ---------------- shared engine: BIC-weighted subset regression ----------

def fit_target(data: List[Dict[int, float]], target: int, k: int,
               max_parents: int = 3):
    """BIC-weighted hypotheses over (parent subset, family) for one target."""
    X = np.array([[row[v] for v in range(k)] for row in data])
    y = X[:, target]
    others = [v for v in range(k) if v != target]
    hyps = []
    for r in range(0, max_parents + 1):
        for subset in itertools.combinations(others, r):
            for fam in (("linear",) if r == 0 else ("linear", "quadratic")):
                cols = [np.ones(len(y))]
                for u in subset:
                    cols.append(X[:, u] if fam == "linear" else X[:, u] ** 2 / 2)
                A = np.vstack(cols).T
                beta, *_ = np.linalg.lstsq(A, y, rcond=None)
                resid = y - A @ beta
                sigma2 = max(float(resid @ resid) / len(y), 1e-9)
                bic = len(y) * np.log(sigma2) + (r + 1) * np.log(len(y))
                hyps.append({"subset": subset, "family": fam,
                             "beta": beta, "bic": bic})
    b = np.array([h["bic"] for h in hyps])
    w = np.exp(-(b - b.min()) / 2)
    w /= w.sum()
    for h, wi in zip(hyps, w):
        h["weight"] = float(wi)
    hyps.sort(key=lambda h: -h["weight"])
    return hyps


def predict_y(hyps, props: Dict[int, float], k: int, top: int = 8):
    preds, ws = [], []
    for h in hyps[:top]:
        val = h["beta"][0]
        for j, u in enumerate(h["subset"]):
            x = props[u]
            val += h["beta"][j + 1] * (x if h["family"] == "linear" else x * x / 2)
        preds.append(float(val))
        ws.append(h["weight"])
    ws = np.array(ws) / sum(ws)
    return preds, ws


# ---------------- intervention menus ------------------------------------

def build_menu(k: int, comp: str, rng) -> List[Tuple[int, float]]:
    if comp == "screen":  # balanced: every property at +/-2 twice
        base = [(p, v) for p in range(k - 1) for v in (-2.0, 2.0)]
        return base + base  # 4(k-1) items
    # neutral = CONCENTRATED: matched size, but coverage-starved (all budget
    # on 2 random properties). Pilot: uniform-random matched menus showed NO
    # deficit vs screening (random designs identify regressions well); the
    # component's causal value is COVERAGE insurance, so the neutral scaffold
    # is the coverage-starved analogue.
    props = rng.choice(k - 1, size=2, replace=False)
    return [(int(rng.choice(props)), float(np.round(rng.uniform(-4, 4), 1)))
            for _ in range(4 * (k - 1))]


def run_episode(seed, acq, sub, comp, model, complete=None):
    env = CausaLabEnv(k=10, seed=seed)
    rng = np.random.default_rng(seed + 7)
    k = env.k
    tel = {"acq_fail": 0, "sub_fail": 0}
    data = []
    for _ in range(env.obs_budget):
        o = env.observe()
        data.append({**{int(v): x for v, x in o["properties"].items()},
                     k - 1: o["frequency"]})
    menu = build_menu(k, comp, rng)
    # two planning stages of half the budget each
    half = env.int_budget // 2
    for stage in range(2):
        opts = list(range(len(menu)))
        if acq == "engine":
            # engine policy: sequential coverage of the menu (screen menu is
            # balanced by construction; neutral menu taken as-is)
            picks = opts[stage * half:(stage + 1) * half]
        else:
            summary = "\n".join(
                f"obs/int record {i}: props={{{', '.join(f'{v}:{row[v]:+.2f}' for v in range(k-1))}}} freq={row[k-1]:+.2f}"
                for i, row in enumerate(data[-6:]))
            menu_txt = "\n".join(f"  [{i}] set property {p} to {v:+.1f}"
                                 for i, (p, v) in enumerate(menu))
            idx, status = llm_indices(complete, (
                "You control experiment selection in a causal-discovery lab. "
                f"Hidden SCM over properties 0-{k-2} and frequency Y. Choose "
                f"{half} interventions (menu indices) to identify Y's "
                "mechanism.\nRecent data:\n" + summary + "\nMenu:\n" + menu_txt
                + f'\nReply JSON ONLY: {{"menu_indices": [{half} ints]}}'),
                len(menu), half)
            if idx is None:
                tel["acq_fail"] += 1
                picks = opts[stage * half:(stage + 1) * half]
            else:
                picks = idx
        for i in picks:
            p, v = menu[i]
            r = env.intervene(p, v)
            if r is None:
                break
            data.append({**{int(a): b for a, b in r["properties"].items()},
                         k - 1: r["frequency"]})
    # fit shared model, build prediction candidates
    hyps = fit_target(data, k - 1, k)
    props = env.reactor_properties()
    preds, ws = predict_y(hyps, props, k)
    ranked = sorted(zip(preds, ws), key=lambda t: -t[1])[:5]
    if sub == "engine":
        pred = float(np.average([p for p, _ in ranked],
                                weights=[w for _, w in ranked]))
    else:
        opts_txt = "\n".join(f"  [{i}] predict {p:+.2f} (model weight {w:.2f})"
                             for i, (p, w) in enumerate(ranked))
        idx, status = llm_indices(complete, (
            "You control the FINAL prediction of the reactor crystal's "
            f"frequency. Reactor properties: "
            + ", ".join(f"{v}:{props[v]:+.2f}" for v in range(k - 1))
            + "\nModel-scored candidate predictions:\n" + opts_txt
            + '\nReply JSON ONLY: {"menu_indices": [<one int>]}'), len(ranked), 1)
        if idx is None:
            tel["sub_fail"] += 1
            pred = float(ranked[0][0])
        else:
            pred = float(ranked[idx[0]][0])
    # mechanism report from top hypothesis (graph edges into Y only - the
    # scored mechanism; property-property edges not claimed)
    top_h = hyps[0]
    edges = [(int(u), k - 1) for u in top_h["subset"]]
    coefs = {f"{u}->{k-1}": float(top_h["beta"][j + 1])
             for j, u in enumerate(top_h["subset"])}
    res = env.submit(pred, graph_edges=edges, coef_estimates=coefs)
    return {**res, **tel, "acq": acq, "sub": sub, "comp": comp,
            "model": model, "seed": seed}


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--blocks", type=int, default=8)
    ap.add_argument("--seed0", type=int, default=0)
    ap.add_argument("--model", default="gpt-5.4-mini")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    complete = make_llm(args.model)
    cells = list(itertools.product(["llm", "engine"], ["llm", "engine"],
                                   ["screen", "neutral"]))
    rows = []
    rng = np.random.default_rng(999)
    for b in range(args.seed0, args.seed0 + args.blocks):
        order = cells[:]
        rng.shuffle(order)
        for acq, sub, comp in order:
            rows.append(run_episode(b, acq, sub, comp, args.model, complete))
        Path(args.out).write_text(json.dumps(rows, indent=1))
        print(f"block {b} done", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
