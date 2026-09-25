"""
MPI factorial on the frozen CausaLab adaptation
(hash 1fd9817e1c6860dd; causalab/env.py untouched).

Matched-menu protocol (m3_agent.py supplies shared numerical helpers):

1. Common action support at SUBMISSION: the prediction menu is the 5
   top-weight candidates PLUS the BIC-weighted average as an explicit
   6th option.  Both selectors choose an index from this identical menu,
   displayed at identical precision; the enacted value is the exact
   unrounded value of the chosen option for BOTH selectors.
   Engine policy (frozen): choose the weighted-average option.
   LLM-failure ITT default: index 0 (top-weight candidate; matches v1).
2. Paired blinded shadow recommendations at BOTH stages: both selectors'
   recommendations are computed before enactment in every cell and
   logged in decision_log; preassigned authority determines enactment.
   Shadow failures do not affect enactment and are logged, not counted
   in acq_fail/sub_fail (which count enactment-relevant failures only).
3. decision_log per episode: two acquisition entries
   {stage, kind:"acq", engine:[...], llm:[...]|None, llm_status,
    enacted:[...]} and one submission entry
   {kind:"sub", menu:[displayed strings], engine:int, llm:int|None,
    llm_status, enacted:int}.

Unchanged: env parameters and budgets, fit_target / predict_y,
build_menu, the engine acquisition policy (sequential coverage slices),
acquisition prompts, ITT rules, the 2x2x2 (acq x sub x comp) cell
structure, mechanism report from the top hypothesis.

Transport: OpenAI-compatible API configured through environment variables;
temperature 0.4 for gpt-5.4-mini (as all prior mini runs), 1.0 for
gpt-5.6-sol (the API rejects non-default values for this model --
disclosed deviation).
"""

import itertools
import json
import os
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from causalab.env import CausaLabEnv
from causalab.m3_agent import (build_menu, fit_target, llm_indices,
                               predict_y)
from causal_agent.propose import openai_chat_complete


def make_llm(model):
    temp = 0.4 if "mini" in model else 1.0
    from mpi_transport import completion
    return completion(model, temperature=temp)


def run_episode(seed, acq, sub, comp, model, complete):
    env = CausaLabEnv(k=10, seed=seed)
    rng = np.random.default_rng(seed + 7)
    k = env.k
    tel = {"acq_fail": 0, "sub_fail": 0}
    decision_log = []
    data = []
    for _ in range(env.obs_budget):
        o = env.observe()
        data.append({**{int(v): x for v, x in o["properties"].items()},
                     k - 1: o["frequency"]})
    menu = build_menu(k, comp, rng)
    half = env.int_budget // 2
    for stage in range(2):
        opts = list(range(len(menu)))
        engine_picks = opts[stage * half:(stage + 1) * half]
        # Blinded LLM recommendation, computed in EVERY cell before enactment.
        summary = "\n".join(
            f"obs/int record {i}: props={{{', '.join(f'{v}:{row[v]:+.2f}' for v in range(k-1))}}} freq={row[k-1]:+.2f}"
            for i, row in enumerate(data[-6:]))
        menu_txt = "\n".join(f"  [{i}] set property {p} to {v:+.1f}"
                             for i, (p, v) in enumerate(menu))
        llm_picks, status = llm_indices(complete, (
            "You control experiment selection in a causal-discovery lab. "
            f"Hidden SCM over properties 0-{k-2} and frequency Y. Choose "
            f"{half} interventions (menu indices) to identify Y's "
            "mechanism.\nRecent data:\n" + summary + "\nMenu:\n" + menu_txt
            + f'\nReply JSON ONLY: {{"menu_indices": [{half} ints]}}'),
            len(menu), half)
        if acq == "engine":
            picks = engine_picks
        elif llm_picks is None:
            tel["acq_fail"] += 1
            picks = engine_picks
        else:
            picks = llm_picks
        decision_log.append({"stage": stage, "kind": "acq",
                             "engine": engine_picks, "llm": llm_picks,
                             "llm_status": status, "enacted": picks})
        for i in picks:
            p, v = menu[i]
            r = env.intervene(p, v)
            if r is None:
                break
            data.append({**{int(a): b for a, b in r["properties"].items()},
                         k - 1: r["frequency"]})
    # Shared model -> COMMON submission menu: 5 candidates + weighted average.
    hyps = fit_target(data, k - 1, k)
    props = env.reactor_properties()
    preds, ws = predict_y(hyps, props, k)
    ranked = sorted(zip(preds, ws), key=lambda t: -t[1])[:5]
    avg = float(np.average([p for p, _ in ranked],
                           weights=[w for _, w in ranked]))
    sub_values = [float(p) for p, _ in ranked] + [avg]
    sub_display = ([f"[{i}] predict {p:+.2f} (model weight {w:.2f})"
                    for i, (p, w) in enumerate(ranked)]
                   + [f"[5] predict {avg:+.2f} (weighted average of the "
                      f"candidates)"])
    engine_idx = 5                      # frozen engine policy: the average
    # Blinded LLM recommendation in EVERY cell.
    llm_idx, status = llm_indices(complete, (
        "You control the FINAL prediction of the reactor crystal's "
        f"frequency. Reactor properties: "
        + ", ".join(f"{v}:{props[v]:+.2f}" for v in range(k - 1))
        + "\nModel-scored candidate predictions:\n"
        + "\n".join("  " + s for s in sub_display)
        + '\nReply JSON ONLY: {"menu_indices": [<one int>]}'),
        len(sub_values), 1)
    llm_choice = llm_idx[0] if llm_idx is not None else None
    if sub == "engine":
        enacted = engine_idx
    elif llm_choice is None:
        tel["sub_fail"] += 1
        enacted = 0                     # ITT default: top-weight candidate
    else:
        enacted = llm_choice
    decision_log.append({"kind": "sub", "menu": sub_display,
                         "engine": engine_idx, "llm": llm_choice,
                         "llm_status": status, "enacted": enacted})
    pred = sub_values[enacted]
    top_h = hyps[0]
    edges = [(int(u), k - 1) for u in top_h["subset"]]
    coefs = {f"{u}->{k-1}": float(top_h["beta"][j + 1])
             for j, u in enumerate(top_h["subset"])}
    res = env.submit(pred, graph_edges=edges, coef_estimates=coefs)
    return {**res, **tel, "acq": acq, "sub": sub, "comp": comp,
            "model": model, "seed": seed, "decision_log": decision_log}


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--blocks", type=int, default=8)
    ap.add_argument("--seed0", type=int, default=1000)
    ap.add_argument("--model", default="gpt-5.4-mini")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    complete = make_llm(args.model)
    cells = list(itertools.product(["llm", "engine"], ["llm", "engine"],
                                   ["screen", "neutral"]))
    out = Path(args.out)
    rows = json.loads(out.read_text()) if out.exists() else []
    done = {(r["seed"], r["acq"], r["sub"], r["comp"]) for r in rows}
    rng = np.random.default_rng(999)
    for b in range(args.seed0, args.seed0 + args.blocks):
        order = cells[:]
        rng.shuffle(order)
        for acq, sub, comp in order:
            if (b, acq, sub, comp) in done:
                continue
            rows.append(run_episode(b, acq, sub, comp, args.model, complete))
        out.write_text(json.dumps(rows, indent=1))
        print(f"block {b} done", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
