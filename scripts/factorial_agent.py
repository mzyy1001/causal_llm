"""
MPI: 2x2 acquisition/submission policy comparison.

Policy comparison conditions:
  - Both controllers act on the IDENTICAL finite action set: a canonical
    scored menu built once per decision. No free-form designs in the primary
    factorial (proposal rights are a separate study).
  - At every decision BOTH controllers' recommendations are computed and
    logged (shadow recommendations); the preassigned configuration determines only
    which one is ENACTED.
  - ENGINE policy is the frozen full pipeline policy: unexplored balanced
    screening first, then EVSI explore vs leader (expected_decision_gains),
    matching the claimed causal-dominated architecture.
  - LLM prompts are stateless (full state each call), constrained to
    {"menu_index": int}; one retry on parse failure; then the prespecified
    consequence: acquisition -> enact deterministic default (menu[0]) and
    record acq_controller_failure; submission -> no submission (ITT: victory
    False, survival coded 0.0), record sub_controller_failure.
  - Infrastructure exceptions are recorded separately (infra_error) and the
    run is marked for blind rerun, never as controller failure.
  - Guardian OFF. Token/call counts logged per cell (shadow calls separated).
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from causal_agent.adaptive import (
    build_candidate_pool, screening_batch, stage2_win_matrix,
    expected_decision_gains, _shortlist_decisions,
)
from causal_agent.design import DesignSpace
from causal_agent.hypotheses import HypothesisEnsemble
from causal_agent.propose import (
    HeuristicProposer, ProposalContext, openai_chat_complete,
)
from causal_agent.session import CausalSession

MIN_GAIN = 0.004  # frozen from the pipeline policy


def parse_rate(raw):
    """The submit endpoint reports survival_rate as '88.0%'; older builds as 0.88."""
    if raw is None:
        return None
    if isinstance(raw, str):
        raw = raw.strip()
        if raw.endswith("%"):
            return float(raw[:-1]) / 100.0
        raw = float(raw)
    raw = float(raw)
    return raw / 100.0 if raw > 1.0 else raw


def make_llm(model: str):
    from mpi_transport import completion
    return completion(model, temperature=0.4)


class Telemetry:
    def __init__(self):
        self.llm_calls_active = 0
        self.llm_calls_shadow = 0
        self.acq_controller_failures = 0
        self.sub_controller_failure = False
        self.infra_errors = []
        self.decisions = []  # per-decision shadow log


def llm_pick_index(complete, prompt, n_options, telemetry, shadow=False):
    """Constrained menu-index choice; None = controller failure (after retry)."""
    for attempt in range(2):
        try:
            reply = complete(prompt)
            if shadow:
                telemetry.llm_calls_shadow += 1
            else:
                telemetry.llm_calls_active += 1
        except Exception as exc:
            telemetry.infra_errors.append(f"llm-transport: {exc}")
            return None, "infra"
        try:
            start, end = reply.find("{"), reply.rfind("}")
            js = json.loads(reply[start:end + 1])
            idx = int(js["menu_index"])
            if 0 <= idx < n_options:
                return idx, "ok"
        except Exception:
            continue
    return None, "controller"


def render_menu(menu, scores):
    lines = []
    for i, ((d, e), sc) in enumerate(zip(menu, scores)):
        s = " ".join(f"{k}={v}" for k, v in sc.items())
        lines.append(f"  [{i}] design={d} equipment={e} {s}")
    return "\n".join(lines)


def state_text(session, ensemble):
    arms = "\n".join(f"  {a.label}: {dict(a.design)} eq={a.equipment} -> "
                     f"{a.survived}/{a.deployed}" for a in session.arms)
    post = (", ".join(f"{f.hypothesis.name}={f.weight:.2f}"
                      for f in ensemble.ranking(6))
            if ensemble.fitted else "(no fit yet)")
    return (f"Budget: {session.drones_remaining} drones, "
            f"{session.deployments_remaining} deployments. "
            f"Victory threshold {session.victory_threshold:.0%}.\n"
            f"Deployments so far:\n{arms or '  none'}\n"
            f"Hypothesis posterior (Bayesian evidence over shared library): {post}")


def run_cell(client, acq, sub, model, seed, verbose=True):
    rng = np.random.default_rng(seed)
    t0 = time.time()
    tel = Telemetry()
    space = DesignSpace(client.get_action_space())
    session = CausalSession(client)
    hyps = HeuristicProposer().propose(space, ProposalContext())
    ensemble = HypothesisEnsemble(hyps, space, rng=rng)
    eq_refs = [(t.keys[0], t.option) for h in ensemble.hypotheses
               for t in h.terms if t.kind == "equipment"]
    pool = build_candidate_pool(space, rng, equipment_options=eq_refs,
                                max_equipment_variants=6)
    screen = screening_batch(space)
    complete = make_llm(model)  # created for ALL cells (shadow logging)

    def log(m):
        if verbose:
            print(m, flush=True)

    # ---------------- acquisition rounds ------------------------------
    while session.deployments_remaining > 0 and session.drones_remaining > 0:
        if session.arms:
            try:
                ensemble.fit(session.arms)
            except Exception as exc:
                tel.infra_errors.append(f"fit: {exc}")
                break
        n_next = min(50, max(1, session.drones_remaining
                             // max(session.deployments_remaining, 1)))

        # canonical menu: unexplored screening rows + engine shortlist +
        # engine leader; scored identically for both controllers
        done = {tuple(sorted(a.design.items())) for a in session.arms}
        menu = [(d, e) for d, e in screen
                if tuple(sorted(d.items())) not in done][:4]
        scores = [{"type": "screening-row"} for _ in menu]
        engine_idx = 0 if menu else None
        if ensemble.fitted:
            dec_idx = _shortlist_decisions(ensemble, pool)[:6]
            decisions = [pool[i] for i in dec_idx]
            try:
                W = stage2_win_matrix(ensemble, decisions,
                                      session.victory_threshold, draws=1000)
                win = ensemble.weights @ W
                gains = expected_decision_gains(ensemble, decisions, W, n_next)
            except Exception as exc:
                tel.infra_errors.append(f"engine-score: {exc}")
                win = np.zeros(len(decisions))
                gains = np.zeros(len(decisions))
            base = len(menu)
            for j, (d, e) in enumerate(decisions):
                menu.append((d, e))
                scores.append({"P(win)": f"{win[j]:.2f}",
                               "info_gain": f"{gains[j]:.3f}"})
            # frozen engine policy: screening rows first; else EVSI explore
            # if worthwhile, else leader
            if engine_idx is None:
                if float(np.max(gains)) > MIN_GAIN:
                    engine_idx = base + int(np.argmax(gains))
                else:
                    engine_idx = base + int(np.argmax(win))
        if engine_idx is None:
            # nothing fitted and screening spent: deterministic default
            menu = [pool[int(i)] for i in rng.choice(len(pool), 4, replace=False)]
            scores = [{"type": "pool"} for _ in menu]
            engine_idx = 0

        # both recommendations, always (shadow logging)
        prompt = ("You control INTERVENTION SELECTION in a budgeted causal-"
                  "discovery mission. Survivor records are censored; trust "
                  "counts. Choose which candidate to deploy next "
                  f"({n_next} drones).\n" + state_text(session, ensemble)
                  + "\nMenu:\n" + render_menu(menu, scores)
                  + '\nReply JSON ONLY: {"menu_index": <int>}')
        llm_idx, llm_status = llm_pick_index(
            complete, prompt, len(menu), tel, shadow=(acq != "llm"))

        if acq == "llm":
            if llm_idx is None:
                if llm_status == "controller":
                    tel.acq_controller_failures += 1
                    enacted = 0  # prespecified consequence: default menu[0]
                else:
                    enacted = engine_idx  # infra: blind fallback, flagged
            else:
                enacted = llm_idx
        else:
            enacted = engine_idx
        tel.decisions.append({"round": len(session.arms), "kind": "acq",
                              "engine": engine_idx, "llm": llm_idx,
                              "llm_status": llm_status, "enacted": enacted})
        design, equipment = menu[enacted]
        try:
            arm = session.deploy(design, n_next, equipment,
                                 label=f"r{len(session.arms)}")
        except Exception as exc:
            tel.infra_errors.append(f"deploy: {exc}")
            break
        log(f"  {arm}  (engine->{engine_idx} llm->{llm_idx} enacted {enacted})")

    # ---------------- submission --------------------------------------
    outcome = {"rate": None, "victory": False}
    if session.arms:
        try:
            ensemble.fit(session.arms)
            dec_idx = _shortlist_decisions(ensemble, pool, top_mixture=40)
            decisions = [pool[i] for i in dec_idx]
            W = stage2_win_matrix(ensemble, decisions,
                                  session.victory_threshold, draws=3000)
            win = ensemble.weights @ W
        except Exception as exc:
            tel.infra_errors.append(f"final-score: {exc}")
            decisions, win = [], np.array([])
        if len(decisions):
            top = list(np.argsort(-win)[:6])
            menu = [decisions[int(j)] for j in top]
            scores = [{"P(win)": f"{win[int(j)]:.2f}"} for j in top]
            engine_idx = 0  # argmax is menu[0] by construction
            prompt = ("You control the FINAL SUBMISSION (irreversible; "
                      "1000-drone fleet vs threshold).\n"
                      + state_text(session, ensemble)
                      + "\nEngine-scored options:\n" + render_menu(menu, scores)
                      + '\nReply JSON ONLY: {"menu_index": <int>}')
            llm_idx, llm_status = llm_pick_index(
                complete, prompt, len(menu), tel, shadow=(sub != "llm"))
            tel.decisions.append({"kind": "sub", "engine": engine_idx,
                                  "llm": llm_idx, "llm_status": llm_status})
            if sub == "llm":
                if llm_idx is None and llm_status == "controller":
                    tel.sub_controller_failure = True
                    enacted = None  # ITT: no submission
                elif llm_idx is None:
                    enacted = engine_idx  # infra fallback, flagged
                else:
                    enacted = llm_idx
            else:
                enacted = engine_idx
            if enacted is not None:
                design, equipment = menu[enacted]
                try:
                    res = session.submit(design, equipment)
                    outcome = {"rate": parse_rate(res.get("survival_rate")),
                               "victory": bool(res.get("victory"))}
                    log(f"  SUBMIT ({acq},{sub}) -> {res.get('survival_rate')} "
                        f"victory={res.get('victory')}")
                except Exception as exc:
                    tel.infra_errors.append(f"submit: {exc}")

    return {
        "acq": acq, "sub": sub, "model": model, "seed": seed,
        "rate": outcome["rate"],
        "rate_itt": outcome["rate"] if outcome["rate"] is not None else 0.0,
        "victory": outcome["victory"],
        "acq_controller_failures": tel.acq_controller_failures,
        "sub_controller_failure": tel.sub_controller_failure,
        "infra_errors": tel.infra_errors,
        "needs_blind_rerun": bool(tel.infra_errors),
        "llm_calls_active": tel.llm_calls_active,
        "llm_calls_shadow": tel.llm_calls_shadow,
        "decision_log": tel.decisions,
        "duration_s": round(time.time() - t0, 1),
    }


def main():
    from prepare_data import use_causalgame
    use_causalgame()
    from agent.client import CanyonClient
    ap = argparse.ArgumentParser()
    ap.add_argument("--experiment", required=True)
    ap.add_argument("--acq", choices=["llm", "engine"], required=True)
    ap.add_argument("--sub", choices=["llm", "engine"], required=True)
    ap.add_argument("--model", default="gpt-5.4-mini")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--base-url", default="http://localhost:8000")
    ap.add_argument("--out", default=None)
    ap.add_argument("--quiet", action="store_true")
    args = ap.parse_args()
    client = CanyonClient(base_url=args.base_url,
                          model_name=f"fact2-{args.acq}-{args.sub}",
                          agent_name=f"seed{args.seed}",
                          experiment=args.experiment, execution_mode="legacy")
    row = run_cell(client, args.acq, args.sub, args.model, args.seed,
                   verbose=not args.quiet)
    row["experiment"] = args.experiment
    print(json.dumps({k: v for k, v in row.items() if k != "decision_log"}))
    if args.out:
        p = Path(args.out)
        rows = json.loads(p.read_text()) if p.exists() else []
        rows.append(row)
        p.write_text(json.dumps(rows, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
