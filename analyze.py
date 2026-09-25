"""Paired t intervals for newly generated MPI runs; fail on incomplete blocks."""
import argparse
from collections import defaultdict
import itertools
import json
import math
from pathlib import Path
import numpy as np
from scipy.stats import t


def summarize(values, alpha=0.05, family=1, delta=5.0):
    d = np.asarray(values, dtype=float)
    if len(d) < 2 or not np.isfinite(d).all():
        raise ValueError("At least two finite block contrasts are needed")
    mean = float(d.mean())
    se = float(d.std(ddof=1) / math.sqrt(len(d)))
    width = float(t.ppf(1 - alpha / (2 * family), len(d) - 1)) * se
    lo, hi = mean - width, mean + width
    return {"n_blocks": len(d), "estimate": mean, "interval": [lo, hi],
            "alpha": alpha, "family_size": family, "delta": delta,
            "decision": "DELEGATE" if hi < delta else "RETAIN" if lo > delta else "INCONCLUSIVE"}


def factorial(rows, study):
    blocks = defaultdict(dict)
    flags = 0
    for r in rows:
        block = (r["model"], r.get("experiment", "causalab"), r["seed"])
        cell = (r["acq"], r["sub"]) + ((r["comp"],) if study == "causalab" else ())
        if cell in blocks[block]:
            raise ValueError(f"Duplicate cell in {block}: {cell}")
        value = r["accuracy"] if study == "causalab" else r["rate_itt"]
        if value is None or not math.isfinite(float(value)):
            raise ValueError(f"Missing/nonfinite endpoint in {block}")
        blocks[block][cell] = 100 * float(value)
        flags += bool(r.get("infra_errors"))
    cells = set(itertools.product(["llm", "engine"], repeat=2))
    if study == "causalab":
        cells = {(a, s, f) for a, s in cells for f in ["screen", "neutral"]}
    contrasts = defaultdict(list)
    for (model, scene, seed), table in blocks.items():
        if set(table) != cells:
            raise ValueError(f"Incomplete/unexpected cells: {model}/{scene}/{seed}")
        for stage, index in [("acquisition", 0), ("submission", 1)]:
            diff = np.mean([v for c, v in table.items() if c[index] == "engine"]) - np.mean(
                [v for c, v in table.items() if c[index] == "llm"])
            contrasts[(model, stage)].append(float(diff))
    return contrasts, flags


def care_contrasts(root, seed_min=0):
    result = {}
    for dataset in ["minerva", "chemlex"]:
        arms = {}
        for arm in ["incumbent", "gate", "judge"]:
            path = root / dataset / arm / "run_rows.json"
            if not path.exists():
                raise FileNotFoundError(path)
            rows = json.loads(path.read_text())
            by_seed = {}
            for row in rows:
                seed = int(row["seed"])
                if seed < seed_min:
                    continue
                if seed in by_seed:
                    raise ValueError(f"Duplicate seed in {path}: {seed}")
                value = row.get("final_best_yield")
                if value is None or not math.isfinite(float(value)):
                    raise ValueError(f"Missing endpoint: {dataset}/{arm}/{seed}")
                by_seed[seed] = float(value)
            arms[arm] = by_seed
        if not (arms["gate"].keys() == arms["judge"].keys() == arms["incumbent"].keys()):
            raise ValueError(f"Unmatched seeds: {dataset}")
        ss = sorted(arms["gate"])
        result[dataset] = {"arm_means": {a: float(np.mean(list(v.values()))) for a, v in arms.items()},
                           "gate_minus_judge": [arms["gate"][s] - arms["judge"][s] for s in ss],
                           "gate_minus_incumbent": [arms["gate"][s] - arms["incumbent"][s] for s in ss]}
    return result


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("study", choices=["causalgame", "causalab", "care"])
    ap.add_argument("inputs", nargs="+", help="JSON result files, or a CARE output directory")
    ap.add_argument("--alpha", type=float, default=.05)
    ap.add_argument("--family-size", type=int, default=1)
    ap.add_argument("--delta", type=float, default=5)
    ap.add_argument("--seed-min", type=int, default=0, help="CARE seed subset (10 for confirmatory Minerva)")
    ap.add_argument("--scene-group", choices=["mech", "additional", "all"], default="mech")
    a = ap.parse_args()
    if not 0 < a.alpha < 1 or a.family_size < 1:
        ap.error("alpha must be in (0,1); family-size must be positive")
    stats = lambda x: summarize(x, a.alpha, a.family_size, a.delta)
    if a.study == "care":
        if len(a.inputs) != 1:
            ap.error("CARE expects one root directory")
        out = care_contrasts(Path(a.inputs[0]), a.seed_min)
        for item in out.values():
            for contrast in ["gate_minus_judge", "gate_minus_incumbent"]:
                item[contrast] = stats(item[contrast])
    else:
        rows = [row for f in a.inputs for row in json.loads(Path(f).read_text())]
        if a.study == "causalgame" and a.scene_group != "all":
            from run import MECH, EXTRA
            scenes = MECH if a.scene_group == "mech" else EXTRA
            rows = [r for r in rows if r["experiment"] in scenes]
        if not rows:
            ap.error("No rows match the requested analysis")
        contrasts, flags = factorial(rows, a.study)
        out = {"policy_contrasts": {f"{m}/{s}": stats(x) for (m, s), x in contrasts.items()},
               "infra_flagged_episodes_retained": flags}
    print(json.dumps(out, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
