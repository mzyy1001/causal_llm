"""Run MPI evaluations. Planning is the default; API use requires --execute."""
import argparse
import itertools
import json
import os
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parent
MECH = ["antenna_trap", "antenna_trap_high_def", "antenna_trap_simpsons_paradox"]
EXTRA = ["deployment_zone_trap_categorical", "deployment_zone_trap_env_shift", "weather_noise"]
POLICIES = {
    "incumbent": "public_expert_only_meta_controller",
    "gate": "true_self_evolving_api_care",
    "judge": "true_self_evolving_api_care_llm_judge_gate",
}


def seeds(text):
    values = set()
    for part in text.split(","):
        bounds = part.split("-")
        if len(bounds) == 1:
            values.add(int(part))
        elif len(bounds) == 2:
            lo, hi = map(int, bounds)
            if hi < lo:
                raise ValueError("Seed range must be ascending")
            values.update(range(lo, hi + 1))
        else:
            raise ValueError("Use seeds such as 0,1 or 1000-1031 (inclusive)")
    if not values or min(values) < 0:
        raise ValueError("Nonnegative seeds are required")
    return sorted(values)


def save(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(rows, indent=2, allow_nan=False))
    temporary.replace(path)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("study", choices=["causalgame", "causalab", "care"])
    ap.add_argument("--seeds", default="0")
    ap.add_argument("--model", default="gpt-5.4-mini")
    ap.add_argument("--scene-set", choices=["mech", "all"], default="mech")
    ap.add_argument("--scenes", help="Optional comma-separated subset of the six supported scenarios")
    ap.add_argument("--server-url", default="http://127.0.0.1:8000")
    ap.add_argument("--datasets", default="minerva,chemlex")
    ap.add_argument("--arms", default="incumbent,gate,judge")
    ap.add_argument("--output", default="outputs")
    ap.add_argument("--execute", action="store_true", help="Actually run; may incur API charges")
    a = ap.parse_args()
    ss = seeds(a.seeds)
    scenes = a.scenes.split(",") if a.scenes else MECH + (EXTRA if a.scene_set == "all" else [])
    if not set(scenes) <= set(MECH + EXTRA):
        ap.error("Unknown scenario")
    datasets, arms = a.datasets.split(","), a.arms.split(",")
    if not set(datasets) <= {"minerva", "chemlex"} or not set(arms) <= set(POLICIES):
        ap.error("Unknown CARE dataset or arm")
    n = len(ss) * ({"causalgame": 4 * len(scenes), "causalab": 8,
                   "care": len(datasets) * len(arms)}[a.study])
    print(json.dumps({"study": a.study, "model": a.model, "seeds": ss,
                      "planned_episodes": n, "execute": a.execute}, indent=2))
    if not a.execute:
        return
    from mpi_transport import credentials
    base, key = credentials()
    out = Path(a.output).resolve()
    if a.study == "care":
        care = ROOT / "external/care"
        if not (care / ".mpi-prepared.json").is_file():
            ap.error("Run python prepare_data.py care first")
        env = os.environ.copy()
        env.update(COMMONSTACK_API_KEY=key, COMMONSTACK_API_ENDPOINT=base)
        for dataset in datasets:
            config = json.loads((ROOT / f"configs/care/{dataset}_care_case2.json").read_text())
            config["model"] = a.model
            config_path = out / "care" / dataset / "run_config.json"
            save(config_path, config)
            # Each arm runs in a fresh process: no cross-arm client/controller state.
            for arm in arms:
                target = out / "care" / dataset / arm
                if (target / "run_rows.json").exists():
                    raise FileExistsError(f"Refusing to overwrite existing CARE run: {target}")
                subprocess.run([sys.executable, "runners/run_self_evolving_proof_suite.py",
                                str(config_path), "--seeds", ",".join(map(str, ss)),
                                "--max-rounds", "10", "--policies", POLICIES[arm],
                                "--parallel-workers", "1", "--output-dir", str(target)],
                               cwd=care, env=env, check=True)
        return
    import numpy as np
    path = out / f"{a.study}.json"
    rows = json.loads(path.read_text()) if path.exists() else []
    if any(r.get("model") != a.model for r in rows):
        raise ValueError("Use a separate output directory for each model")
    fields = ["seed", "experiment", "acq", "sub"] if a.study == "causalgame" else ["seed", "acq", "sub", "comp"]
    done = {tuple(r[k] for k in fields) for r in rows}
    cells = list(itertools.product(["llm", "engine"], repeat=2))
    if a.study == "causalgame":
        from prepare_data import use_causalgame
        use_causalgame()
        from agent.client import CanyonClient
        from scripts.factorial_agent import run_cell
        rng = np.random.default_rng(2027)
        for seed in ss:
            for scene in scenes:
                order = cells[:]
                rng.shuffle(order)
                for acq, sub in order:
                    if (seed, scene, acq, sub) in done:
                        continue
                    client = CanyonClient(base_url=a.server_url, model_name=a.model,
                                          agent_name=f"seed{seed}", experiment=scene,
                                          execution_mode="legacy")
                    row = run_cell(client, acq, sub, a.model, seed, verbose=False)
                    row["experiment"] = scene
                    rows.append(row)
                    save(path, rows)
                    print(f"saved {scene} seed={seed} {acq}/{sub}", flush=True)
    else:
        from causalab.m3_agent_v2 import run_episode, make_llm
        complete = make_llm(a.model)
        cells = [(x, y, z) for x, y in cells for z in ["screen", "neutral"]]
        rng = np.random.default_rng(999)
        for seed in ss:
            order = cells[:]
            rng.shuffle(order)
            for acq, sub, comp in order:
                if (seed, acq, sub, comp) in done:
                    continue
                rows.append(run_episode(seed, acq, sub, comp, a.model, complete))
                save(path, rows)
                print(f"saved seed={seed} {acq}/{sub}/{comp}", flush=True)


if __name__ == "__main__":
    main()
