"""Download pinned benchmark dependencies or generate adaptation records."""
import argparse
import hashlib
import io
import json
from pathlib import Path, PurePosixPath
import shutil
import subprocess
import sys
import tempfile
import urllib.request
import zipfile

ROOT = Path(__file__).resolve().parent
CAUSALGAME_COMMIT = "34223510a0e7466ad399ff19714425f8708b7fbe"
CAUSALGAME_URL = f"https://codeload.github.com/CausalGame/CausalGame/zip/{CAUSALGAME_COMMIT}"
CAUSALGAME_SHA256 = "ebcb876e7d049c6abb54b958db82c22cc52408ccd8015688ebc5a2058c2fef40"
CARE_COMMIT = "9b1c741dbd47161986fe0c050ba45d9e83e0f386"
CARE_URL = f"https://codeload.github.com/SHITIANYU-hue/care/zip/{CARE_COMMIT}"
DATA_HASHES = {
    "datasets/minerva/suzuki_i.csv": "a05f5678760b78a7d4b648f062af93de72c083ae44dac2628f78747bfb29b4c2",
    "datasets/chemlex/acid_amine_wetlab.csv": "1aa659885f7d9acbb450ad322912b8f287f77fe4e146d42c1cbe9e342bcfcaf1",
}


def use_causalgame():
    """Make the separately downloaded, pinned client/backend importable."""
    target = ROOT / "external/causalgame"
    marker = target / ".mpi-prepared.json"
    if not marker.is_file() or json.loads(marker.read_text()).get("upstream_commit") != CAUSALGAME_COMMIT:
        raise RuntimeError("Run python prepare_data.py causalgame first")
    if str(target) not in sys.path:
        sys.path.insert(0, str(target))
    return target


def prepare_causalgame(archive=None):
    from run import MECH, EXTRA
    target = ROOT / "external/causalgame"
    if target.exists():
        raise FileExistsError("external/causalgame already exists; use a fresh checkout to prepare again")
    if archive:
        payload = Path(archive).read_bytes()
    else:
        with urllib.request.urlopen(CAUSALGAME_URL, timeout=120) as response:
            payload = response.read()
    if hashlib.sha256(payload).hexdigest() != CAUSALGAME_SHA256:
        raise ValueError("CausalGame archive checksum mismatch")
    with tempfile.TemporaryDirectory(prefix="mpi-causalgame-") as temporary:
        stage = Path(temporary) / "causalgame"
        stage.mkdir()
        with zipfile.ZipFile(io.BytesIO(payload)) as z:
            for entry in z.infolist():
                parts = PurePosixPath(entry.filename).parts
                if entry.is_dir() or len(parts) < 2:
                    continue
                rel = PurePosixPath(*parts[1:])
                if ".." in rel.parts or rel.is_absolute():
                    raise ValueError("Unsafe archive path")
                keep = (str(rel) in {"LICENSE", "README.md", "agent/client.py"}
                        or (rel.parts[0] == "api" and rel.suffix == ".py")
                        or (len(rel.parts) == 3 and rel.parts[0] == "experiments"
                            and rel.parts[1] in MECH + EXTRA and rel.suffix == ".json"))
                if keep:
                    dst = stage / str(rel)
                    dst.parent.mkdir(parents=True, exist_ok=True)
                    dst.write_bytes(z.read(entry))
        (stage / ".mpi-prepared.json").write_text(json.dumps({
            "upstream_commit": CAUSALGAME_COMMIT,
            "archive_sha256": CAUSALGAME_SHA256}, indent=2))
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copytree(stage, target)
    print("Prepared external/causalgame (fixed upstream version, checked archive)")


def prepare_care(archive=None):
    target = ROOT / "external/care"
    if target.exists():
        raise FileExistsError("external/care already exists; use a fresh checkout to prepare again")
    print("CARE data: Minerva CC BY 4.0; ChemLex CC BY-NC 4.0. See upstream artifact terms.")
    if archive:
        payload = Path(archive).read_bytes()
    else:
        with urllib.request.urlopen(CARE_URL, timeout=120) as response:
            payload = response.read()
    allowed = {"configs", "datasets", "evaluation", "replay_core",
               "research_tool_agent_full_pool", "runners"}
    root_files = {"pyproject.toml", "README.md", "LICENSE_AND_ARTIFACTS.md"}
    with tempfile.TemporaryDirectory(prefix="mpi-care-") as temporary:
        stage = Path(temporary) / "care"
        stage.mkdir()
        with zipfile.ZipFile(io.BytesIO(payload)) as z:
            for entry in z.infolist():
                parts = PurePosixPath(entry.filename).parts
                if entry.is_dir() or len(parts) < 2:
                    continue
                rel = PurePosixPath(*parts[1:])
                if ".." in rel.parts or rel.is_absolute():
                    raise ValueError("Unsafe archive path")
                if rel.parts[0] not in allowed and str(rel) not in root_files:
                    continue
                if rel.suffix not in {".py", ".json", ".csv", ".md", ".toml", ".sh"}:
                    continue
                dst = stage / str(rel)
                dst.parent.mkdir(parents=True, exist_ok=True)
                dst.write_bytes(z.read(entry))
        for rel, digest in DATA_HASHES.items():
            actual = hashlib.sha256((stage / rel).read_bytes()).hexdigest()
            if actual != digest:
                raise ValueError(f"Dataset checksum mismatch: {rel}")
        patch = ROOT / "patches/care_mpi.patch"
        # Apply only the bundled, reviewed patch in this newly created staging tree.
        subprocess.run(["git", "apply", "--check", str(patch)], cwd=stage, check=True)
        subprocess.run(["git", "apply", str(patch)], cwd=stage, check=True)
        for config in (ROOT / "configs/care").glob("*.json"):
            shutil.copyfile(config, stage / "configs" / config.name)
        (stage / ".mpi-prepared.json").write_text(json.dumps({
            "upstream_commit": CARE_COMMIT, "data_sha256": DATA_HASHES,
            "patch_sha256": hashlib.sha256(patch.read_bytes()).hexdigest()}, indent=2))
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copytree(stage, target)
    print("Prepared external/care (fixed upstream version, checked data, applied MPI patch)")


def generate_causalab(seed_text, output):
    from run import seeds, save
    from causalab.env import CausaLabEnv
    rows = []
    for seed in seeds(seed_text):
        env = CausaLabEnv(k=10, seed=seed)
        rows.append({"seed": seed, "observations": [env.observe() for _ in range(env.obs_budget)],
                     "reactor_properties": env.reactor_properties(),
                     "observation_budget": env.obs_budget, "intervention_budget": env.int_budget})
    save(Path(output), rows)
    print(f"Generated {len(rows)} public initial records (no hidden targets) at {output}")


def generate_causalgame(seed_text, output):
    from run import seeds, save, MECH, EXTRA
    source = use_causalgame()
    # Seeds describe runner configurations; the backend generates public history and
    # simulation outcomes at episode creation/execution, not from a stored dataset.
    rows = [{"experiment": scene, "seed": seed,
             "game": json.loads((source / "experiments" / scene / "game.json").read_text()),
             "action_space": json.loads((source / "experiments" / scene / "action_space.json").read_text())}
            for seed in seeds(seed_text) for scene in MECH + EXTRA]
    save(Path(output), rows)
    print(f"Generated {len(rows)} task configurations at {output}; histories are generated by the backend")


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("study", choices=["care", "causalgame", "causalab"])
    ap.add_argument("--seeds", default="1000-1031")
    ap.add_argument("--output")
    ap.add_argument("--archive", help="Optional offline ZIP of the pinned CausalGame or CARE commit")
    a = ap.parse_args()
    if a.study == "care":
        prepare_care(a.archive)
    elif a.study == "causalab":
        generate_causalab(a.seeds, a.output or "data/causalab_initial_records.json")
    else:
        prepare_causalgame(a.archive)
        if a.output:
            generate_causalgame(a.seeds, a.output)


if __name__ == "__main__":
    main()
