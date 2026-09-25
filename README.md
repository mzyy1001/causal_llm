# Matched Policy Intervention

Code for **From System Performance to Decision Attribution in Agent Evaluation**.

MPI compares preassigned decision policies under matched external information,
action menus, budgets, and execution rules. Both recommendations are recorded
before the selected policy's recommendation is executed.

## Setup

Use Python 3.13 and Git. Run commands from this repository's root. The pinned
dependencies match the environment used for the package's offline checks;
they are not a claim about every dependency version used in historical runs.

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

For actual evaluations, export your
own credentials. Do not commit credentials or generated outputs.

```bash
export OPENAI_API_KEY="YOUR_KEY"
export OPENAI_BASE_URL="https://api.openai.com/v1"
```

An OpenAI-compatible endpoint may be substituted. It must support the requested
model and parameters. CARE uses Responses streaming for policy synthesis and
Chat Completions for its decision gate. The other studies use Chat Completions.
The model identifiers in the commands below are the experiment labels; availability
depends on your provider. Changing a model or its sampling parameters changes
the experimental configuration. Seeds do not guarantee identical model responses.

## CausalGame

Download [CausalGame](https://github.com/CausalGame/CausalGame) at the fixed commit
in `prepare_data.py`. The script checks the archive checksum and extracts the
backend, client, and six task configurations into ignored `external/causalgame/`;
upstream environment files are not bundled in this repository. Histories and
simulation outcomes are generated on demand. MPI's prompts are constructed by
our runners.

Start the backend in one terminal (localhost only):

```bash
python prepare_data.py causalgame
CAUSALGAME_SERVE_STATIC=false python -m uvicorn api.app:app --app-dir external/causalgame --host 127.0.0.1 --port 8000
```

In another terminal, with the environment activated and API credentials exported:

```bash
# Print the plan without making API calls.
python run.py causalgame --model gpt-5.4-mini --seeds 400-431

# Run all four acquisition/submission assignments for each task and seed.
python run.py causalgame --model gpt-5.4-mini --seeds 400-431 --output outputs/cg-mini-powered --execute
python run.py causalgame --model gpt-5.4-mini --seeds 8000-8031 --scene-set all --output outputs/cg-mini-rerun --execute
python analyze.py causalgame outputs/cg-mini-powered/causalgame.json outputs/cg-mini-rerun/causalgame.json --alpha .05 --family-size 10
```

Repeat with `gpt-5.6-sol` and separate output directories. The `gpt-5.5` study
uses seeds `500-531`, the three default mechanism-rich scenarios, and no pooled
independent rerun. The additional scenarios in `--scene-set all` are analyzed
separately using `--scene-group additional`. Their historical powered records
are not bundled. Do not pool mechanism-rich and additional scenarios to reproduce
the manuscript's stratified conclusions.

## CARE

Download [CARE](https://github.com/SHITIANYU-hue/care) at the fixed commit in
`prepare_data.py`, prepare its datasets, and apply the supplied MPI patch.
Upstream sources and datasets stay in ignored `external/care/`, not in this repository:

```bash
python prepare_data.py care
python -m pip install -e external/care
python run.py care --model gpt-5.4-mini --seeds 0-29
python run.py care --model gpt-5.4-mini --seeds 0-29 --output outputs/care-study --execute
python analyze.py care outputs/care-study/care
python analyze.py care outputs/care-study/care --seed-min 10
```

The three arms are the public incumbent, deterministic CARE gate, and LLM gate.
The last analysis also restricts ChemLex to the selected seeds; only Minerva's
10-29 subset is the manuscript's confirmatory contrast. Seeds 0-9 of Minerva were
exploratory; its pooled 0-29 analysis is descriptive. ChemLex uses seeds 0-29.
`prepare_data.py` verifies dataset checksums; preprocessing is specified in
`configs/care/`. Dataset licenses are listed in [third-party notices](THIRD_PARTY_NOTICES.md).

## CausaLab adaptation

We use a discrete-action adaptation of the
[CausaLab task specification](https://arxiv.org/abs/2605.26029), not the official
environment. The small `causalab/env.py` implements the environment used in our
experiments and must be retained: downloading the official implementation would
not reproduce this adaptation. Environments are generated from seeds during
evaluation; no external dataset is needed.

```bash
# Optional: generate public initial observations for inspection (not agent outputs).
python prepare_data.py causalab --seeds 1000-1031
python run.py causalab --model gpt-5.4-mini --seeds 1000-1031
python run.py causalab --model gpt-5.4-mini --seeds 1000-1031 --output outputs/cl-mini --execute
python run.py causalab --model gpt-5.6-sol --seeds 1000-1031 --output outputs/cl-sol --execute
python analyze.py causalab outputs/cl-mini/causalab.json outputs/cl-sol/causalab.json --alpha .025 --family-size 4 --delta 5
```

Every seed covers eight cells: acquisition policy × submission policy × menu
construction. Both policies see the same six-option final prediction menu,
including the weighted average as an explicit option.

## Outputs and analysis

`run.py` defaults to planning; only `--execute` makes model calls. A full run can
be expensive, including cells controlled by the statistical policy because the
LLM's unexecuted recommendations are also collected. Test a small seed range first.
Each CausalGame or CausaLab output directory must contain a single model; completed
cells are skipped on restart. A CARE output directory must be fresh.

`analyze.py` computes statistical-minus-LLM effects in percentage points using
paired block contrasts and Student-t intervals. It rejects duplicate or incomplete
blocks and does not silently drop failed runs. `--family-size 1` produces pointwise
intervals; use the manuscript-specific family and alpha values above for simultaneous
policy bounds. New API runs need not reproduce the historical scores exactly.
Stored historical model responses and episode logs are not included.

## Layout

- `run.py`, `prepare_data.py`, `analyze.py`: entry points.
- `scripts/factorial_agent.py`, `causal_agent/`: CausalGame policy comparison.
- `external/`: downloaded CausalGame and CARE dependencies, excluded from Git.
- `causalab/`: our CausaLab adaptation and matched-menu policies.
- `configs/care/`, `patches/care_mpi.patch`: CARE configuration and intervention.
- `tests/`: offline checks; mock responses are test fixtures, not reported results.

Historical implementation identifiers (`engine`, `authority`, `M2`, `M3`, `MAI`)
are retained where needed for compatibility with output schemas. In the manuscript,
`engine` means the statistical policy and execution assignment is part of MPI.
See [third-party notices](THIRD_PARTY_NOTICES.md) before redistribution.

After preparing CausalGame, run `python -m unittest discover -s tests -v` for the
offline checks (no model API calls). Without the download, the CausalGame
integration check is explicitly skipped. Both download commands accept
`--archive /path/to/pinned-source.zip` for offline setup.
