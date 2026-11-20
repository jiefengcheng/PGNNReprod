# Independent TxGNN Reproduction

**Status:** Step 1 source-only components exist; independent raw-to-bundle implementation is incomplete. **No TxGNN/DGL installation is required.** No biomedical datasets are stored here.

This repository curates code and reproducible source-only artifacts for an independently developed, fully trainable TxGNN pipeline grounded in the published paper and pinned source as a static specification. The first lane is **construct the heterogeneous training graph and freeze its node inputs**.

## GitHub repository

**Canonical repository:** https://github.com/jiefengcheng/PGNNReprod

This source-only workspace is maintained in `jiefengcheng/PGNNReprod`.
The repository was user-created and is currently **public**. No downloaded KG files,
prepared graph arrays, model weights, private medical data, or generated datasets
are included in Git. Keep research downloads under the ignored local directories.

`bash scripts/publish_github.sh` is an optional **local sync helper** for an
already-cloned repository (it does not create, rename, or change visibility).

## Agent handoff

For the next implementation agent, start with **[Step 1 execution handoff](docs/AGENT_HANDOFF_STEP1.md)**. It specifies exact Phase A commands and evidence requirements, Phase B scope boundaries, PR/report format, and stop conditions. Do not begin model training until Step 1 is independently verified.

## Step 1 execution runbook

Read **[Step 1 — Heterogeneous graph and frozen node inputs](docs/STEP1_GRAPH_INPUTS_RUNBOOK.md)** for the end-to-end preparation plan, source identities, commands, exact reference comparison gates, unresolved independent-preprocessing work, and no-data repository policy. The notebook remains at [`notebooks/TxGNN_Step1_Graph_and_Inputs.ipynb`](notebooks/TxGNN_Step1_Graph_and_Inputs.ipynb).

## What is included

| Path | Purpose | Readiness |
| --- | --- | --- |
| [`notebooks/TxGNN_Step1_Graph_and_Inputs.ipynb`](notebooks/TxGNN_Step1_Graph_and_Inputs.ipynb) | Original research notebook, preserved as authored | Synthetic checks pass in the notebook; full-data gates not run |
| `src/txgnn_stage1/graph.py` | Independent graph constructor for **already prepared** TxGNN tables | Synthetic tests only |
| `src/txgnn_stage1/inputs.py` | Frozen CPU Xavier input creation | Needs pinned-reference parity |
| `src/txgnn_stage1/assets.py` | Exact Dataverse-file acquisition and hash locking | Ready, real downloads not performed in this repository |
| `src/txgnn_stage1/reference.py` | Snapshot exporter around pinned upstream code | Requires legacy upstream runtime |
| `src/txgnn_stage1/compare.py` | Reference graph/input parity from prepared tables | Pending full dataset and oracle |
| `tests/` | Source-only synthetic graph tests | Executable without datasets |
| `docs/` | Scope, evidence, milestones and acceptance criteria | Planning artifacts |

## Get started (without datasets)

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e '.[test]'
python -m pytest -q tests/test_graph.py
```

## Step 1: independent preparation (NO upstream runtime)

**Do not install or execute TxGNN, DGL or PyG for Step 1.** Start with [the corrected agent handoff](docs/AGENT_HANDOFF_STEP1.md) and [the runbook](docs/STEP1_GRAPH_INPUTS_RUNBOOK.md). Upstream source is a **read-only specification**. The previous legacy-reference execution plan has been withdrawn as a prerequisite.

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e '.[test]'
python -m pytest -q
python scripts/check_no_data.py

# Optional released asset download, into gitignored local directory:
python scripts/stage1.py fetch --raw-dir data/raw
```

**Next engineering work:** implement `raw KG → normalized typed IDs → base splits → reverse rows → NumPy training adjacency → frozen owned inputs → serialized reloadable bundle`. The existing `graph.py` only handles **already prepared** tables. `reference.py`, `compare.py`, and `scripts/stage1.py reference|compare` are **legacy optional tools** and must not be required or attempted by the Step 1 agent.

## Repository rules

Never commit downloaded KG files, disease grouping mapping tables, generated graph arrays, input tensors, model checkpoints, results containing dataset rows, or private medical information. `.gitignore` excludes common directories/extensions; `scripts/check_no_data.py` checks staged files and fails on prohibited binary/data files.

Record source URLs, reference SHA, settings, checksums, and validation reports in the **local** output manifest, not raw data. Commit only source code, documentation, notebooks, tests, and small aggregate non-sensitive results that have been reviewed.

## Reference and attribution

- TxGNN upstream: https://github.com/mims-harvard/TxGNN
- Reference commit: `0e0203c3939f7d12da27aaf10e55daaff5424023`
- TxGNN paper: https://doi.org/10.1101/2023.03.19.23287458
- Released KG: https://doi.org/10.7910/DVN/IXA7BM

This is an independent research implementation, **not** an official TxGNN release. See [`docs/ROADMAP.md`](docs/ROADMAP.md) and [`docs/ACCEPTANCE.md`](docs/ACCEPTANCE.md).
