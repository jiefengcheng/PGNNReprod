# Independent TxGNN Reproduction

**Status:** Step 1 research implementation staged; **NOT** a validated full-KG reproduction. No biomedical datasets are stored here.

This repository curates code and reproducible source-only artifacts for an independently developed, fully trainable TxGNN pipeline checked against a pinned public reference. The first lane is **construct the heterogeneous training graph and freeze its node inputs**.

## GitHub repository

**Canonical repository:** https://github.com/jiefengcheng/PGNNReprod

This source-only workspace is maintained in `jiefengcheng/PGNNReprod`.
The repository was user-created and is currently **public**. No downloaded KG files,
prepared graph arrays, model weights, private medical data, or generated datasets
are included in Git. Keep research downloads under the ignored local directories.

`bash scripts/publish_github.sh` is an optional **local sync helper** for an
already-cloned repository (it does not create, rename, or change visibility).

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

## Data acquisition and real-reference parity (opt-in)

All downloaded files and generated snapshots remain in ignored local directories. Only run these commands on a machine with the relevant dependencies installed.

```bash
# Download the three exact archived KG files locally (NOT into Git).
python scripts/stage1.py fetch --raw-dir data/raw

# Keep upstream checkout in ignored reference/TxGNN, pinned at the exact commit:
git clone https://github.com/mims-harvard/TxGNN.git reference/TxGNN
git -C reference/TxGNN checkout 0e0203c3939f7d12da27aaf10e55daaff5424023

# In a compatible isolated legacy TxGNN environment (not this minimal environment):
python scripts/stage1.py reference --raw-dir data/raw --upstream-repo reference/TxGNN

# Use the actual snapshot directory reported above:
python scripts/stage1.py compare --snapshot artifacts/local/reference_XXXX/snapshot
```

The reference invocation requires the pinned TxGNN installation imported from `reference/TxGNN` and compatible DGL/PyTorch/PyG dependencies. Its output is not a finished independently constructed raw-data split. **Independent raw normalization and disease-area split parity remain open work**, and no real-data accuracy or training results are claimed.

## Repository rules

Never commit downloaded KG files, disease grouping mapping tables, generated graph arrays, input tensors, model checkpoints, results containing dataset rows, or private medical information. `.gitignore` excludes common directories/extensions; `scripts/check_no_data.py` checks staged files and fails on prohibited binary/data files.

Record source URLs, reference SHA, settings, checksums, and validation reports in the **local** output manifest, not raw data. Commit only source code, documentation, notebooks, tests, and small aggregate non-sensitive results that have been reviewed.

## Reference and attribution

- TxGNN upstream: https://github.com/mims-harvard/TxGNN
- Reference commit: `0e0203c3939f7d12da27aaf10e55daaff5424023`
- TxGNN paper: https://doi.org/10.1101/2023.03.19.23287458
- Released KG: https://doi.org/10.7910/DVN/IXA7BM

This is an independent research implementation, **not** an official TxGNN release. See [`docs/ROADMAP.md`](docs/ROADMAP.md) and [`docs/ACCEPTANCE.md`](docs/ACCEPTANCE.md).
