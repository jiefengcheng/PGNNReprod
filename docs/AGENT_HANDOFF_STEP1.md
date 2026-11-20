# Agent handoff — Step 1 independently, WITHOUT TxGNN/DGL installation

> **OVERRIDING EXECUTION INSTRUCTION — 2026-11-20.**
> The earlier handoff incorrectly required installing upstream TxGNN, DGL and PyG and running a legacy numerical oracle. **That is cancelled. Do not attempt those installs or execute `scripts/stage1.py reference` or `scripts/stage1.py compare`.** This is an independent implementation task. Use published source as a **read-only specification**, not as an executable dependency.
>
> If you are currently working on a DGL/TxGNN setup, **stop that effort immediately**. Do not spend additional time debugging the legacy environment.

**Repository:** https://github.com/jiefengcheng/PGNNReprod  
**Scope:** Step 1 — independently construct a heterogeneous *training* graph and reproducibly create/store frozen node inputs. No GNN training, GraphMask or model inference.

## Intent and priorities

Build code that accepts the **released** TxGNN/PrimeKG snapshot and produces a well-defined local graph bundle using **Python + NumPy + Pandas**. Retain upstream grouping decisions from the released files. The optional historical implementation, pinned at `0e0203c3939f7d12da27aaf10e55daaff5424023`, is a **static code reference** for conventions, *not a runtime target*.

**Explicit non-goals:** installing/importing `txgnn`, `dgl`, `torch_geometric`; creating a DGL graph; reproducing any original checkpoint or exact random values; running a model/training script; exact numeric parity with an upstream live process. Torch may be used **later**, but must not be necessary to complete the first NumPy graph-bundle milestone.

## Read before editing

1. This handoff, then [Step 1 technical runbook](STEP1_GRAPH_INPUTS_RUNBOOK.md) (note the updated independence notice at its top).
2. [Acceptance contract](ACCEPTANCE.md), [roadmap](ROADMAP.md), and [source ledger](SOURCES.md).
3. Existing code: `src/txgnn_stage1/graph.py`, `assets.py`, `inputs.py`, `scripts/stage1.py`, `tests/`.
4. Existing [notebook](../notebooks/TxGNN_Step1_Graph_and_Inputs.ipynb), as research background only.

Prior code under `reference.py`, `compare.py` and the CLI's legacy `reference`/`compare` commands is **optional archival tooling** and is **NOT** the next work item.

## First deliverable: independent source-to-bundle pipeline

**Work on a feature branch and open a PR. Do not push directly to `main`.**

### A. Verify source-only starting point

```bash
git clone https://github.com/jiefengcheng/PGNNReprod.git
cd PGNNReprod
git switch -c work/step1-no-dgl
python -m venv .venv
source .venv/bin/activate
python -m pip install -e '.[test]'
python -m pytest -q
python scripts/check_no_data.py
```

Note exact outputs, failures, and skips. Existing Torch-dependent tests may skip; never call a skip PASS. No command in this phase should import DGL.

### B. Acquire raw released files in ignored local directories

```bash
python scripts/stage1.py fetch --raw-dir data/raw
```

Pinned Dataverse IDs: `kg.csv` **7144484** (CSV); `node.csv` **7144482** (**tab-separated**); `edges.csv` **7144483** (CSV). Use `src/txgnn_stage1/assets.py`; preserve SHA-256, byte counts and schema validation in ignored local manifests. Do not commit downloaded data or paste raw rows into GitHub.

If external download is blocked, write deterministic synthetic fixtures and report acquisition **BLOCKED** with exact error. Do not claim real-data acceptance.

### C. Independently implement raw-to-graph preparation

Implement a focused module (suggested `src/txgnn_stage1/preprocess.py`) that:

1. Reads the **released, already harmonized** KG, retaining input row order and source identities. Do not rebuild historical MONDO groups using a modern ontology.
2. Maintains **external ID / source-global index / model-local per-type index** as separate identity spaces.
3. Follows the pinned source's observable conventions for representative orientation, `convert2str` normalization, lexicographically sorted type-local indices, and homogeneous relationship de-duplication. Preserve raw IDs and audit collisions.
4. Assigns base facts to `complex_disease` train/validation/test with an **explicit owned seed**, before reverse-row augmentation. Keep therapeutic disease holdouts protected.
5. Creates reverse relations **inside the same partition**; homogeneous relations reverse endpoints but retain relation name. Capture self-loop/parallel-edge multiplicity; do not introduce test edges into training.
6. Produces ordered canonical triples and **NumPy int64 [2, E]** adjacency arrays from **training edges only**, while retaining all nodes from the complete processed graph, including held-out/isolated nodes.
7. Records exactly which assumptions came from static public source and which are our own protocol decisions; do not claim exact paper parity without evidence.

Use the existing `graph.build_graph(...)` component for the already-prepared-table-to-adjacency step. Avoid reimplementing that tested part without reason.

### D. Independently create frozen inputs and serialize

For each node type `t` with `N_t` nodes, create one complete FP32 array `[N_t, 100]` using an explicit **owned deterministic RNG** and Xavier-uniform bound `sqrt(6/(N_t + 100))` (default gain). Save actual tensors, dtype, shape, seed, RNG algorithm and content hashes in ignored local output; future Torch code must **load** those values, not resample.

A NumPy generator can reproduce the **distribution** but will not reproduce historical Torch random bit patterns. State this clearly. Keep the existing optional Torch helper as an alternative, not a blocker.

Implement a data-format-independent reader (suggested `bundle.py`) that restores schema, IDs, split metadata, edge arrays and inputs **without any random-number generation**. Verify in a **fresh process** using only NumPy/Pandas.

### E. Add tests that can catch wrong behavior

At minimum: stable ID sorting, identity collisions/aliases, grouped disease mapping, representative orientation, train/test separation, reverse-edge leakage, homogeneous duplicate/self-loop behavior, isolated-node retention, canonical relation order, deterministic input matrices, and exact fresh-process bundle reload. Include tiny hand-computed fixtures and property assertions.

Run tests without DGL/PyG/TxGNN. If full data are available, write a non-sensitive **aggregate** audit (per type/relation counts, missing mappings, excluded edges, total/held-out counts, hashes) to local output. Distinguish original-paper totals from computed released-file counts.

## Acceptance (NO upstream runtime required)

A PR may claim Step 1 accepted **only when** all of the following have actual evidence:

- Input assets hash-locked, or clearly **BLOCKED** (then Step 1 is not accepted).
- End-to-end independent raw-to-graph pipeline produces a typed, directed, split-aware graph using **no DGL**.
- Identity crosswalks and any grouped disease membership are demonstrably consistent with the released snapshot.
- No forbidden held-out therapeutic edge, in either orientation, survives in train adjacency.
- Serialized graph/input bundle loads identically in a new process without re-running preparation or RNG.
- Deterministic repeat runs reproduce schema, ordered edges, masks, inputs and aggregate hashes.
- All required tests have explicit pass/fail/skipped counts and any deviations from static source behavior are documented.
- No source data, generated graph arrays, checkpoint, or raw output was added to the public repository.

**Do not claim** equivalence to the authors' historical DGL object, trained metrics, or original seed tensors. Those are separate, optional investigations.

## PR and status report format

Commit **code, small test fixtures, docs and sanitized non-sensitive statistics only**. Open a PR with a `docs/STEP1_EXECUTION_REPORT.md` stating:

```text
Base and head SHA; branch; PR:
Environment and exact executed commands:
Assets (IDs, sizes, SHA-256; no contents):
Independent preparation outputs (local paths, summary counts/hashes):
Tests: run/passed/failed/skipped:
Identity, reverse-edge and holdout-leakage checks:
Input generator, shape/dtype/seed, save and independent reload:
Source-derived decisions vs owned choices:
Known deviations and blockers:
Next single action:
```

**Priority:** Stop legacy setup. Implement the independent Python data transformation and tests, then frozen inputs/bundle. Remain within this scope.