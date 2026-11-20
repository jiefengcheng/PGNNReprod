# Agent Handoff — TxGNN Step 1 (execute, verify, do not redesign)

**Repository:** https://github.com/jiefengcheng/PGNNReprod
**Pinned upstream reference:** https://github.com/mims-harvard/TxGNN/tree/0e0203c3939f7d12da27aaf10e55daaff5424023
**Status at handoff (2026-11-20):** Source-only synthetic graph tests exist. Real-data asset acquisition, pinned-reference execution and full-data parity are NOT_RUN. Independent raw normalization, disease/group mapping and split reconstruction are NOT_IMPLEMENTED. Do not misrepresent these statuses.

## Your assignment

You are the implementation agent taking over **Step 1 only: construct a heterogeneous training graph and initialize frozen node inputs** for an independently owned TxGNN reproduction. **Your job is to make a verified implementation, not to produce a plausible design.**

Read in this order before changing code:

1. [README](../README.md)
2. [Step 1 detailed runbook](STEP1_GRAPH_INPUTS_RUNBOOK.md)
3. [Acceptance criteria](ACCEPTANCE.md)
4. [Roadmap](ROADMAP.md)
5. [Source ledger](SOURCES.md)
6. [Original notebook](../notebooks/TxGNN_Step1_Graph_and_Inputs.ipynb)
7. `scripts/stage1.py`, `src/txgnn_stage1/{assets,reference,graph,inputs,compare}.py` and `tests/`

**Never describe an unexecuted command, incomplete implementation or missing evidence as PASS.**

## Non-negotiable constraints

- Do not train TxGNN, implement the encoder, change the hypothesis family or start GraphMask in this assignment.
- Keep the pinned upstream commit as the **oracle**, not a dependency of the owned final implementation. Do not mix 2023 paper hyperparameters (512) with the pinned reproduction script's first test configuration (100).
- First integration split: **`complex_disease`, fold seed 42, preprocessing seed 42, Torch CPU input seed 0, input dimension 100**. Autoimmune/DOID 417 is a **separate later bundle**.
- The GitHub repository is **public**. **Never commit downloaded KG/ontology/disease source data, derived data rows, masks, graph adjacency arrays, node-input tensors, weights, checkpoints, private clinical data or archives.** Use ignored `data/raw/`, `reference/TxGNN/`, and `artifacts/local/` paths. Do not put raw data into an issue, PR body or workflow artifact.
- Work on a new branch; open a PR for review. Do not push directly to `main` or use `scripts/publish_github.sh` for PR work (that helper pushes main).
- No silent fixes to the pinned reference. For a discrepancy, isolate and document the difference with before/after evidence; an intentional changed policy must be a separate named variant.
- Record exactly what ran, with commands, exit codes, measured counts, SHA-256 fingerprints, environment versions and the reasons for blockers. Never invent metrics.

## Immediate job: Phase A — the first PR

**Purpose:** Verify the existing **prepared-tables → independent graph/input** path on the *actual* released data. This is NOT yet independent raw-to-split reproduction.

### A0. Establish starting state

```bash
git clone https://github.com/jiefengcheng/PGNNReprod.git
cd PGNNReprod
git status --short
git rev-parse HEAD
git switch -c work/txgnn-step1-reference-parity

python -m venv .venv
source .venv/bin/activate
pip install -e '.[test]'
python -m pytest -q
python scripts/check_no_data.py
```

Save command output and environment versions in local logs (not raw data in Git). If tests fail, investigate before proceeding. `inputs.py` requires Torch to run its optional test, so a minimal environment may skip it: **a skipped test is not a Torch parity PASS**.

### A1. Acquire source assets, without committing them

```bash
python scripts/stage1.py fetch --raw-dir data/raw
```

This fetches exactly:

- Dataverse 7144484 → `kg.csv` (CSV)
- Dataverse 7144482 → `node.csv` (**tab-separated**)
- Dataverse 7144483 → `edges.csv` (CSV)

Check `data/raw/asset_lock.json`, actual file existence and header validation. Report filenames, IDs, sizes, SHA-256 hashes, schema-check status and any download error; **never paste row data**. A successful download is not itself proof of a complete graph.

### A2. Establish the upstream runtime as an isolated, measured prerequisite

```bash
git clone https://github.com/mims-harvard/TxGNN.git reference/TxGNN
git -C reference/TxGNN checkout 0e0203c3939f7d12da27aaf10e55daaff5424023
git -C reference/TxGNN rev-parse HEAD
```

In a separate compatible environment, resolve pinned upstream imports including Torch, DGL and PyG. The upstream README begins with Python 3.8/DGL 0.5.2, but the whole dependency combination must be **actually tested**. Use CPU for graph preparation if possible. Confirm `txgnn.TxData` is imported from the **pinned local checkout** (see the guard in `reference.py`). If the legacy stack is incompatible with the available OS/hardware, record precise error and propose the smallest verified workaround; do not silently change upstream behavior.

### A3. Capture the upstream oracle and perform Comparison A

From the repository root, in that compatible environment:

```bash
python scripts/stage1.py reference \
  --raw-dir data/raw \
  --upstream-repo reference/TxGNN \
  --split complex_disease \
  --fold-seed 42 \
  --preprocess-seed 42 \
  --input-seed 0 \
  --input-dim 100

# Replace the placeholder with the actual printed snapshot directory.
python scripts/stage1.py compare \
  --snapshot artifacts/local/reference_XXXX/snapshot

python -m pytest -q
python scripts/check_no_data.py
git status --short
```

Do not literally run the placeholder `reference_XXXX`. Save the concrete path in your local run log.

Compare **all** node-type counts, ordered canonical relation types, edge endpoint arrays **including multiplicity**, node-input CPU FP32 values, RNG states and new-process replay. If one comparison fails, stop, isolate the smallest failing component and add a focused regression test. Report **FAIL**, not PASS. The existing `compare.py` verifies a graph built from *reference-prepared* tables, not independent preprocessing.

### Required Phase A report / PR

Commit source-only code fixes (if justified), new synthetic tests and a **sanitized** `docs/STEP1_EXECUTION_REPORT.md` with:

| Gate | Mandatory status | Evidence |
| --- | --- | --- |
| Source-only tests | PASS / FAIL / BLOCKED | command, tests run/skipped, failure summary |
| Dataset acquisition | PASS / FAIL / BLOCKED | IDs, sizes, SHA-256, schema checks |
| Pinned runtime/import | PASS / FAIL / BLOCKED | package versions, import path |
| Reference snapshot | PASS / FAIL / BLOCKED | exact split/seeds, output manifest fingerprint |
| Comparison A | PASS / FAIL / BLOCKED / NOT_RUN | numerical/array mismatches and counts |
| Full owned raw-to-split pipeline | **NOT_IMPLEMENTED** | remain explicit until Phase B |
| No-data policy | PASS / FAIL | `python scripts/check_no_data.py`, `git ls-files` |

A blocked environment or dataset is **a valid report**, not permission to fabricate progress. Open a PR; provide link, commit, exact reproducible commands, evidence and **one next blocker**.

## Subsequent job: Phase B — separate PR, only after A is verified

Implement **independent raw assets → representative base edges → typed IDs and disease membership → base train/valid/test → reverse-augmented splits → training graph → frozen input bundle**. Do not call the upstream `preprocess_kg` or `create_split` in the **owned** data preparation path; those functions may be used only for oracle output comparison.

Key fidelity rules:

1. **Preserve three identity layers:** external IDs, released global indices, per-type local indices. Preserve the released manually curated MONDO disease groups; do not recompute grouping from newer ontologies.
2. Reference ID normalization converts numeric-looking IDs to float then string; per-type index assignment sorts normalized strings lexicographically. Audit identity collisions.
3. Preserve representative direction and row-order quirks; detect relation-name/endpoint-type ambiguity.
4. **Split base facts before generating reverse relations.** The forward and inverse of a held-out therapeutic fact must never enter training. Keep both cached base test and final filtered evaluation views.
5. Use the **full processed node inventory**, even for isolated or held-out nodes; only the *edges* are restricted to training.
6. Freeze all preprocessing masks and split membership, not merely random seeds. For autoimmune, DOID `417` and the default two-hop sampling logic are distinct from the `complex_disease` fixture.
7. Node inputs: initialize whole matrices per type with Torch CPU FP32 Xavier-uniform, in the reference node-type order; set `requires_grad=False`; persist **actual matrices** with shapes, digests and RNG states. Never regenerate inputs when reloading.
8. Compare each intermediate stage with the pinned oracle, including ordered arrays and edge multisets; document discrepancies. Test fresh-process load with **no RNG calls**.

**Phase B exit:** full source-to-bundle equivalence under the explicitly declared protocol (or an approved, separately named variant), no holdout leakage, and complete fresh-process restoration. Do not declare Step 1 fully accepted based on Phase A alone.

## Required final response format (short)

```text
Repository / branch / PR:
Base commit and current commit:
What I executed (actual commands):
Changes made (paths):
Tests: executed / passed / failed / skipped:
Gate table: A0 … A3, then Phase B if applicable:
Reference parity differences:
Data-safety check:
Blockers (specific evidence, no guesses):
One next actionable step:
```

**Priority order:** Get A0/A1 working first, then A2/A3, then request review. Do not expand the scope while any earlier gate is unverified. Scientific novelty, model metrics and training are deliberately out of scope for Step 1.
