# Step 1 runbook — TxGNN heterogeneous graph and frozen node inputs

**Track:** Independent TxGNN reproduction · **Owner repository:** [jiefengcheng/PGNNReprod](https://github.com/jiefengcheng/PGNNReprod) · **Reference SHA:** `0e0203c3939f7d12da27aaf10e55daaff5424023`  
**Scope:** Construct the **training** heterogeneous graph, typed identity registry and frozen input tensors. This is **not** model training or GraphMask implementation.

> **Current execution status (2026-11-20):** The owned `complex_disease_owned_v1` pipeline has been executed on the locked Dataverse files. See [STEP1_EXECUTION_REPORT.md](STEP1_EXECUTION_REPORT.md). That run does not claim historical DGL or Torch-bit parity.

> **CURRENT EXECUTION OVERRIDE (2026-11-20):** **Do NOT install or run TxGNN, DGL or PyG for Step 1.** This lane is an **independent Python/NumPy/Pandas implementation**. The published source code is a static specification. All historical upstream-runtime parity and original Torch random-bit matching discussed in older drafts are optional future work, **not acceptance prerequisites**. The authoritative next-agent instructions are [here](AGENT_HANDOFF_STEP1.md).

## Goal and evidence boundary

**Acceptance statement:** Given the versioned released TxGNN assets and a declared split, an independent implementation must produce a complete typed node registry, source-to-model crosswalks, ordered canonical relations, split-aware training edges, and **actual persisted CPU FP32 node-input tensors**. A new process must restore them without drawing random values. Validate source-derived invariants and hand-computed fixtures; describe decisions where the static code leaves ambiguity. A running upstream oracle is not required.

The 2023 TxGNN methods (§1, pp. 22–23) describe a graph with ten node types, 29 undirected edge types, 123,527 nodes and 8,063,026 edges, compiled from multiple biomedical resources. The authors harmonized DrugBank/MONDO identities, selected a largest connected component and applied **manually reviewed disease grouping** (22,205 original MONDO concepts into 17,080 disease groups). Therefore:

- **Start with the authors' released harmonized snapshot**, not today's releases of all upstream data sources.
- Treat historical ontology/clinical curation as a fixed **upstream data product**; independently reproduce the *downstream training preparation*.
- Preserve source global indices, typed model-local indices and original/grouped disease memberships separately. **Never use names alone as keys.**
- This project keeps all downloaded datasets, graph arrays, masks, input matrices, checkpoints and raw data derived outputs **out of Git**. Public code and literature URLs are reference links, not permission to commit data bytes.

See [SOURCES.md](SOURCES.md), [ACCEPTANCE.md](ACCEPTANCE.md), and the full [Step 1 notebook](../notebooks/TxGNN_Step1_Graph_and_Inputs.ipynb).

## 1. Freeze the reproducible experiment identity

Reference:

- Code: https://github.com/mims-harvard/TxGNN/tree/0e0203c3939f7d12da27aaf10e55daaff5424023
- Paper: https://doi.org/10.1101/2023.03.19.23287458
- Released graph: https://doi.org/10.7910/DVN/IXA7BM
- Entry-point code: `txgnn/TxData.py`, `txgnn/utils.py`, `txgnn/data_splits/datasplit.py`
- Reproduction driver: `reproduce/train.py`

For the first non-area integration fixture, use:

| Setting | Value | Meaning |
| --- | --- | --- |
| `split` | `complex_disease` | Preliminary integration setting; not the final autoimmune study |
| `fold_seed` | `42` | Split assignment |
| `preprocess_seed` | `42` | Explicit seed for our preprocessing protocol |
| `input_seed` | `0` | Torch CPU input-tensor initialization |
| `input_dim` | `100` | Matches pinned reproduction driver (the 2023 preprint separately lists 512) |
| `test_size` | `0.05` | Disease-area masking default, where applicable |
| `one_hop` | `false` | Reference two-hop selection path, where applicable |
| `mask_ratio` | `0.1` | Separate one-hop path parameter |
| `no_kg` | `false` | Preserve full graph context |

The **effective config**, file hashes, prepared split rows and actual tensors are the reproducibility contract. A seed alone is not sufficient. `autoimmune` will require a separately frozen split and mask; do not quietly relabel the `complex_disease` fixture.

## 2. Acquire and verify source assets (outside Git)

The pinned `TxData` class downloads **specific Dataverse file IDs**:

| Local path under ignored `data/raw/` | URL | Parser |
| --- | --- | --- |
| `kg.csv` | https://dataverse.harvard.edu/api/access/datafile/7144484 | CSV |
| `node.csv` | https://dataverse.harvard.edu/api/access/datafile/7144482 | **Tab-delimited** |
| `edges.csv` | https://dataverse.harvard.edu/api/access/datafile/7144483 | CSV |

The raw files, not their basename alone, identify the source snapshot. File IDs may differ from generic PrimeKG links. Acquire with:

```bash
git clone https://github.com/jiefengcheng/PGNNReprod.git
cd PGNNReprod

python -m venv .venv
source .venv/bin/activate
pip install -e '.[test]'
python -m pytest -q

# Opt-in download into gitignored directory; see README for runtime caveats.
python scripts/stage1.py fetch --raw-dir data/raw
```

For every downloaded file, record original URL and Dataverse ID, bytes, SHA-256, delimiter, required columns, acquisition time, and parsing errors. Reject a truncated download or mismatched lock. The acquisition module is `src/txgnn_stage1/assets.py`.

Additional pinned-repository resources (also not committed as copies) include:

- `txgnn/data_splits/HumanDO.obo`
- `txgnn/data_splits/mondo_references.csv`
- `txgnn/data_splits/kg_grouped_diseases_bert_map.csv`
- `data/disease_files/<split>.csv`

**Gate A — asset integrity:** All expected files pass format/schema verification and are locked by content hash. No generated asset or source KG appears in `git ls-files`.

## 3. Build a lossless typed identity and disease-group registry

Expected released node types: `drug`, `disease`, `gene/protein`, `effect/phenotype`, `anatomy`, `pathway`, `biological_process`, `molecular_function`, `cellular_component`, `exposure`. Confirm their exact spelling and counts from the acquired TxGNN data before freezing.

Three identifier layers have different meanings:

| Identity layer | Example columns | Role |
| --- | --- | --- |
| **External** | type, DrugBank/MONDO/GO/etc. ID, name, source | Interpret and cross-reference biomedical claims |
| **Source global** | `node_index`, `x_index`, `y_index` | Ontology-neighborhood masking and released edge joins |
| **Model-local per type** | `x_idx`, `y_idx` | Rows of input tensors, DGL node IDs, relation adjacency |

The pinned `preprocess_kg` uses `convert2str`: a numeric-looking ID without underscore is cast to `float` then string; type-local indices are assigned from **sorted normalized strings**. As a result, e.g. `"00123"` becomes `"123.0"`; lexical sorting can put `"10.0"` before `"2.0"`. For compatibility, reproduce this behavior *exactly*, but retain the raw ID for audit and flag collisions rather than silently merging them.

Build:

- A `nodes` table: `node_type, local_idx, reference_id, name, source`.
- A source-index crosswalk: released global index → typed local node.
- A disease-membership crosswalk: original MONDO concept → released grouped disease identity → local disease index.
- Alias/collision and missing-identity reports. One local index is unique only **within a type**.

Do **not** rerun today's ClinicalBERT grouping and present it as the authors' 2023 curated groups. The published grouping included manual decisions; use the released map.

**Gate B — identity fidelity:** Every prepared edge endpoint maps to one typed node; reverse lookup recovers the source identity; no unexplained normalization collision remains.

## 4. Produce the representative base edge table

Starting from released `kg.csv`, reproduce the pinned preparation rules:

1. Preserve source-row ordering and provenance.
2. Determine the representative direction of each released relation as the pinned `preprocess_kg` does, with homogeneous relation de-duplication by unordered endpoints.
3. Normalize endpoint IDs and assign type-local integer indices.
4. Preserve the full processed inventory, canonical endpoint types, relation label, and, for disease-area experiments, assigned `split` column.
5. Keep source `relation` distinct from display-oriented text `display_relation`.

A relation must be identified by the **full canonical triple** `(source_type, relation_name, target_type)`; never infer meaning from relation name alone without checking the observed type pairing. The upstream graph constructor assumes a relation name identifies a consistent type pairing, which should be asserted against data.

Examples of therapeutic edges are:

```python
("drug", "indication", "disease")
("drug", "contraindication", "disease")
("drug", "off-label use", "disease")
```

Retain actual biological *absence* relationships (phenotype absent, expression absent, etc.) as ordinary KG relations. They are **not** negative examples generated by a later sampler.

Flag special reference behavior instead of silently "repairing" it: representative direction can depend on source row order; area preprocessing may de-duplicate using endpoints without relation; reverse augmentation can introduce homogeneous self-loop multiplicity. Quantify real-data effects before defining a corrected variant.

**Gate C — base table fidelity:** deterministic base relation inventory, ordered rows, multiplicities and type-local mappings agree with hand-computed fixtures and source-derived rules; record any ambiguity.

## 5. Create the split-aware base membership and capture masks

**Splitting occurs before reverse edge generation.** For `complex_disease` the reference separates therapeutic edges by disease, with other relations partitioned by its fold routine. For disease-area splits:

1. Use the pinned Disease Ontology and DOID→MONDO/grouped identity map.
2. Resolve the selected ontology root plus descendants to source global nodes.
3. Select all applicable drug–disease therapeutic edges for the held-out area.
4. Sample the specified biomedical neighborhood edge set under **an explicitly controlled NumPy seed**.
5. Save the **actual selected mask**, not merely the seed and fraction.
6. Freeze base `train/valid/test` membership and any final evaluation-only filtering separately.

For `autoimmune` the pinned `preprocess_kg` maps to DOID `417`; its default `get_edge_group` path uses a **two-hop** neighborhood (despite a misleading comment), with budget derived from the global edge count. The `one_hop` branch instead applies its own local mask ratio. Both must have distinguishable experiment IDs.

The pinned area split uses ambient `np.random` state in neighbor sampling: `prepare_split(seed=...)` alone does **not** deterministically fix the earlier mask. Our protocol seeds it explicitly and archives selected pairs, pre/post RNG state and split files.

The cached base test file and filtered `evaluation_test` view are not always identical. Keep both; final evaluation-only filtering need not preserve all base reverse counterparts.

**Gate D — holdout integrity:** no held-out forbidden therapeutic relation is present in training, forward or reverse. Saved partition membership and mask are byte-for-byte reloadable.

## 6. Generate reverse edges only within each partition

For each already partitioned base fact:

- If source and target types differ, add a reverse fact with type `(target_type, "rev_" + relation, source_type)`.
- If they are the same type, reverse the endpoints while **retaining the same relation name**, in line with the pinned reference and methods §3.2.
- Keep the forward and reverse representations of the **same base fact in the same partition**.
- Preserve homogeneous self-loop multiplicity for reference parity; any alternative de-duplication policy belongs in a separately named experiment.

Assert no reverse-only copy of a held-out fact survives in the training adjacency.

**Gate E — directed split integrity:** no forbidden holdout fact or reverse copy enters training; edge types/counts and multiplicities are reproducible under the declared owned policy.

## 7. Construct the heterogeneous training graph

Use the **full processed node inventory** and **training edges only**:

\[
G_{\mathrm{train}} = (V_{\mathrm{processed}}, E_{\mathrm{train,directed}}).
\]

This is crucial: held-out or isolated disease nodes retain valid local IDs and input rows, even when all their eligible edges have been masked. Do not recompute a largest connected component after splitting.

Keep:

- Ordered node types and `num_nodes_dict`.
- Ordered `canonical_etypes` and relation-parameter-row mapping.
- Ordered per-relation `int64` source/destination arrays (edge-ID order).
- Per-relation multiplicities and a separate multiset equality digest.
- Reference-compatible relation schema: do not silently inject holdout-only empty relation types into training and change the parameter order.

The independently tested implementation for this stage is `src/txgnn_stage1/graph.py`. It consumes **already prepared** indexed tables, not raw knowledge-base dumps.

**Gate F — constructor integrity:** independently prepared tables yield the expected typed counts, ordered relation arrays and multiplicity; report which raw transformations remain unverified. No live DGL reference is required.

## 8. Initialize and persist fixed node inputs

For each type \(t\), produce \(\mathbf X_t\in\mathbb R^{N_t\times d}\), with \(d=100\) in the pinned-code experiment. The reference creates CPU FP32 tensors in graph node-type order using **one Xavier-uniform draw per full matrix** and `requires_grad=False`. In the default gain case, Xavier's bound is \(\sqrt{6/(N_t+d)}\); do not Xavier-initialize each row individually.

Example:

```python
import torch
torch.manual_seed(0)                    # set AFTER imports
features = {}
rng_before = torch.get_rng_state().clone()
for node_type in ordered_node_types:
    x = torch.empty((node_counts[node_type], 100), dtype=torch.float32, device="cpu")
    torch.nn.init.xavier_uniform_(x)
    x.requires_grad_(False)
    features[node_type] = x
rng_after = torch.get_rng_state().clone()
```

Save all actual tensors, their ordered node registries, dtype/shape, logical hashes and RNG states. Later training **loads** those tensors; it must not generate them again as a side effect of model restore. A NumPy generator may produce the same distribution but not the same bit patterns.

Implemented helper: `src/txgnn_stage1/inputs.py`.

**Gate G — owned input replay:** saved NumPy FP32 Xavier matrices reproduce exactly under the chosen owned RNG/version and load in a fresh process without invoking RNG. Historical Torch bit identity is not required.

## 9. Prove the **independent** source-to-bundle pipeline

**Mandatory comparison 1 — hand-computed synthetic fixtures.** Exercise raw ID normalization, orientation, distinct typed IDs, grouped disease memberships, base split isolation, reverse relations, self-loops, relation ordering, held-out/isolated node retention, frozen features and exact serialized reload.

**Mandatory comparison 2 — real released data invariants.** Starting from locally downloaded and SHA-locked release files, run the owned pipeline end to end without importing `txgnn`, `dgl`, or `torch_geometric`. Save local (gitignored) aggregate reports of counts by node/relation/split, missing mappings, collision counts, graph hashes and data leakage checks. Repeat from the identical inputs and compare output hashes.

**Optional future investigation — historical reference parity.** The repository retains `reference.py` and `compare.py` as archival tools. Do **not** attempt their DGL/TxGNN setup as part of this milestone. The older reference-snapshot method only compared a graph built from already prepared tables; it did not establish independent raw-data preparation.

Example workflow (the raw-to-bundle command is **to be implemented by the agent**, not a currently working command):

```bash
python -m pytest -q
python scripts/check_no_data.py
python scripts/stage1.py fetch --raw-dir data/raw
# Implement and document your owned `prepare`/`verify` commands.
# Rerun them from a clean local output directory and compare hashes.
```

**Gate H — acceptance:** all real-data checks executed and passed or precisely marked blocked; deterministic output hashes and independent reload; no forbidden training leakage; no data files committed.

## 10. Durable outputs, acceptance and next handoff

The final **local ignored** preparation bundle should include:

| Artifact | Required meaning |
| --- | --- |
| `manifest.json` | Dataset/code/environment hashes, parameters and artifact ledger |
| `nodes.parquet` | Typed node-local registry and external identities |
| `source_node_crosswalk.parquet` | Source global indices → typed local nodes |
| `disease_membership.parquet` | MONDO originals → manually curated released groups |
| `relations.json` | Ordered canonical triples, reverse mapping, parameter-row order |
| `base_edges.parquet` | Representative facts and source provenance |
| `split_membership.parquet` | Base fact split and reverse-pair membership |
| `evaluation_targets.parquet` | Final filtered evaluation view, separate from base test |
| `graph/edge_*.npy` | Training-only typed source/destination arrays |
| `inputs/input_*.npy` | Persisted CPU FP32 input matrices |
| `rng_states/` | State needed to audit generation |
| `checks.json` | Real-data parity, leakage and fresh-process results |

This is a **target artifact contract**, not a claim that all of these files are currently emitted by the staged implementation. The existing reference/compare utilities emit smaller named snapshots under `artifacts/local/`.

A future model-facing loader should expose `node_counts`, `node_types`, `canonical_etypes`, `edge_index`, `node_inputs`, `node_registry`, `relation_to_parameter_row`, `split_views`, and a content `fingerprint`.

**Step 1 is ACCEPTED only if:** locked assets, typed crosswalks, independent source-to-split-to-directed graph invariants, holdout-leakage exclusions, owned frozen input persistence and fresh-process restore pass. A passing toy test suite alone is not enough. Historical DGL parity is NOT required.

### Actual repo commands and no-data gate

```bash
# Run source-only tests without any datasets:
python -m pytest -q

# Check tracked-file policy before pushing any later change:
python scripts/check_no_data.py
git ls-files
```

Note: `.gitignore` is a prevention layer, not a guarantee against `git add -f`; Git history must be reviewed if an accidentally included data artifact is ever committed. Use artifact hashes in documentation instead of raw records. Current repository visibility is **public**.

### References / follow-on

- [2023 TxGNN methods](https://doi.org/10.1101/2023.03.19.23287458) — pp. 22–23 KG curation, p. 24 input initialization, pp. 33–35 splits and reverse edges.
- [Pinned TxGNN code](https://github.com/mims-harvard/TxGNN/tree/0e0203c3939f7d12da27aaf10e55daaff5424023) — read-only specification, not runtime prerequisite.
- [Harvard Dataverse source publication](https://doi.org/10.7910/DVN/IXA7BM) — source files, fetched separately.
- [Repo development roadmap](ROADMAP.md) — planned later model/training/evaluation stages.

**Next engineering task:** implement and verify the independent *raw-to-base-to-splits* pipeline, with an explicit holdout-leakage report and complete bundle loader. Do not start legacy TxGNN/DGL setup. Only advance to model work after this is accepted.
