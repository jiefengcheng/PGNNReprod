# Step 1 execution report — independent NumPy/Pandas bundle

**Date:** 2026-11-20  
**Protocol:** `complex_disease_owned_v1`  
**Claim:** The checks below pass for this owned protocol. They are not a claim of historical DGL graph identity, Torch random-bit identity, trained metrics, or the preprint's published totals.

```text
Base and head SHA; branch; PR:
Base: bf010b9962a219f193f338fcb68be0da83f8d543
Branch: work/step1-no-dgl
Head: recorded at commit time
PR: opened from this branch

Environment and exact executed commands:
macOS 27.2 arm64
Python 3.14.5
numpy 2.5.5
pandas 3.1.1
pytest 9.2.0
The prepare/verify interpreter is .venv and does not import txgnn, dgl, or torch_geometric.

.venv/bin/python -m pytest -q
.venv/bin/python scripts/check_no_data.py
.venv/bin/python scripts/stage1.py prepare \
  --raw-dir data/raw \
  --out artifacts/local/owned_complex_disease_v1 \
  --group-map reference/TxGNN/txgnn/data_splits/kg_grouped_diseases_bert_map.csv \
  --fold-seed 42 --preprocess-seed 42 --input-seed 0 --input-dim 100
.venv/bin/python scripts/stage1.py verify \
  --bundle artifacts/local/owned_complex_disease_v1
.venv/bin/python scripts/stage1.py prepare \
  --raw-dir data/raw \
  --out artifacts/local/owned_complex_disease_v1_repeat \
  --group-map reference/TxGNN/txgnn/data_splits/kg_grouped_diseases_bert_map.csv \
  --fold-seed 42 --preprocess-seed 42 --input-seed 0 --input-dim 100

Assets (IDs, sizes, SHA-256; no contents):
kg.csv    id 7144484  bytes 981751236  sha256 0a5a010fcd88dcec572a573d98c26c1c244a11567092b6161d3ded611cfb5c69
node.csv  id 7144482  bytes 8893757    sha256 18ebcf24887e8529b95c7a7c96c81089cc7c5ba22d53a5c05d56480331961af8
          tab-separated released node table. Dataverse metadata MD5 matches format=original
          (7869553-byte CSV), not these TSV bytes. The loader reads the TSV, as the static TxData path does.
edges.csv id 7144483  bytes 386582390  sha256 57c405049d0adb5ee9070d8b3fbb38f55a053b5c8d45b8b5af1ce23ee5d3ad7c
          Locked and unused. The static complex_disease path does not read it either.
Group map (not committed; read from the pinned checkout file
txgnn/data_splits/kg_grouped_diseases_bert_map.csv):
sha256 fb863ca7b043a4b7066bb0d734e589da624eb69ca84e48f0fd77a5ad17898594

Independent preparation outputs (local paths, summary counts/hashes):
artifacts/local/owned_complex_disease_v1
Repeat directory artifacts/local/owned_complex_disease_v1_repeat
66 bundle files excluding manifest.json are byte-identical across the two runs.
graph ordered hash:   1ad9d24b8f0041350281e3859349921c5efc1f8ff405b84f11622d4c72c10e19
graph multiset hash:  242cc3fa5bbb64ac73e3fbe0baa298751dd32c0a960f57ba3c861331a264c1ce

Released KG rows read: 8100498
Representative base rows kept: 4050249
Node counts (typed endpoints of the processed KG):
  anatomy 14033
  biological_process 28642
  cellular_component 4176
  disease 17080
  drug 7957
  effect/phenotype 15311
  exposure 818
  gene/protein 27610
  molecular_function 11169
  pathway 2516
  total 129312
Training canonical relations: 50, all with at least one training edge
Training directed endpoints: 6733072
Augmented rows: train 6733072, valid 963472, test 403954
Base rows before reverse: train 3366536, valid 481736, test 201977
Diseases with a therapeutic edge: 2054 (train 1707, valid 244, test 103)
These counts are this protocol's counts. They are not the preprint totals
(123527 nodes, 8063026 undirected edges).

Tests: run/passed/failed/skipped:
.venv/bin/python -m pytest -q
20 passed, 0 failed, 1 skipped, 4 subtests passed
The skip is tests/test_inputs.py because this venv has no PyTorch.
That skip is not a NumPy input failure. tests/test_preprocess.py covers the NumPy Xavier matrices.
prepare exit 0. verify exit 0. repeat prepare exit 0.

Identity, reverse-edge and holdout-leakage checks:
Normalization collision groups: 0
Global index pointing at more than one typed id: 0
Typed identities with two released global indices: 63 (129249 identities have one index; none have more than two)
Crosswalk rows: 129375, equal to the node-file row count
Node-file normalized ids absent from the graph: 0
Matched typed ids: 129312
Source conflicts between KG and node file: 0
Grouped-disease map rows: 6392
Resolved group ids: 6392
Unresolved group ids: 0
Distinct original ids: 6392
Distinct group ids: 1267
Therapeutic directed tuples, including rev_ names:
  train 70398, valid 11666, test 3198
  train/valid overlap 0
  train/test overlap 0
  valid/test overlap 0
build_graph also rejected any shared canonical fact, therapeutic or otherwise, before writing edges.

Input generator, shape/dtype/seed, save and independent reload:
Algorithm: numpy.random.Generator(PCG64)
Seed: 0
One uniform draw per node type, in alphabetical type order, on [-sqrt(6/(N+100)), +sqrt(6/(N+100))]
Stored dtype: float32
Dimension: 100
Fresh-process reload checked shapes, dtype, finiteness, and edge index bounds without calling the RNG.
A second prepare reproduced every bundle file except manifest.json.
These arrays are not Torch CPU Xavier bit patterns.

Source-derived decisions vs owned choices:
From the static TxGNN source, not by executing it:
- convert2str
- homogeneous unordered-endpoint de-duplication on raw id text, before normalization
- other relations keep the first row's source type
- lexicographic per-type indices after normalization
- therapeutic relations are indication, contraindication, and off-label use
- split fractions 0.83125 / 0.11875 / 0.05, with disease cuts using int(fraction * count)
- reverses are created after the split; heterogeneous names use rev_; homogeneous names stay and endpoints swap

Owned choices:
- Split and input draws use PCG64, not pandas.DataFrame.sample and not the legacy NumPy MT19937 stream.
- preprocess_seed is stored and is not consumed. Representative selection has no random draw.
- Duplicate directed facts are assigned together and keep their multiplicity. A row-wise sample can place copies in different splits; this protocol does not.
- Node-type order is alphabetical. Training relation order is first-seen order in the reverse-augmented training table.
- edges.csv is not an input to this split.

Known deviations and blockers:
- No DGL, TxGNN, or PyG runtime was used. Historical numerical parity was not attempted.
- The optional Torch input test is skipped in the source-only venv.
- Autoimmune and other disease-area splits are not implemented.
- 63 typed identities have two global indices. That is reported; it is not an id-normalization collision.
- The grouped map is a file in the pinned source tree. It is not one of the three Dataverse downloads and it is not committed here.

Next single action:
Keep this protocol fixed. A later lane can add the autoimmune area split under a separate protocol id. Do not reopen DGL installation for Step 1 acceptance.
```
