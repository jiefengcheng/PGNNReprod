# Step 1 Phase A execution report

**Date:** 2026-11-20  
**Repository:** https://github.com/jiefengcheng/PGNNReprod  
**Branch:** `work/txgnn-step1-reference-parity`  
**Base commit:** `7ca0c19fbe3abea7acaa02cb0c3f3ca084d9ea05`  
**Scope:** Phase A only. Phase B is not implemented. Step 1 is not accepted.

The workspace was already a clean checkout of `origin/main` at the base commit, so the feature branch was created there instead of by a second clone. No command below was described without being run. Status words are only `PASS`, `FAIL`, `BLOCKED`, or `NOT_RUN`, except the full owned pipeline row, which remains `NOT_IMPLEMENTED` as required by the handoff.

## Gate table

| Gate | Status | Evidence |
| --- | --- | --- |
| Source-only tests | PASS | Protocol venv: `12 passed, 1 skipped, 4 subtests passed`. The skip is `tests/test_inputs.py` because that venv has no PyTorch. Reference env: `13 passed, 4 subtests passed`, including the synthetic Xavier test. A skipped or synthetic input test is not reference input parity. |
| Dataset acquisition | PASS | Three Dataverse IDs downloaded with the documented fetch command after a User-Agent fix. Sizes match Dataverse `filesize`. `kg.csv` and `edges.csv` MD5 match Dataverse metadata. `node.csv` is the ingested TSV, not the original CSV; see below. Schema checks passed. Files are gitignored. |
| Pinned runtime/import | BLOCKED | README stack Python 3.8 / DGL 0.5.2 cannot be installed on this host. Imports from the pinned checkout were proven on a documented substitute: Python 3.10.22, PyTorch 2.1.2.post4 CPU, DGL 1.1.3. PyG was not installed and was not used. |
| Reference snapshot | PASS | `scripts/stage1.py reference` exit 0. Split `complex_disease`, fold seed 42, preprocessing seed 42, Torch input seed 0, input dimension 100. Snapshot `artifacts/local/reference_h8_6z264/snapshot`. Upstream SHA `0e0203c3939f7d12da27aaf10e55daaff5424023`, `txgnn` tree clean, `TxData` loaded from that tree. Produced with DGL 1.1.3, not DGL 0.5.2. |
| Comparison A | FAIL | `build_graph` rejected the reference splits before edge or input comparison. 33 identical `drug_protein` facts are in both train and valid. Edge arrays, input tensors, RNG states, and fresh-process restoration were not compared. |
| Full owned raw-to-split pipeline | NOT_IMPLEMENTED | No independent raw normalization, disease-group crosswalk, or split reconstruction was added. |
| No-data policy | PASS | `python scripts/check_no_data.py` before this commit: `PASS: 23 staged/tracked files`. After staging this report and `tests/test_assets.py`: `PASS: 25 staged/tracked files`. |

## A0 — baseline

Host: macOS 27.2, arm64, Apple M5 Max. System Python used for the protocol venv: 3.14.5.

```bash
git switch -c work/txgnn-step1-reference-parity
python3 -m venv .venv
.venv/bin/python -m pip install -e '.[test]'
.venv/bin/python -m pytest -q
.venv/bin/python scripts/check_no_data.py
```

Installed for that run: numpy 2.5.5, pandas 3.1.1, pytest 9.2.0, pip 26.3. No PyTorch.

First result, before the code changes in this PR:

```text
9 passed, 1 skipped, 4 subtests passed
SKIPPED tests/test_inputs.py: PyTorch not installed in minimal source-only CI
PASS: 23 staged/tracked files contain no prohibited data or artifacts
```

Both commands exited 0. `git status --short` was empty. `HEAD` was `7ca0c19fbe3abea7acaa02cb0c3f3ca084d9ea05`.

After the acquisition fix and the regression test, the same venv reports `12 passed, 1 skipped, 4 subtests passed`. The reference environment below reports `13 passed, 4 subtests passed`.

## A1 — data acquisition

First fetch, using unmodified `urllib.request.urlopen`:

```bash
.venv/bin/python scripts/stage1.py fetch --raw-dir data/raw
```

Exit 1. `urllib.error.HTTPError: HTTP Error 403: Forbidden` for `https://dataverse.harvard.edu/api/access/datafile/7144484`. A direct request with the default `Python-urllib` user agent returns 403. The same URL with user agent `PGNNReprod-stage1/0.1 (+https://github.com/jiefengcheng/PGNNReprod)` returns the file. `src/txgnn_stage1/assets.py` now sends that user agent. `tests/test_assets.py` checks the header without downloading the KG.

Second fetch, after that change, exit 0. Acquisition time was 2026-11-20T10:12:36Z through 2026-11-20T10:13:14Z. Lock file: `data/raw/asset_lock.json` (gitignored).

| Local file | Dataverse ID | Bytes | SHA-256 | Dataverse MD5 | Delimiter | Schema check |
| --- | --- | ---: | --- | --- | --- | --- |
| `kg.csv` | 7144484 | 981751236 | `0a5a010fcd88dcec572a573d98c26c1c244a11567092b6161d3ded611cfb5c69` | `aac8191d4fbc5bf09cdf8c3c78b4e75f` (matches) | comma | required columns present. Header: `relation`, `display_relation`, `x_index`, `x_id`, `x_type`, `x_name`, `x_source`, `y_index`, `y_id`, `y_type`, `y_name`, `y_source` |
| `node.csv` | 7144482 | 8893757 | `18ebcf24887e8529b95c7a7c96c81089cc7c5ba22d53a5c05d56480331961af8` | downloaded MD5 `4924de04fb3deefa1e0a8dada424538e` does **not** match metadata MD5 `7f9ab4109c54049e819ecd14e15a6038` | tab | required columns present. Header: `node_index`, `node_id`, `node_type`, `node_name`, `node_source` |
| `edges.csv` | 7144483 | 386582390 | `57c405049d0adb5ee9070d8b3fbb38f55a053b5c8d45b8b5af1ce23ee5d3ad7c` | `5d4d211a22e88544b78fde2735e797bc` (matches) | comma | required columns present. Header: `relation`, `display_relation`, `x_index`, `y_index` |

`node.csv` is a tabular Dataverse file. Metadata `filesize` 8893757 matches the default API download used by pinned `TxData` and by this fetcher. Metadata MD5 `7f9ab4109c54049e819ecd14e15a6038` matches `?format=original`: 7869553 bytes, comma-separated, SHA-256 `301644c421b436fb6d5f548954a5d77749d87ff8ecbf27d42f41c07a075a2ad3`, same column names. The original was hashed and removed. The pipeline keeps the default TSV because that is the file the pinned loader reads with `sep='\t'`.

`git check-ignore` matches `/data/` for all four files. No dataset bytes are in this commit.

## A2 — reference environment

Pinned checkout:

```bash
git clone https://github.com/mims-harvard/TxGNN.git reference/TxGNN
git -C reference/TxGNN checkout 0e0203c3939f7d12da27aaf10e55daaff5424023
git -C reference/TxGNN rev-parse HEAD
git -C reference/TxGNN diff --exit-code HEAD -- txgnn
```

`HEAD` is `0e0203c3939f7d12da27aaf10e55daaff5424023`. The `txgnn` tree has no local diff. The directory is gitignored.

Exact legacy stack, attempted and blocked:

- `uv pip install 'dgl==0.5.2'` on CPython 3.11.15, macOS arm64, exit 1. Resolver: wheels only for ABI tags `cp36m`, `cp37m`, `cp38`.
- `uv pip install 'dgl==2.2.1'` on the same interpreter, exit 1. PyPI files for 2.2.1 are `win_amd64` only.
- conda `dglteam` has `dgl==0.5.2` for `osx-64`, not `osx-arm64`. conda-forge `osx-arm64` builds start at 1.1.3.
- `arch -x86_64 /usr/bin/uname -m` failed: `Bad CPU type in executable`. Rosetta is not installed, so the `osx-64` DGL 0.5.2 package cannot run.

Substitute that was actually installed and imported, under ignored `artifacts/local/conda-ref`:

| Package | Version |
| --- | --- |
| Python | 3.10.22 (conda-forge) |
| pandas | 1.5.3 (`DataFrame.append` present; required by pinned `preprocess_kg`) |
| numpy | 1.26.4 |
| torch | 2.1.2.post4, default dtype float32, CPU |
| dgl | 1.1.3 (`dgl.ops.edge_softmax` imports) |
| scipy | 1.15.2 |
| scikit-learn | 1.7.2 |

`DGLBACKEND=pytorch`. `MPLBACKEND=Agg`. Editable installs: this repository with `--no-deps`, and `reference/TxGNN` with `--no-deps`.

Import proof:

```text
TxData file: reference/TxGNN/txgnn/TxData.py
repo is a parent of that file: True
```

`from txgnn import TxData` loads the pinned package's `__init__.py`, which also imports `TxGNN` and `TxEval`. Those imports succeeded. PyTorch Geometric was not installed. `complex_disease` does not call `DataSplitter.get_test_kg_for_disease`, so this run does not exercise PyG.

This substitute is not the upstream README environment. DGL 1.1.3 can order or store a heterograph differently from DGL 0.5.2. Edge-array parity against DGL 0.5.2 was not measured.

## A3 — reference snapshot and Comparison A

```bash
export DGLBACKEND=pytorch MPLBACKEND=Agg PYTHONUNBUFFERED=1
artifacts/local/conda-ref/bin/python scripts/stage1.py reference \
  --raw-dir data/raw \
  --upstream-repo reference/TxGNN \
  --split complex_disease \
  --fold-seed 42 \
  --preprocess-seed 42 \
  --input-seed 0 \
  --input-dim 100
```

Exit 0. Printed path: `artifacts/local/reference_h8_6z264/snapshot`. Upstream log included `Found local copy` for the three assets, `Creating splits...`, `split_data_path: artifacts/local/reference_h8_6z264/data/complex_disease_42`, `Creating DGL graph....`, and `Done!`.

Aggregate snapshot contents, not row values:

| Item | Measured value |
| --- | --- |
| Node types | 10, alphabetical DGL order: `anatomy`, `biological_process`, `cellular_component`, `disease`, `drug`, `effect/phenotype`, `exposure`, `gene/protein`, `molecular_function`, `pathway` |
| Node counts | anatomy 14033; biological_process 28642; cellular_component 4176; disease 17080; drug 7957; effect/phenotype 15311; exposure 818; gene/protein 27610; molecular_function 11169; pathway 2516 |
| Node total | 129312 |
| Training canonical relation types | 50 |
| Validation canonical relation types | 50 |
| Reverse-name map entries | 20 |
| `processed.csv` rows | 4050249 |
| `train.csv` rows | 6735722 |
| `valid.csv` rows | 959334 |
| `base_test.csv` rows | 405442 |
| Saved training edge endpoints | 6735722 across 50 `int64` arrays of shape `(2, N)` |
| Saved inputs | 10 CPU `float32` matrices, all finite, second dimension 100, row counts equal to the node counts above |
| `schema.json` SHA-256 | `d9d5246bb63f89b05990efa2e81d97693e28ade59b1e788025d62773796ae406` |
| `manifest.json` SHA-256 | `ad40e5fd3b76573b3e9545d3b24917734302e6b3f711e6cd26dbbb481d55e6d0` |
| `environment.json` SHA-256 | `e52701c6914c898bfa10ab88e38da36f97c22f5407b2ff842f23133b6bdaa520` |

Drug count 7957 and disease count 17080 match the pinned README. The node total 129312 is this reference graph's `num_nodes`. It is not the preprint's 123527, and the 6735722 training endpoints are not the preprint's 8063026 undirected edges. Those preprint figures were not used as pass criteria.

Comparison command actually run:

```bash
artifacts/local/conda-ref/bin/python scripts/stage1.py compare \
  --snapshot artifacts/local/reference_h8_6z264/snapshot
```

Exit 1:

```text
ValueError: Leakage between train and valid
```

The raised example is one `drug` / `drug_protein` / `gene/protein` fact. Identifiers are omitted here. Counts measured from the snapshot tables:

| Check | Result |
| --- | --- |
| Canonical-fact overlap, train vs valid | 33, all `drug_protein` from `drug` to `gene/protein` |
| Canonical-fact overlap, train vs base test | 12, same relation and endpoint types |
| Canonical-fact overlap, valid vs base test | 1, same relation and endpoint types |
| Exact directed tuple overlap, train vs valid | 66: 33 `drug_protein` and 33 `rev_drug_protein` |
| Base `drug_protein` rows | 25653, all with source type `drug` |
| Base unique `drug_protein` facts | 25468 |
| Base facts stored twice | 185 |
| Shared train/valid facts whose base multiplicity is 2 | 33 of 33 |

The reference split assigns rows, not unique facts. For those 33 facts, the two identical base copies landed in train and valid. Reverse augmentation then wrote the reverse copy into both splits. The owned leakage check is behaving as written. This failure is in the exported CSV splits. It does not depend on the DGL version.

Not compared, because `build_graph` stopped first: per-relation ordered edge arrays, edge-multiset hashes, CPU input values, pre/post Torch RNG states, and fresh-process restoration. Those items are `NOT_RUN`.

`tests/test_graph.py::test_same_directed_fact_in_two_splits_rejected` reproduces the same rejection on a two-row synthetic case.

## Unresolved blockers

1. Comparison A failed. The pinned `complex_disease` split, fold seed 42, contains duplicate `drug_protein` facts in more than one split. Ordered edge and input parity was not executed.
2. DGL 0.5.2 / Python 3.8 cannot run on this Mac. The snapshot comes from DGL 1.1.3. Even after the leakage failure is handled, edge order still needs a DGL 0.5.2 check or an explicit named variant.
3. PyG was not part of this run.
4. Independent raw-to-split preparation is not implemented.

## One next action

Classify the 185 duplicated base `drug_protein` rows and the 33 train/valid collisions as either reference behavior the owned constructor must keep rejecting, or a named split variant, and only then re-run `scripts/stage1.py compare` on this snapshot. Do not start Phase B on the strength of this run.
