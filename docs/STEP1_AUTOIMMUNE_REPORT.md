# Autoimmune area split — execution report

**Date:** 2026-11-20  
**Protocol:** `autoimmune_owned_v1`  
**Branch:** `work/step1-autoimmune`  
**Base branch:** `work/step1-no-dgl` (`80813b9a206d0121afcf4546cf897a1ec9695f92`)

This run does not install TxGNN, DGL, or PyG, and it does not claim a historical unseeded mask or Torch bit patterns. `complex_disease_owned_v1` is unchanged.

## Commands

```bash
.venv/bin/python -m pytest -q
.venv/bin/python scripts/stage1.py prepare \
  --raw-dir data/raw \
  --out artifacts/local/owned_autoimmune_v1 \
  --group-map reference/TxGNN/txgnn/data_splits/kg_grouped_diseases_bert_map.csv \
  --obo reference/TxGNN/txgnn/data_splits/HumanDO.obo \
  --mondo-xref reference/TxGNN/txgnn/data_splits/mondo_references.csv \
  --disease-list reference/TxGNN/data/disease_files/autoimmune.csv \
  --split autoimmune \
  --fold-seed 42 --preprocess-seed 42 --input-seed 0 --input-dim 100 \
  --hops 2 --test-size 0.05
.venv/bin/python scripts/stage1.py verify --bundle artifacts/local/owned_autoimmune_v1
```

The same prepare command was repeated into `artifacts/local/owned_autoimmune_v1_repeat`.

Environment: Python 3.14.5, numpy 2.5.5, pandas 3.1.1.

## Auxiliary files, not committed

| File | Bytes | SHA-256 |
| --- | ---: | --- |
| `HumanDO.obo` | 5877929 | `e39e5f2052001c1c586fcbc16cab4746b584c386c455d28c17dfa1377d868cb2` |
| `mondo_references.csv` | 2499360 | `714f258b7b8c11bcbb6728b862a276d59a9dfd9b0b494a4a7d74b28449169972` |
| `autoimmune.csv` | 3065 | `4a818a685950ea09e58314e0518f162be836aa7b5c273b264ac977e7757af5b9` |
| grouped map | 1334344 | `fb863ca7b043a4b7066bb0d734e589da624eb69ca84e48f0fd77a5ad17898594` |

The three Dataverse files are the same locked assets as the complex-disease report.

## Results

| Check | Value |
| --- | --- |
| prepare / verify / repeat | exit 0 / exit 0 / exit 0 |
| Repeat file identity, excluding `manifest.json` | 68 files identical |
| DOID root | 417 |
| Normalized descendant closure | 88 |
| MONDO ids in that closure | 79 |
| Direct MONDO nodes | 60 |
| Grouped stand-ins | 15 |
| Seed global indices | 75 |
| `edges.csv` rows | 8100498 |
| Two-hop nodes reached | 29219 |
| Induced edges | 5318302 |
| Therapeutic edges in the mask | 1752 |
| Random draw | 403273 |
| Unique mask pairs | 404885 |
| Mask budget | 405025 |
| Endpoint pairs removed before orientation | 1214 |
| Base rows | 4049642 |
| Base split rows | train 3203175, valid 457597, test 388870 |
| Augmented rows | train 6406350, valid 915194, test 777740 |
| Training relations / endpoints | 50 / 6406350 |
| Node total | 129312 |
| Normalization collision groups | 0 |
| Therapeutic overlap train/valid, train/test, valid/test | 0, 0, 0 |
| Evaluation reverse-therapeutic rows dropped | 6413 |
| Evaluation rows kept | 771327 |
| Ordered graph hash | `4af13ede84b3bfd2829fd10ecfea2ea974797e8d2b9d52c81e3525d939b147d2` |

The unique mask is 140 pairs smaller than the budget because drug-disease edges and the random draw can select the same directed pair. The saved mask keeps one copy.

Tests: `23 passed, 0 failed, 1 skipped, 4 subtests passed`. The skip is the optional Torch helper.

## Owned decisions

- DOID tokens are normalized with `str(int(token))`. The static parser keeps leading zeros on `id` lines and removes them on some `is_a` lines, so those strings do not always join.
- The neighborhood draw is `PCG64(preprocess_seed)`. The static area mask uses an unseeded `np.random.choice`.
- Validation of the non-masked rows uses `PCG64(fold_seed)` and a 0.125 fraction. The static fold call used `random_state=1` and a zero test fraction.
- A surviving representative row is test when either directed global-index pair was masked. Orientation would otherwise be able to retain the opposite direction in train.
- The evaluation file drops reverse therapeutic test rows whose disease is outside `autoimmune.csv`. Those rows stay in the base test split and out of train.

## Next action

Do not retune this mask to chase a historical unseeded sample. The next lane is the encoder, not another area split.
