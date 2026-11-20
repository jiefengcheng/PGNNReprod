# Implementation roadmap — independent TxGNN reproduction

## Objective

Own the full scientific preparation, training state, evaluation and model exports. The published paper and pinned public source (`0e0203c3939f7d12da27aaf10e55daaff5424023`) are **static specifications**, not a runnable dependency of Step 1.

## Stage 1 — heterogeneous graph and frozen inputs (CURRENT, NO DGL)

- [x] Preserve the original Step 1 research notebook.
- [x] Implement a graph constructor from already prepared indexed tables and synthetic tests.
- [x] Add local asset acquisition, no-data guards, and Step 1 runbook.
- [ ] Acquire/fingerprint archived raw TxGNN files locally, or document exact access blocker.
- [ ] Independently implement raw ID normalization, curated grouped disease crosswalks, and representative edge selection using NumPy/Pandas.
- [ ] Independently implement the declared `complex_disease` split, protect held-out therapeutic facts, then add reverse edges per split.
- [ ] Build the training-only NumPy adjacency using full node inventory; serialize ordered canonical triples and all mappings.
- [ ] Generate and persist **owned deterministic** FP32 Xavier initial features; no original Torch-bit parity required.
- [ ] Prove independent fresh-process bundle reload and rerun determinism; publish only sanitized checks/results, not data bytes.

**Not required:** installing/running upstream TxGNN, DGL, PyG, Torch, pretrained checkpoints, or training on a GPU. Legacy reference exporter/comparer in this repo is archival/optional only and is not a Step 1 gate.

## Later lanes

- Stage 2: independently implement RGCN encoder and DistMult scoring, with small numerical unit tests.
- Stage 3: owned pretraining and fine-tuning.
- Stage 4: protected evaluation, comparison and repeatability.
- Stage 5: GraphMask explainer and scientific evidence auditing.

Keep scientific deviations from the published description as explicit, separately versioned methods.
