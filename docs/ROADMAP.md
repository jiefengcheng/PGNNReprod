# Implementation lane

## Objective
Independent, fully trainable TxGNN with owned preparation, training state, evaluation and model exports. A pinned public reference is an oracle, not the training dependency for owned implementation.

## Stage 1 — Preparation (current)

- [x] Preserve full Step 1 notebook as a source-only artifact.
- [x] Extract independent indexed-table graph constructor and synthetic tests.
- [x] Exclude KG data and generated tensors/checkpoints from Git.
- [ ] Acquire and fingerprint fixed raw TxGNN KG assets locally.
- [ ] Freeze reference Python/DGL/PyTorch/PyG environment.
- [ ] Export pinned reference snapshot and compare graph/input arrays.
- [ ] Independently implement raw ID normalization, grouped disease identity and representative-edge selection.
- [ ] Independently implement and validate split/mask preparation with reverse-edge protection.
- [ ] Freeze graph + inputs + mapping bundle; pass fresh-process reload.

## Later lanes (not implemented)

- Stage 2: RGCN encoder and DistMult predictor.
- Stage 3: all-relation pretraining and drug–disease fine-tuning.
- Stage 4: protected evaluation, baseline comparisons and repeatability.
- Stage 5: GraphMask explainer and scientific evidence auditing.

Keep any scientific deviation from upstream in a separate named experiment.
