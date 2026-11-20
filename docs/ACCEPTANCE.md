# Step 1 acceptance contract — independent implementation (NO DGL)

**Status:** The owned NumPy/Pandas `complex_disease` bundle has executed evidence in [STEP1_EXECUTION_REPORT.md](STEP1_EXECUTION_REPORT.md). Historical DGL/Torch parity remains out of scope. See [AGENT_HANDOFF_STEP1.md](AGENT_HANDOFF_STEP1.md).

1. **Source identity:** fixed released Dataverse file IDs, successful schema reads, and local SHA-256 manifest; source files remain outside Git.
2. **Typed registry:** external IDs, released global indices, model-local indices and original/grouped disease identities are kept distinct, reverse-resolvable, and free of unexplained normalization collisions.
3. **Preprocessing:** source-order-aware representative edge selection, stable typed index assignment and relation inventory are reproducible from locked raw inputs without importing upstream TxGNN.
4. **Splits:** explicit owned seed, saved membership, forward/reverse pairing, and no forbidden held-out therapeutic edges in training.
5. **Training graph:** full processed node inventory including isolated nodes; only training adjacency; canonical relation order; integer typed endpoints; multiplicity/ordered-array digests.
6. **Frozen inputs:** deterministic owned FP32 Xavier matrices, saved actual array bytes and dtype/shape/RNG metadata; historical Torch random bit parity **not required**.
7. **Independent replay:** fresh process restores the exact schema, mappings, split metadata, edges and inputs with no RNG and matching hashes.
8. **Proof:** hand-computed tests plus real-data aggregate validations (if assets obtained), with exact command logs, executed/skipped/failed counts, and known deviations from the pinned static code specification.
9. **No data committed:** the public repository contains only audited source, docs and synthetic tests. No source KG, derived fact tables, graph arrays, node matrices, or checkpoints.

**Acceptance is evidence-driven:** synthetic tests alone are not full real-data acceptance; inability to obtain the data must be reported BLOCKED. Do not claim equality to the authors' historical DGL graph or model performance. Any upstream numerical-oracle study is an optional separate lane.
