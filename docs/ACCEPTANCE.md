# Step 1 acceptance contract

**Current status: synthetic graph construction only. Full-data parity is NOT_RUN.**

1. **Source identity:** each acquired file's Dataverse file ID, SHA-256, bytes, delimiter and schema are captured; raw files stay outside Git.
2. **Node registry:** canonical typed identities, original/grouped disease memberships and local indices are stable; alias and collision audit is recorded.
3. **Graph schema:** exact node-type and canonical relation order; full node inventory retained even for isolated test diseases.
4. **Split integrity:** forward/reverse facts remain in the same split; held-out relationships are absent from train adjacency; all applied masks are serialized.
5. **Graph parity:** per-relation ordered arrays equal the pinned reference, or deviations are recorded and named; multiset hash also compares edge multiplicity.
6. **Input parity:** CPU FP32 Xavier tensors match the pinned reference for a frozen node order and seed, or are a declared owned seed/version; actual tensors persist.
7. **Fresh-process replay:** independent process loads identical graph, mappings and inputs without invoking random data generation.
8. **No-data repository:** CI checks source files only; raw CSV, matrices, tensors, and trained checkpoint artifacts never enter git history.

This is a preparation milestone, not a claim of trained TxGNN performance or published metrics.
