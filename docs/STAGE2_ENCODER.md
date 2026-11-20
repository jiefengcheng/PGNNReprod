# Stage 2 — RGCN encoder and DistMult scorer

**Date:** 2026-11-20  
**Branch:** `work/step2-rgcn-distmult`  
**Base:** `work/step1-autoimmune`

This stage adds an owned NumPy encoder. It does not train, sample negatives, or load a checkpoint. TxGNN and DGL are not imported.

The non-attention specification at TxGNN commit `0e0203c` is:

1. Apply one affine map per relation name to the source embedding.
2. Average the messages of each relation at the destination.
3. Sum those averages across relations.
4. Apply leaky ReLU with slope `0.01` after the first layer only.
5. Score a triple with DistMult: the sum of `head * relation * tail`. Pretraining then applies a sigmoid.

Owned choices, checked by `tests/test_model.py`:

- A node with no incoming messages is an explicit zero of the output width, so every Step 1 node row stays addressable.
- Attention, disease prototypes, and GraphMask are not implemented.
- Weights are arguments. Nothing in this module draws them or updates them.

`python -m pytest tests/test_model.py -q` passed 4 tests. The hand checks are the two-message mean `2.5`, the cross-relation sum `6`, the two-layer value `-2 * 0.01 * 3 + 1`, and the DistMult value `63`.

## Next action

Add an owned training step on the tiny fixture only: one positive triple, one corrupted triple, and a finite loss. Do not start full-graph optimization.
