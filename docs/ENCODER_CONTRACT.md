# Encoder contract

**Target id:** `pinned_nonattention_v1`  
**Source:** TxGNN commit `0e0203c3939f7d12da27aaf10e55daaff5424023`, `HeteroRGCNLayer.forward` and `DistMultPredictor.apply_edges`.  
**Reference implementation:** [src/txgnn_stage1/model.py](../src/txgnn_stage1/model.py), float64 forward only.

This is the encoder a later port must match. [src/txgnn_stage1/model.py](../src/txgnn_stage1/model.py) is a numerical reference for these equations. It is not the final trainable architecture.

## Layer

For relation name `r` with weight `W_r` of shape `(out, in)` and bias `b_r` of shape `(out,)`:

```text
message(e) = W_r * source(e) + b_r
h_r(v) = mean { message(e) : e arrives at v on r }
h(v) = sum_r h_r(v)
```

A node with no incoming message on any relation has `h(v) = 0`. The output keeps one row per node in the frozen type order.

The encoder has two layers. Leaky ReLU with negative slope `0.01` is applied to every coordinate after layer 1 and is not applied after layer 2. The relation-name affine map is shared by every canonical triple that uses that name. This graph's names are unique, so each training relation has one map.

## Score

Relation vectors are indexed by the frozen canonical-relation order. For a directed pair with embeddings `h_u` and `h_v` and relation vector `h_r`:

```text
score = sum_k h_u[k] * h_r[k] * h_v[k]
```

Pretraining probabilities are `sigmoid(score)`. The raw score is the value used before that sigmoid.

## Parked variant

`paper_residual_v1` is the 2023 paper's Equation 3 residual update, in which a transformed previous representation is added back to the aggregated messages. The pinned `HeteroRGCNLayer.forward` does not add that residual. `paper_residual_v1` is not the target of this gate and is not implemented.

## Out of scope

Attention, disease-similarity prototypes, degree-dependent mixing, negative sampling, and parameter updates are specified separately or left for a later milestone. See [PROTOTYPE_CONTRACT.md](PROTOTYPE_CONTRACT.md).
