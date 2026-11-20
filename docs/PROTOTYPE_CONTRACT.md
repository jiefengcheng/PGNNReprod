# Prototype and degree-augmentation contract

**Status:** interface only. Not implemented.  
**Source:** TxGNN commit `0e0203c`, `DistMultPredictor.forward` when `proto` is true, plus `obtain_disease_profile`, `sim_matrix`, and `exponential` in `utils.py`.

A TxGNN model is not complete until this contract is implemented and tested. This document does not add that implementation. The readiness encoder scores frozen embeddings with DistMult and does not modify them.

## Relations that participate

Only these canonical triples use the prototype path:

| Source | Relation | Destination | Disease end |
| --- | --- | --- | --- |
| drug | contraindication | disease | destination |
| drug | indication | disease | destination |
| drug | off-label use | disease | destination |
| disease | rev_contraindication | drug | source |
| disease | rev_indication | drug | source |
| disease | rev_off-label use | drug | source |

## Degree-nonzero selection

For a participating triple, the disease query set is:

- source nodes whose out-degree on that triple is nonzero, when the disease is the source
- destination nodes whose in-degree on that triple is nonzero, when the disease is the destination

The key set is the same rule applied to the full training graph rather than to the current minibatch. Nodes with degree zero on that triple are not queries and are not keys.

## Profile

`all_nodes_profile` concatenates binary neighbor indicators for `disease_disease` over disease nodes and `rev_disease_protein` over gene/protein nodes.

`all_nodes_profile_more` adds `disease_phenotype_positive` over effect/phenotype nodes and `rev_exposure_disease` over exposure nodes.

`protein_profile` uses only `rev_disease_protein`.

`protein_random_walk` starts at `rev_disease_protein`, continues on `protein_protein` for `path_length` steps, repeats `num_walks` times, and stores either a bit vector or a visit frequency.

`bert` and `profile+bert` require external disease vectors that are not in this repository. They stay unspecified until those arrays are given a local hash.

Similarity is cosine with norms floored at `1e-8`:

```text
sim(a, b) = normalize(a) dot normalize(b)
```

## Gated mix

Training retrieval drops the top match, which is the disease itself, and keeps the next `proto_num` keys. Evaluation keeps `proto_num` keys. Coefficients are L1-normalized. `out` is the corresponding sum of key embeddings.

The query embedding is then mixed with `out`. The pinned code selects one of these mixes; a later implementation must name which one it uses:

| Name | Mix |
| --- | --- |
| learn | `sigmoid(W_gate([query, out]))` gates `out` against the query. `W_gate` maps `2 * width` to 1. |
| heuristics-0.8 | `0.8 * query + 0.2 * out` |
| avg | `0.5 * query + 0.5 * out` |
| rarity | coefficient `lambda * exp(-lambda * degree) + 0.2`, using the training-graph degree on that triple |
| 100proto | `out` alone |

The augmented disease rows replace the encoder embedding only while that triple is scored. The original rows are restored before the next triple.

## Tensors that would require gradients

| Tensor | Role | Gradient |
| --- | --- | --- |
| encoder output `h` | query and key embeddings | yes, into the RGCN |
| `W_gate` | learned mix | yes, only for `learn` |
| relation vector `h_r` | DistMult after the mix | yes |
| profile indicators and random-walk profiles | discrete neighbor structure | no |
| degree and `lambda` | rarity coefficient | no, unless `lambda` is declared trainable |

Until those paths have fixtures, the model-facing score is the plain DistMult score in [ENCODER_CONTRACT.md](ENCODER_CONTRACT.md).
