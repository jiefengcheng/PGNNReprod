# Model-port readiness

**Date:** 2026-11-20  
**Branch:** `work/model-port-readiness`  
**Decision:** the five gates below are met. This does not authorize full TxGNN training. [src/txgnn_stage1/model.py](../src/txgnn_stage1/model.py) stays a float64 forward reference. Prototypes are specified and not implemented.

## Gate table

| Gate | Status | Evidence |
| --- | --- | --- |
| Frozen graph contract | MET | [GRAPH_CONTRACT.md](GRAPH_CONTRACT.md). Model-facing bundle is `autoimmune_owned_v1`: 129,312 nodes, 50 relations, 6,406,350 training endpoints, float32 width 100, ordered hash `4af13ede84b3bfd2829fd10ecfea2ea974797e8d2b9d52c81e3525d939b147d2`. Preprint totals 123,527 nodes and 8,063,026 undirected edges are a different representation. |
| Mathematical contract | MET | [ENCODER_CONTRACT.md](ENCODER_CONTRACT.md). Target is `pinned_nonattention_v1`. `paper_residual_v1` is named and not implemented. |
| FP32 forward and gradients | MET | [src/txgnn_stage1/model_grad.py](../src/txgnn_stage1/model_grad.py) and `tests/test_model_grad.py`. Float32 forward matches the float64 reference. Analytical gradients for the weight, bias, source features, and DistMult vector match central differences. One SGD step on the two-node fixture lowers a finite squared loss. |
| Execution budget | MET | `scripts/benchmark_encoder.py` on synthetic graphs. The 6,406,350-edge bundle was not loaded and was not trained. |
| Prototype interfaces | MET | [PROTOTYPE_CONTRACT.md](PROTOTYPE_CONTRACT.md) defines the six drug-disease triples, degree-nonzero selection, profiles, similarity, and mixes. No prototype code was added. |

## Local tests

```text
.venv/bin/python -m pytest -q
30 passed, 0 failed, 1 skipped, 4 subtests passed
```

The skip is `tests/test_inputs.py` because this environment has no PyTorch. `python scripts/check_no_data.py` reported `PASS: 42 staged/tracked files`. CI now runs `python -m pytest -q` rather than only `tests/test_graph.py`. That remote run is not claimed here.

## Measured budget

Synthetic graph: 10 node types, 256 nodes each, 8 relations, width 32. One forward and one backward, no optimizer.

| Edges | Forward seconds | Backward seconds | Forward traced peak | Backward traced peak |
| ---: | ---: | ---: | ---: | ---: |
| 10,000 | 0.00206 | 0.00228 | 1,116,656 | 787,992 |
| 100,000 | 0.0164 | 0.0204 | 3,996,656 | 6,634,928 |
| 1,000,000 | 0.157 | 0.219 | 32,796,656 | 65,134,928 |

These timings use 256 nodes per type, so they measure edge aggregation more than the frozen 129,312-node activation set. Process `ru_maxrss` after the 1,000,000-edge run was 169,590,784 bytes on this Darwin host, where that field is in bytes.

Float32 storage estimate for the frozen autoimmune graph at width 100, excluding optimizer state and per-edge message buffers:

| Term | Bytes |
| --- | ---: |
| Input matrices | 51,724,800 |
| Two activation tensors | 103,449,600 |
| Two layers of affine parameters | 4,040,000 |
| DistMult relation vectors | 20,000 |
| Int64 training endpoint index | 102,501,600 |
| Sum | 261,736,000 |

## Still not authorized

Full-graph optimization, negative sampling, prototype execution, attention, and GraphMask. A complete TxGNN model still requires an implemented and tested [PROTOTYPE_CONTRACT.md](PROTOTYPE_CONTRACT.md).
