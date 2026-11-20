#!/usr/bin/env python3
"""Time one forward and one backward on synthetic heterogeneous graphs.

Writes a summary under artifacts/local/. Does not train and does not read the
released autoimmune bundle.
"""
import json
import resource
import time
import tracemalloc
from pathlib import Path
import numpy as np
from txgnn_stage1.model import rgcn_layer
from txgnn_stage1.model_grad import message_mean_gradients

NODE_TYPES = (
    "anatomy", "biological_process", "cellular_component", "disease", "drug",
    "effect/phenotype", "exposure", "gene/protein", "molecular_function", "pathway",
)
RELATIONS = (
    ("drug", "indication", "disease"),
    ("disease", "rev_indication", "drug"),
    ("gene/protein", "protein_protein", "gene/protein"),
    ("drug", "drug_protein", "gene/protein"),
    ("gene/protein", "rev_drug_protein", "drug"),
    ("disease", "disease_disease", "disease"),
    ("anatomy", "anatomy_anatomy", "anatomy"),
    ("exposure", "exposure_disease", "disease"),
)
WIDTH = 32
FROZEN_NODES = 129312
FROZEN_EDGES = 6406350
FROZEN_RELATIONS = 50
FROZEN_WIDTH = 100


def synthetic(n_edges, seed=0):
    rng = np.random.Generator(np.random.PCG64(seed))
    per_type = 256
    features = {
        node_type: rng.normal(size=(per_type, WIDTH)).astype(np.float64)
        for node_type in NODE_TYPES
    }
    edges = {}
    weights = {}
    biases = {}
    share, remainder = divmod(n_edges, len(RELATIONS))
    for index, triple in enumerate(RELATIONS):
        count = share + (1 if index < remainder else 0)
        source_type, relation, destination_type = triple
        edges[triple] = np.stack([
            rng.integers(0, per_type, size=count, dtype=np.int64),
            rng.integers(0, features[destination_type].shape[0], size=count, dtype=np.int64),
        ])
        weights[relation] = rng.normal(size=(WIDTH, WIDTH)).astype(np.float64) * 0.01
        biases[relation] = np.zeros(WIDTH, dtype=np.float64)
    return features, edges, weights, biases


def frozen_byte_estimate():
    activation = FROZEN_NODES * FROZEN_WIDTH * 4
    relation_affine = FROZEN_RELATIONS * (FROZEN_WIDTH * FROZEN_WIDTH + FROZEN_WIDTH) * 4
    relation_vector = FROZEN_RELATIONS * FROZEN_WIDTH * 4
    edge_index = FROZEN_EDGES * 2 * 8
    return {
        "input_float32": activation,
        "two_layer_activations_float32": activation * 2,
        "two_layer_affine_parameters_float32": relation_affine * 2,
        "distmult_relation_vectors_float32": relation_vector,
        "training_edge_index_int64": edge_index,
        "sum_of_those_terms": activation + activation * 2 + relation_affine * 2 + relation_vector + edge_index,
    }


def run_once(n_edges):
    features, edges, weights, biases = synthetic(n_edges)
    tracemalloc.start()
    started = time.perf_counter()
    hidden = rgcn_layer(features, edges, list(RELATIONS), weights, biases)
    forward_seconds = time.perf_counter() - started
    _, forward_peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    tracemalloc.start()
    started = time.perf_counter()
    for source_type, relation, destination_type in RELATIONS:
        upstream = np.ones_like(hidden[destination_type])
        message_mean_gradients(
            features[source_type], edges[(source_type, relation, destination_type)],
            weights[relation], upstream)
    backward_seconds = time.perf_counter() - started
    _, backward_peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return {
        "edges": n_edges,
        "forward_seconds": forward_seconds,
        "backward_seconds": backward_seconds,
        "forward_traced_peak_bytes": forward_peak,
        "backward_traced_peak_bytes": backward_peak,
        "process_ru_maxrss": rss,
    }


def main():
    report = {
        "width": WIDTH,
        "node_types": len(NODE_TYPES),
        "relations": len(RELATIONS),
        "runs": [run_once(count) for count in (10_000, 100_000, 1_000_000)],
        "frozen_autoimmune_byte_estimate": frozen_byte_estimate(),
        "note": "Synthetic edges only. The 6406350-edge bundle was not loaded.",
    }
    out = Path("artifacts/local/encoder_budget.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
