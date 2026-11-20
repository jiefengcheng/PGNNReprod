"""Owned RGCN encoder and DistMult scorer. No DGL and no training loop.

The static TxGNN specification, commit 0e0203c, defines the non-attention layer as:

- one affine map per relation name, applied to the source embedding
- mean of those messages at each destination
- sum of the per-relation means
- leaky ReLU with slope 0.01 between two layers, and no activation after the second
- DistMult score sum(h_head * h_relation * h_tail)

Owned choices for this stage:

- NumPy only, so the checks run without PyTorch or DGL.
- A node with no incoming messages keeps an explicit zero vector. The node inventory
  therefore stays aligned with the Step 1 bundle.
- Prototype augmentation, attention, and GraphMask are not part of this module.
- Weights are caller-supplied. This module does not initialize or train them.
"""
import numpy as np

LEAKY_RELU_SLOPE = 0.01


def leaky_relu(values, negative_slope=LEAKY_RELU_SLOPE):
    values = np.asarray(values, dtype=np.float64)
    return np.where(values >= 0.0, values, negative_slope * values)


def _zeros_like_features(features, width):
    return {node_type: np.zeros((values.shape[0], width), dtype=np.float64)
            for node_type, values in features.items()}


def rgcn_layer(features, edges, canonical_etypes, weights, biases):
    """One heterogeneous layer. `weights[relation]` has shape (out, in)."""
    if not canonical_etypes:
        raise ValueError("canonical_etypes is empty")
    out_width = int(next(iter(biases.values())).shape[0])
    accumulated = _zeros_like_features(features, out_width)
    for source_type, relation, destination_type in canonical_etypes:
        if relation not in weights or relation not in biases:
            raise ValueError("Missing affine parameters for " + relation)
        endpoints = np.asarray(edges[(source_type, relation, destination_type)])
        if endpoints.size == 0 or endpoints.shape[1] == 0:
            continue
        weight = np.asarray(weights[relation], dtype=np.float64)
        bias = np.asarray(biases[relation], dtype=np.float64)
        source = np.asarray(features[source_type], dtype=np.float64)
        messages = source @ weight.T + bias
        source_index = endpoints[0].astype(np.int64, copy=False)
        destination_index = endpoints[1].astype(np.int64, copy=False)
        summed = np.zeros((features[destination_type].shape[0], out_width), dtype=np.float64)
        counts = np.zeros(features[destination_type].shape[0], dtype=np.float64)
        np.add.at(summed, destination_index, messages[source_index])
        np.add.at(counts, destination_index, 1.0)
        present = counts > 0.0
        summed[present] /= counts[present, None]
        accumulated[destination_type] += summed
    return accumulated


def encode(features, edges, canonical_etypes, layer_parameters, negative_slope=LEAKY_RELU_SLOPE):
    """Two RGCN layers. Leaky ReLU is applied only after the first layer."""
    if len(layer_parameters) != 2:
        raise ValueError("This encoder uses exactly two layers")
    hidden = rgcn_layer(features, edges, canonical_etypes,
                        layer_parameters[0]["weights"], layer_parameters[0]["biases"])
    hidden = {node_type: leaky_relu(values, negative_slope)
              for node_type, values in hidden.items()}
    return rgcn_layer(hidden, edges, canonical_etypes,
                      layer_parameters[1]["weights"], layer_parameters[1]["biases"])


def distmult(head, relation, tail):
    """Sum of the elementwise triple product. The last axis is the embedding."""
    head = np.asarray(head, dtype=np.float64)
    relation = np.asarray(relation, dtype=np.float64)
    tail = np.asarray(tail, dtype=np.float64)
    return np.sum(head * relation * tail, axis=-1)


def score_edges(embeddings, relation_vectors, canonical_etype, endpoints, relation_index):
    """Score one relation's endpoint array. `endpoints` has shape (2, E)."""
    source_type, _relation, destination_type = canonical_etype
    endpoints = np.asarray(endpoints)
    if endpoints.size == 0 or endpoints.shape[1] == 0:
        return np.zeros(0, dtype=np.float64)
    head = embeddings[source_type][endpoints[0]]
    tail = embeddings[destination_type][endpoints[1]]
    relation = np.asarray(relation_vectors[relation_index], dtype=np.float64)
    return distmult(head, relation, tail)


def pretrain_probability(scores):
    scores = np.asarray(scores, dtype=np.float64)
    return 1.0 / (1.0 + np.exp(-scores))
