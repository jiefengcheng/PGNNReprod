"""Float32 forward and analytical gradients for the pinned non-attention formulas.

This is a readiness proof for one relation and one optimizer step. It is not a
training loop and it does not replace the float64 reference in model.py.
"""
import numpy as np
from .model import LEAKY_RELU_SLOPE, leaky_relu


def _as(values, dtype):
    return np.asarray(values, dtype=dtype)


def rgcn_layer_dtype(features, edges, canonical_etypes, weights, biases, dtype):
    """Same aggregation as model.rgcn_layer, stored in `dtype`."""
    out_width = int(next(iter(biases.values())).shape[0])
    accumulated = {
        node_type: np.zeros((values.shape[0], out_width), dtype=dtype)
        for node_type, values in features.items()
    }
    for source_type, relation, destination_type in canonical_etypes:
        endpoints = np.asarray(edges[(source_type, relation, destination_type)])
        if endpoints.size == 0 or endpoints.shape[1] == 0:
            continue
        weight = _as(weights[relation], dtype)
        bias = _as(biases[relation], dtype)
        source = _as(features[source_type], dtype)
        messages = source @ weight.T + bias
        source_index = endpoints[0].astype(np.int64, copy=False)
        destination_index = endpoints[1].astype(np.int64, copy=False)
        summed = np.zeros((features[destination_type].shape[0], out_width), dtype=dtype)
        counts = np.zeros(features[destination_type].shape[0], dtype=dtype)
        np.add.at(summed, destination_index, messages[source_index])
        np.add.at(counts, destination_index, _as(1.0, dtype))
        present = counts > 0
        summed[present] /= counts[present, None]
        accumulated[destination_type] = accumulated[destination_type] + summed
    return accumulated


def encode_dtype(features, edges, canonical_etypes, layer_parameters, dtype,
                 negative_slope=LEAKY_RELU_SLOPE):
    hidden = rgcn_layer_dtype(
        features, edges, canonical_etypes,
        layer_parameters[0]["weights"], layer_parameters[0]["biases"], dtype)
    hidden = {
        node_type: leaky_relu(values, negative_slope).astype(dtype, copy=False)
        for node_type, values in hidden.items()
    }
    return rgcn_layer_dtype(
        hidden, edges, canonical_etypes,
        layer_parameters[1]["weights"], layer_parameters[1]["biases"], dtype)


def distmult_dtype(head, relation, tail, dtype):
    head = _as(head, dtype)
    relation = _as(relation, dtype)
    tail = _as(tail, dtype)
    return np.sum(head * relation * tail, axis=-1)


def message_mean_gradients(source_features, endpoints, weight, destination_gradient):
    """Gradients of a mean aggregator. `weight` has shape (out, in)."""
    source_features = _as(source_features, np.float64)
    weight = _as(weight, np.float64)
    destination_gradient = _as(destination_gradient, np.float64)
    source_index = np.asarray(endpoints[0], dtype=np.int64)
    destination_index = np.asarray(endpoints[1], dtype=np.int64)
    counts = np.bincount(destination_index, minlength=destination_gradient.shape[0]).astype(np.float64)
    incoming = destination_gradient[destination_index] / counts[destination_index, None]
    gradient_weight = incoming.T @ source_features[source_index]
    gradient_bias = incoming.sum(axis=0)
    gradient_source = np.zeros_like(source_features)
    np.add.at(gradient_source, source_index, incoming @ weight)
    return gradient_weight, gradient_bias, gradient_source


def one_relation_loss(source_features, endpoints, weight, bias, relation_vector, scored_edge, target):
    """Squared error of one DistMult score after a single mean layer.

    `scored_edge` is (source_node, destination_node). Both nodes are the same
    type, and every training edge uses that type on both ends.
    """
    features = {"node": _as(source_features, np.float64)}
    edges = {("node", "rel", "node"): np.asarray(endpoints)}
    hidden = rgcn_layer_dtype(
        features, edges, [("node", "rel", "node")],
        {"rel": weight}, {"rel": bias}, np.float64)["node"]
    head = hidden[int(scored_edge[0])]
    tail = hidden[int(scored_edge[1])]
    score = float(distmult_dtype(head, relation_vector, tail, np.float64))
    residual = score - float(target)
    loss = 0.5 * residual * residual
    relation = _as(relation_vector, np.float64)
    gradient_head = residual * relation * tail
    gradient_tail = residual * relation * head
    gradient_relation = residual * head * tail
    gradient_hidden = np.zeros_like(hidden)
    gradient_hidden[int(scored_edge[0])] += gradient_head
    gradient_hidden[int(scored_edge[1])] += gradient_tail
    gradient_weight, gradient_bias, gradient_source = message_mean_gradients(
        source_features, endpoints, weight, gradient_hidden)
    return {
        "loss": loss,
        "score": score,
        "hidden": hidden,
        "weight": gradient_weight,
        "bias": gradient_bias,
        "source": gradient_source,
        "relation": gradient_relation,
    }


def sgd_step(value, gradient, learning_rate):
    return _as(value, np.float64) - float(learning_rate) * _as(gradient, np.float64)
