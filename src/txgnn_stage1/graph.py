"""Owned, backend-neutral graph assembly from already prepared TxGNN tables.

This does NOT harmonize raw IDs, group diseases, or create experimental folds.
IDs must already be normalized strings; local indices may be integral floats.
The manifest supplies ordered node_types, ordered training canonical_etypes,
optional validation_canonical_etypes for the complete schema, and a
reverse_relations mapping from reverse relation name to forward relation name.
Memory use includes pandas copies and joins; benchmark before a full-scale run.
"""
from dataclasses import dataclass
import hashlib
import json
from itertools import combinations
import numpy as np
import pandas as pd

COLS = ["x_type", "x_id", "x_idx", "relation", "y_type", "y_id", "y_idx"]
FACT = ["x_type", "x_id", "relation", "y_type", "y_id"]
TRIPLE = ["x_type", "relation", "y_type"]


def _table(frame, label):
    missing = set(COLS) - set(frame.columns)
    if missing:
        raise ValueError(f"{label}: missing columns {sorted(missing)}")
    out = frame[COLS].copy()
    for col in FACT:
        if not out[col].map(lambda v: isinstance(v, str) and bool(v)).all():
            raise ValueError(f"{label}.{col}: nonempty strings required")
    for col in ["x_idx", "y_idx"]:
        values = out[col].to_numpy()
        if (not pd.api.types.is_numeric_dtype(out[col].dtype)
                or pd.api.types.is_bool_dtype(out[col].dtype)):
            raise ValueError(f"{label}.{col}: numeric integer indices required")
        if (not np.isfinite(values).all() or (values < 0).any()
                or (values >= 2**63).any()
                or (values != np.floor(values)).any()):
            raise ValueError(f"{label}.{col}: invalid local index")
        out[col] = values.astype(np.int64)
    return out


def _registry(full, node_types):
    pieces = []
    for side in ["x", "y"]:
        part = full[[f"{side}_type", f"{side}_id", f"{side}_idx"]].copy()
        part.columns = ["type", "id", "idx"]
        pieces.append(part)
    nodes = pd.concat(pieces, ignore_index=True).drop_duplicates()
    if set(nodes.type) != set(node_types):
        raise ValueError("Manifest node types must equal the full-table inventory")
    registry = {}
    for ntype in node_types:
        part = nodes[nodes.type == ntype].sort_values("idx")
        if part.id.duplicated().any() or part.idx.duplicated().any():
            raise ValueError(f"{ntype}: ID/index mapping is not a bijection")
        if not np.array_equal(part.idx.to_numpy(), np.arange(len(part))):
            raise ValueError(f"{ntype}: local indices are not contiguous from zero")
        registry[ntype] = tuple(part.id)
    return registry


def _canonical_facts(frame, reverse_relations):
    out = frame[FACT].copy()
    rev = out.relation.isin(reverse_relations)
    for left, right in [("x_type", "y_type"), ("x_id", "y_id")]:
        out.loc[rev, [left, right]] = out.loc[rev, [right, left]].to_numpy()
    out.loc[rev, "relation"] = out.loc[rev, "relation"].map(reverse_relations)
    swap = (out.x_type == out.y_type) & (out.x_id > out.y_id)
    out.loc[swap, ["x_id", "y_id"]] = out.loc[swap, ["y_id", "x_id"]].to_numpy()
    return out.drop_duplicates()


def assert_no_leakage(splits, reverse_relations, full_processed=None):
    """Reject shared typed facts, including their reverse equivalents.

    Same-type relations are treated as undirected, matching this TxGNN target.
    When full_processed is supplied, also reject facts absent from that source.
    This does not certify the intended disease-split membership or proportions.
    """
    facts = {name: _canonical_facts(frame, reverse_relations)
             for name, frame in splits.items()}
    if full_processed is not None:
        source = _canonical_facts(full_processed, reverse_relations)
        for name, frame in facts.items():
            checked = frame.merge(source, on=FACT, how="left", indicator=True)
            missing = checked[checked["_merge"] == "left_only"]
            if len(missing):
                example = missing.iloc[0][FACT].to_dict()
                raise ValueError(f"{name}: fact absent from full source: {example}")
    for a, b in combinations(facts, 2):
        overlap = facts[a].merge(facts[b], on=FACT, how="inner")
        if len(overlap):
            example = overlap.iloc[0].to_dict()
            raise ValueError(f"Leakage between {a} and {b}: {example}")


@dataclass
class OwnedGraph:
    node_types: tuple
    canonical_etypes: tuple
    node_ids: dict
    edges: dict

    @property
    def num_nodes(self):
        return {kind: len(ids) for kind, ids in self.node_ids.items()}

    def logical_hash(self, *, ordered=True):
        """SHA-256 of schema, node identities, and int64 COO edge content.

        ordered=False ignores within-relation edge order only. It retains
        duplicate multiplicity, node row order, and relation parameter order.
        """
        digest = hashlib.sha256()

        def record(value):
            payload = json.dumps(value, ensure_ascii=False,
                                 separators=(",", ":")).encode("utf-8")
            digest.update(len(payload).to_bytes(8, "little"))
            digest.update(payload)

        record(["owned-graph-v1", self.node_types, self.canonical_etypes])
        for ntype in self.node_types:
            record([ntype, self.node_ids[ntype]])
        for etype in self.canonical_etypes:
            pairs = self.edges[etype].T
            if not ordered and len(pairs):
                pairs = pairs[np.lexsort((pairs[:, 1], pairs[:, 0]))]
            record([etype, len(pairs)])
            digest.update(np.ascontiguousarray(pairs, dtype="<i8").tobytes())
        return digest.hexdigest()


def build_graph(full_processed, splits, manifest):
    """Build training COO arrays; validate held-out tables without adding edges.

    splits must contain train/valid/test. Empty frames need the same columns
    with numeric index dtypes. Validation-only relations do not create training
    arrays. Pin canonical_etypes to the actual reference training schema.
    """
    if set(splits) != {"train", "valid", "test"}:
        raise ValueError("Exactly train, valid, and test tables are required")
    node_types = tuple(manifest["node_types"])
    etypes = tuple(tuple(e) for e in manifest["canonical_etypes"])
    validation_etypes = tuple(tuple(e) for e in manifest.get(
        "validation_canonical_etypes", etypes))
    reverse = dict(manifest["reverse_relations"])
    if (len(set(node_types)) != len(node_types)
            or not all(isinstance(t, str) and t for t in node_types)
            or len(set(etypes)) != len(etypes)
            or len(set(validation_etypes)) != len(validation_etypes)):
        raise ValueError("Manifest type orders contain duplicates or invalid names")
    if not set(etypes) <= set(validation_etypes):
        raise ValueError("Training schema is absent from the complete validation schema")
    relation_types = {}
    for etype in validation_etypes:
        if (len(etype) != 3 or not all(isinstance(t, str) and t for t in etype)
                or etype[0] not in node_types or etype[2] not in node_types):
            raise ValueError(f"Invalid canonical edge type {etype}")
        src, relation, dst = etype
        if relation in relation_types:
            raise ValueError(f"Ambiguous relation name: {relation}")
        relation_types[relation] = (src, dst)
    for backward, forward in reverse.items():
        if (backward == forward or forward in reverse
                or backward not in relation_types or forward not in relation_types
                or relation_types[backward] != relation_types[forward][::-1]):
            raise ValueError(f"Invalid reverse relation mapping: {backward}")
    if any(r.startswith("rev_") and r not in reverse for r in relation_types):
        raise ValueError("All rev_ relation names require explicit reverse mappings")
    full = _table(full_processed, "full")
    prepared = {name: _table(frame, name) for name, frame in splits.items()}
    registry = _registry(full, node_types)
    lookup = {kind: {external: i for i, external in enumerate(ids)}
              for kind, ids in registry.items()}
    for name, frame in {"full": full, **prepared}.items():
        allowed = set(etypes if name == "train" else validation_etypes)
        observed = set(frame[TRIPLE].itertuples(index=False, name=None))
        if not observed <= allowed:
            raise ValueError(f"{name}: unknown canonical edge types {observed - allowed}")
        for side in ["x", "y"]:
            for ntype, part in frame.groupby(f"{side}_type", sort=False):
                expected = part[f"{side}_id"].map(lookup[ntype])
                if expected.isna().any() or not np.array_equal(
                        expected.to_numpy(), part[f"{side}_idx"].to_numpy()):
                    raise ValueError(f"{name}: invalid {side} endpoint identity/index")
    assert_no_leakage(prepared, reverse, full_processed=full)
    train = prepared["train"]
    edges = {}
    for src, relation, dst in etypes:
        mask = ((train.x_type == src) & (train.relation == relation)
                & (train.y_type == dst))
        coo = train.loc[mask, ["x_idx", "y_idx"]].to_numpy(dtype=np.int64).T.copy()
        coo.setflags(write=False)
        edges[(src, relation, dst)] = coo
    return OwnedGraph(node_types, etypes, registry, edges)
