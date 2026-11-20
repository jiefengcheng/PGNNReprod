"""Independent raw-KG preparation. No TxGNN, DGL, or PyG imports.

Protocol id: complex_disease_owned_v1.

Static-spec choices, read from TxGNN commit 0e0203c3939f7d12da27aaf10e55daaff5424023
and not executed from that package:

- convert2str: a token with no underscore that parses as float is replaced by str(float(token)).
- Homogeneous relation names are those where the first two underscore-separated pieces match.
  Duplicate unordered endpoint pairs collapse to the first source row. The key uses the raw
  id text, before convert2str.
- Every other relation keeps the source type of its first row and drops the opposite orientation.
- Type-local indices are 0..N-1 over lexicographically sorted normalized ids.
- complex_disease fractions are 0.83125 / 0.11875 / 0.05. Therapeutic relations are
  contraindication, indication, and off-label use, and the disease end is y after orientation.
- Reverse rows are added only after the split. Heterogeneous reverses are named rev_<relation>.
  Homogeneous reverses swap endpoints and keep the relation name, so a self-loop is duplicated.

Owned choices, which are not a claim about historical random bits:

- The split and the input matrices use numpy.random.Generator(PCG64).
- preprocess_seed is recorded and is not consumed. Representative selection is deterministic.
- Duplicate directed facts stay together. Multiplicity is kept, but copies cannot fall into
  different splits. A row-wise draw can separate those copies; this protocol does not.
- Node-type order is alphabetical. Training relation order is first-seen order in the
  reverse-augmented training table.
- edges.csv is locked with the other assets and is not read for this split. The static
  complex_disease path does not use it either.
"""
from pathlib import Path
import json
import numpy as np
import pandas as pd

PROTOCOL = "complex_disease_owned_v1"
THERAPEUTIC = ("contraindication", "indication", "off-label use")
FRACTIONS = (0.83125, 0.11875, 0.05)
FACT_COLUMNS = ["x_type", "x_id", "relation", "y_type", "y_id"]


def convert2str(value):
    text = str(value)
    if "_" not in text:
        try:
            text = str(float(text))
        except (TypeError, ValueError):
            pass
    return text


def is_homogeneous_relation_name(name):
    parts = str(name).split("_")
    return len(parts) >= 2 and parts[0] == parts[1]


def _unordered_pair_key(left, right):
    left = left.astype(str)
    right = right.astype(str)
    swap = left.to_numpy() > right.to_numpy()
    first = left.where(~swap, right)
    second = right.where(~swap, left)
    return first.str.cat(second, sep="_")


def representative_edges(frame):
    """Return source-ordered representative rows and a small rule audit."""
    required = ["x_type", "x_id", "relation", "y_type", "y_id"]
    missing = [column for column in required if column not in frame.columns]
    if missing:
        raise ValueError("KG frame missing columns {}".format(missing))
    had_source = "source_row" in frame.columns
    work = frame.reset_index(drop=True).copy()
    if not had_source:
        work["source_row"] = np.arange(len(work), dtype=np.int64)
    chosen = []
    rules = []
    for relation in np.unique(work.relation.to_numpy()):
        part = work[work.relation == relation]
        if is_homogeneous_relation_name(relation):
            key = _unordered_pair_key(part.x_id, part.y_id)
            keep = part.loc[~key.duplicated()]
            rules.append({"relation": str(relation), "rule": "homogeneous_unordered_first",
                          "input_rows": int(len(part)), "kept_rows": int(len(keep))})
        else:
            anchor = part.x_type.iloc[0]
            keep = part[part.x_type == anchor]
            rules.append({"relation": str(relation), "rule": "first_source_type",
                          "anchor_type": str(anchor), "input_rows": int(len(part)),
                          "kept_rows": int(len(keep))})
        chosen.append(keep)
    if not chosen:
        empty = work.iloc[0:0].copy()
        return empty, rules
    kept = pd.concat(chosen, ignore_index=True).sort_values("source_row")
    return kept.reset_index(drop=True), rules


def _optional_column(frame, side, suffix, default):
    name = "{}_{}".format(side, suffix)
    if name in frame.columns:
        return frame[name]
    return pd.Series(default, index=frame.index)


def _endpoint_records(frame):
    pieces = []
    for side in ("x", "y"):
        piece = pd.DataFrame({
            "node_type": frame["{}_type".format(side)].to_numpy(),
            "raw_id": frame["{}_raw_id".format(side)].to_numpy(),
            "reference_id": frame["{}_id".format(side)].to_numpy(),
            "name": _optional_column(frame, side, "name", "").astype(str).to_numpy(),
            "source": _optional_column(frame, side, "source", "").astype(str).to_numpy(),
            "global_index": _optional_column(frame, side, "index", -1).to_numpy(),
            "source_row": frame.source_row.to_numpy(),
        })
        pieces.append(piece)
    return pd.concat(pieces, ignore_index=True)


def assign_local_indices(frame):
    """Normalize ids, audit collisions, and assign contiguous per-type indices."""
    out = frame.copy()
    out["x_raw_id"] = out.x_id.map(lambda value: str(value))
    out["y_raw_id"] = out.y_id.map(lambda value: str(value))
    out["x_id"] = out.x_id.map(convert2str)
    out["y_id"] = out.y_id.map(convert2str)
    endpoints = _endpoint_records(out)
    collisions = []
    nodes = []
    lookup = {}
    for node_type, part in endpoints.groupby("node_type", sort=True):
        grouped = part.groupby("reference_id", sort=False)
        raw_sets = grouped.raw_id.agg(lambda values: tuple(sorted(set(values))))
        for reference_id, raw_ids in raw_sets.items():
            if len(raw_ids) > 1:
                collisions.append({"node_type": str(node_type), "reference_id": str(reference_id),
                                   "raw_ids": list(raw_ids)})
        order = sorted(raw_sets.index.astype(str))
        type_lookup = {reference_id: index for index, reference_id in enumerate(order)}
        lookup[str(node_type)] = type_lookup
        first = part.sort_values("source_row").drop_duplicates("reference_id")
        first = first.set_index("reference_id")
        for reference_id, local_idx in type_lookup.items():
            row = first.loc[reference_id]
            nodes.append({
                "node_type": str(node_type),
                "local_idx": int(local_idx),
                "reference_id": reference_id,
                "raw_id": str(row.raw_id),
                "name": str(row.name),
                "source": str(row.source),
            })
    out["x_idx"] = [lookup[str(t)][i] for t, i in zip(out.x_type, out.x_id)]
    out["y_idx"] = [lookup[str(t)][i] for t, i in zip(out.y_type, out.y_id)]
    out["x_idx"] = out.x_idx.astype(np.int64)
    out["y_idx"] = out.y_idx.astype(np.int64)
    node_table = pd.DataFrame(nodes).sort_values(["node_type", "local_idx"]).reset_index(drop=True)
    crosswalk = _global_crosswalk(endpoints, lookup)
    return out, node_table, crosswalk, collisions


def _global_crosswalk(endpoints, lookup):
    observed = endpoints[pd.to_numeric(endpoints.global_index, errors="coerce").notna()].copy()
    observed["global_index"] = pd.to_numeric(observed.global_index, errors="coerce")
    observed = observed[observed.global_index >= 0]
    if observed.empty:
        return pd.DataFrame(columns=["global_index", "node_type", "local_idx", "reference_id"])
    observed["global_index"] = observed.global_index.astype(np.int64)
    observed["local_idx"] = [lookup[str(t)][i] for t, i in zip(observed.node_type, observed.reference_id)]
    reduced = observed[["global_index", "node_type", "local_idx", "reference_id"]].drop_duplicates()
    return reduced.sort_values(["global_index", "node_type", "local_idx"]).reset_index(drop=True)


def partition_labels(n_items, fold_seed, validation_seed=1):
    """Owned PCG64 partition. Sizes use round() on the static fractions."""
    labels = np.full(n_items, "train", dtype=object)
    if n_items == 0:
        return labels
    n_test = int(round(FRACTIONS[2] * n_items))
    n_test = min(n_items, max(0, n_test))
    test_draw = np.random.Generator(np.random.PCG64(int(fold_seed)))
    order = test_draw.permutation(n_items)
    test_at = order[:n_test]
    remain = order[n_test:]
    n_valid = int(round((FRACTIONS[1] / (1.0 - FRACTIONS[2])) * len(remain)))
    n_valid = min(len(remain), max(0, n_valid))
    valid_draw = np.random.Generator(np.random.PCG64(int(validation_seed)))
    valid_at = remain[valid_draw.permutation(len(remain))[:n_valid]]
    labels[test_at] = "test"
    labels[valid_at] = "valid"
    return labels


def split_complex_disease(base, fold_seed):
    """Split base facts before any reverse row exists."""
    if not (base.loc[base.relation.isin(THERAPEUTIC), "y_type"] == "disease").all():
        raise ValueError("Therapeutic edges must be oriented toward disease before splitting")
    therapeutic = base[base.relation.isin(THERAPEUTIC)].copy()
    other = base[~base.relation.isin(THERAPEUTIC)].copy()
    diseases = np.unique(therapeutic.y_idx.to_numpy()) if len(therapeutic) else np.array([], dtype=np.int64)
    draw = np.random.Generator(np.random.PCG64(int(fold_seed)))
    diseases = diseases[draw.permutation(len(diseases))]
    cut_train = int(FRACTIONS[0] * len(diseases))
    cut_valid = int((FRACTIONS[0] + FRACTIONS[1]) * len(diseases))
    membership = {}
    for label, values in (("train", diseases[:cut_train]), ("valid", diseases[cut_train:cut_valid]),
                          ("test", diseases[cut_valid:])):
        for value in values.tolist():
            membership[int(value)] = label
    if len(therapeutic):
        therapeutic["split"] = therapeutic.y_idx.map(lambda value: membership[int(value)])
    pieces = [therapeutic] if len(therapeutic) else []
    for relation, part in other.groupby("relation", sort=False):
        keys = FACT_COLUMNS
        facts = part.drop_duplicates(keys).sort_values(keys + ["source_row"])
        facts = facts.copy()
        facts["split"] = partition_labels(len(facts), fold_seed)
        labeled = part.merge(facts[keys + ["split"]], on=keys, how="left")
        if labeled["split"].isna().any():
            raise ValueError("Split assignment missed rows for {}".format(relation))
        pieces.append(labeled)
    if not pieces:
        assigned = base.iloc[0:0].copy()
        assigned["split"] = []
    else:
        assigned = pd.concat(pieces, ignore_index=True).sort_values("source_row")
    assigned = assigned.reset_index(drop=True)
    splits = {name: assigned[assigned.split == name].reset_index(drop=True)
              for name in ("train", "valid", "test")}
    summary = {"diseases_with_therapeutic_edges": int(len(diseases)),
               "disease_split_counts": {name: int(sum(1 for label in membership.values() if label == name))
                                        for name in ("train", "valid", "test")},
               "base_rows": {name: int(len(frame)) for name, frame in splits.items()}}
    return splits, summary


def _swap_endpoints(frame):
    swapped = frame.copy()
    pairs = [("x_type", "y_type"), ("x_id", "y_id"), ("x_idx", "y_idx"), ("x_raw_id", "y_raw_id")]
    for left, right in pairs:
        if left in swapped.columns:
            swapped[[left, right]] = swapped[[right, left]]
    for suffix in ("name", "source", "index"):
        left, right = "x_" + suffix, "y_" + suffix
        if left in swapped.columns and right in swapped.columns:
            swapped[[left, right]] = swapped[[right, left]]
    return swapped


def add_reverse_edges(base, splits):
    triples = base[["x_type", "relation", "y_type"]].drop_duplicates()
    if triples.relation.duplicated().any():
        names = triples.loc[triples.relation.duplicated(keep=False), "relation"].unique().tolist()
        raise ValueError("Relation name is not a unique type pair: {}".format(names))
    reverse_map = {}
    for source_type, relation, target_type in triples.itertuples(index=False, name=None):
        if source_type != target_type:
            reverse_name = "rev_" + relation
            if reverse_name in set(triples.relation) or reverse_name in reverse_map:
                raise ValueError("Reverse name collides with an existing relation: " + reverse_name)
            reverse_map[reverse_name] = relation
    augmented = {}
    for name, frame in splits.items():
        blocks = [frame]
        for source_type, relation, target_type in triples.itertuples(index=False, name=None):
            selected = frame[frame.relation == relation]
            if selected.empty:
                continue
            reversed_rows = _swap_endpoints(selected)
            if source_type != target_type:
                reversed_rows["relation"] = "rev_" + relation
            blocks.append(reversed_rows)
        augmented[name] = pd.concat(blocks, ignore_index=True)
    return augmented, reverse_map, triples


def schema_for_splits(node_types, splits, reverse_map):
    def seen(frame):
        return [list(row) for row in frame[["x_type", "relation", "y_type"]]
                .drop_duplicates().itertuples(index=False, name=None)]
    training = seen(splits["train"])
    known = {tuple(item) for item in training}
    validation = list(training)
    for name in ("valid", "test"):
        for item in seen(splits[name]):
            key = tuple(item)
            if key not in known:
                validation.append(item)
                known.add(key)
    return {"node_types": list(node_types),
            "canonical_etypes": training,
            "validation_canonical_etypes": validation,
            "reverse_relations": dict(reverse_map)}


def disease_membership(node_table, group_map):
    diseases = node_table[node_table.node_type == "disease"][["local_idx", "reference_id", "raw_id", "source"]]
    if group_map is None:
        return diseases.assign(map_status="group_map_absent"), {"group_map": "BLOCKED", "rows": 0}
    mapped = group_map.copy()
    mapped["original_reference_id"] = mapped.node_id.map(convert2str)
    mapped["group_reference_id"] = mapped.group_id_bert.map(convert2str)
    index = dict(zip(diseases.reference_id, diseases.local_idx))
    mapped["group_local_idx"] = mapped.group_reference_id.map(index)
    resolved = int(mapped.group_local_idx.notna().sum())
    audit = {"group_map": "USED", "rows": int(len(mapped)),
             "resolved_group_ids": resolved,
             "unresolved_group_ids": int(len(mapped) - resolved),
             "distinct_original_ids": int(mapped.original_reference_id.nunique()),
             "distinct_group_ids": int(mapped.group_reference_id.nunique())}
    columns = ["node_id", "original_reference_id", "group_id_bert", "group_reference_id", "group_local_idx"]
    return mapped[columns], audit


def crosswalk_audit(crosswalk):
    if crosswalk.empty:
        return {"rows": 0, "global_index_conflicts": 0, "identity_with_multiple_global_indices": 0}
    per_index = crosswalk.groupby("global_index")[["node_type", "reference_id"]].nunique()
    conflicts = int(((per_index.node_type > 1) | (per_index.reference_id > 1)).sum())
    per_identity = crosswalk.groupby(["node_type", "reference_id"]).global_index.nunique()
    return {"rows": int(len(crosswalk)), "global_index_conflicts": conflicts,
            "identity_with_multiple_global_indices": int((per_identity > 1).sum())}


def prepare_tables(kg, group_map=None, fold_seed=42):
    """Run the owned protocol on an in-memory KG frame."""
    base, orientation = representative_edges(kg)
    indexed, nodes, crosswalk, collisions = assign_local_indices(base)
    splits, split_summary = split_complex_disease(indexed, fold_seed)
    augmented, reverse_map, _triples = add_reverse_edges(indexed, splits)
    schema = schema_for_splits(sorted(nodes.node_type.unique()), augmented, reverse_map)
    membership, group_audit = disease_membership(nodes, group_map)
    checks = {"protocol": PROTOCOL,
              "orientation": orientation,
              "normalization_collision_groups": len(collisions),
              "normalization_collisions": collisions,
              "crosswalk": crosswalk_audit(crosswalk),
              "split": split_summary,
              "group_map": group_audit,
              "node_counts": {str(k): int(v) for k, v in nodes.groupby("node_type").size().items()},
              "base_rows": int(len(indexed)),
              "augmented_rows": {name: int(len(frame)) for name, frame in augmented.items()}}
    return {"base": indexed, "nodes": nodes, "crosswalk": crosswalk, "membership": membership,
            "splits": augmented, "schema": schema, "checks": checks}


def read_released_tables(raw_dir):
    raw_dir = Path(raw_dir)
    kg_columns = ["x_index", "x_id", "x_type", "x_name", "x_source", "relation",
                  "y_index", "y_id", "y_type", "y_name", "y_source"]
    kg = pd.read_csv(raw_dir / "kg.csv", usecols=kg_columns, low_memory=False)
    nodes = pd.read_csv(raw_dir / "node.csv", sep="\t", low_memory=False)
    return kg, nodes


def read_group_map(path):
    if path is None:
        return None
    frame = pd.read_csv(path, low_memory=False)
    required = {"node_id", "group_id_bert"}
    missing = required - set(frame.columns)
    if missing:
        raise ValueError("Group map missing columns {}".format(sorted(missing)))
    return frame


def attach_node_file(nodes, node_file):
    """Fill source from the released node table when the normalized id matches."""
    if node_file is None or node_file.empty:
        return nodes, {"node_file_rows": 0, "matched_ids": 0, "source_conflicts": 0,
                       "node_file_ids_absent_from_graph": 0}
    external = node_file.copy()
    external["reference_id"] = external.node_id.map(convert2str)
    external["node_type"] = external.node_type.astype(str)
    external = external.sort_values("node_index").drop_duplicates(["node_type", "reference_id"])
    merged = nodes.merge(external[["node_type", "reference_id", "node_source", "node_index"]],
                         on=["node_type", "reference_id"], how="left")
    conflict = merged.node_source.notna() & merged.source.ne("") & merged.node_source.ne(merged.source)
    merged.loc[merged.node_source.notna(), "source"] = merged.loc[merged.node_source.notna(), "node_source"]
    graph_keys = set(zip(nodes.node_type, nodes.reference_id))
    file_keys = set(zip(external.node_type, external.reference_id))
    audit = {"node_file_rows": int(len(node_file)),
             "matched_ids": int(merged.node_source.notna().sum()),
             "source_conflicts": int(conflict.sum()),
             "node_file_ids_absent_from_graph": int(len(file_keys - graph_keys))}
    return merged.drop(columns=["node_source"]), audit
