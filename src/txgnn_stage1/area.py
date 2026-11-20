"""Disease-area split for autoimmune. No TxGNN, DGL, or PyG imports.

Protocol id: autoimmune_owned_v1.

Static-spec behavior, read from TxGNN commit 0e0203c and not executed:

- The area root is DOID 417. Descendants come from is_a links in HumanDO.obo.
- MONDO crosswalk rows with ontology DOID, plus grouped-disease ids, become global indices.
- The default neighborhood is two directed hops on edges.csv, then the induced edges.
- Drug-disease edges touching the selected nodes are part of the mask.
- The mask budget is round(0.05 * edge_count), and the random draw fills the remainder.
- Endpoint-only de-duplication keeps one row per (x_index, y_index), preferring a masked row.
- Non-masked rows are then split per relation into train and valid. The static fraction is 0.125 valid.
- Reverse rows stay inside the split. The evaluation view drops reverse therapeutic rows whose
  disease endpoint is outside the released autoimmune disease list.

Owned choices:

- DOID tokens are normalized with str(int(token)) so leading zeros do not break is_a links.
  The static parser keeps zeros on ids and strips them on some is_a lines.
- The neighborhood draw uses numpy.random.Generator(PCG64(preprocess_seed)). The static code
  draws with an unseeded np.random.choice. This mask is therefore not a historical mask.
- Validation uses PCG64(fold_seed). The static random_fold call used random_state=1 and a
  zero test fraction.
- After orientation, a surviving row is test when either directed global-index pair was masked.
  Orientation would otherwise be able to keep the opposite, unmasked direction in train.
- Duplicate directed facts that remain after de-duplication stay in one split.
"""
from pathlib import Path
import numpy as np
import pandas as pd
from .preprocess import (
    FACT_COLUMNS, THERAPEUTIC, add_reverse_edges, assign_local_indices, convert2str,
    crosswalk_audit, disease_membership, representative_edges, schema_for_splits,
)

PROTOCOL = "autoimmune_owned_v1"
AREA_ROOT = "417"
VALID_FRACTION = 0.125
REVERSE_THERAPEUTIC = tuple("rev_" + name for name in THERAPEUTIC)


def _doid_token(text):
    token = str(text).strip()
    if token.startswith("DOID:"):
        token = token.split(":", 1)[1]
    try:
        return str(int(token))
    except ValueError:
        return token


def obo_children(path):
    """Map each normalized DOID to the set of normalized direct children."""
    children = {}
    current = None
    parents = []

    def flush():
        nonlocal current, parents
        if current is not None:
            for parent in parents:
                children.setdefault(parent, set()).add(current)
        current = None
        parents = []

    with Path(path).open("r", encoding="utf-8", errors="replace") as stream:
        for line in stream:
            line = line.strip()
            if line == "[Term]":
                flush()
            elif line.startswith("id: DOID:"):
                current = _doid_token(line[4:].split()[0])
            elif line.startswith("is_a: DOID:"):
                parents.append(_doid_token(line[6:].split()[0]))
        flush()
    return children


def descendant_ids(children, root):
    seen = {str(root)}
    stack = [str(root)]
    while stack:
        node = stack.pop()
        for child in children.get(node, ()):
            if child not in seen:
                seen.add(child)
                stack.append(child)
    return seen


def disease_global_indices(node_file, mondo_xref, group_map, doid_ids):
    """Released global indices for MONDO concepts and their grouped stand-ins."""
    xref = mondo_xref[mondo_xref.ontology.astype(str) == "DOID"].copy()
    xref["ontology_id"] = xref.ontology_id.map(_doid_token)
    mondo = set(xref.loc[xref.ontology_id.isin(doid_ids), "mondo_id"].map(convert2str))
    nodes = node_file.copy()
    nodes["reference_id"] = nodes.node_id.map(convert2str)
    direct = nodes[(nodes.node_source.astype(str) == "MONDO") & nodes.reference_id.isin(mondo)]
    if group_map is None:
        grouped_ids = set()
    else:
        groups = group_map.copy()
        groups["reference_id"] = groups.node_id.map(convert2str)
        groups = groups[groups.reference_id.isin(mondo)]
        if "node_source" in groups.columns:
            groups = groups[groups.node_source.astype(str) == "MONDO"]
        grouped_ids = set(groups.group_id_bert.map(convert2str))
    grouped = nodes[(nodes.node_source.astype(str) == "MONDO_grouped") & nodes.reference_id.isin(grouped_ids)]
    indices = np.unique(np.concatenate([
        direct.node_index.to_numpy(dtype=np.int64),
        grouped.node_index.to_numpy(dtype=np.int64),
    ])) if len(direct) or len(grouped) else np.array([], dtype=np.int64)
    return indices, {"mondo_ids": len(mondo), "direct_nodes": int(len(direct)),
                     "grouped_ids": len(grouped_ids), "grouped_nodes": int(len(grouped)),
                     "global_indices": int(len(indices))}


def _pack(left, right):
    return left.astype(np.int64) << 32 | right.astype(np.int64)


def two_hop_induced_mask(edge_x, edge_y, seeds, hops):
    """Directed k-hop node set, then edges with both ends inside that set."""
    if len(edge_x) == 0 or len(seeds) == 0:
        return np.zeros(len(edge_x), dtype=bool), 0
    n_nodes = int(max(edge_x.max(), edge_y.max(), seeds.max())) + 1
    seen = np.zeros(n_nodes, dtype=bool)
    seen[seeds] = True
    frontier = seen.copy()
    for _ in range(int(hops)):
        outgoing = frontier[edge_x]
        if not outgoing.any():
            break
        reached = np.unique(edge_y[outgoing])
        nxt = np.zeros(n_nodes, dtype=bool)
        nxt[reached] = True
        nxt &= ~seen
        if not nxt.any():
            break
        seen |= nxt
        frontier = nxt
    induced = seen[edge_x] & seen[edge_y]
    return induced, int(seen.sum())


def area_edge_pairs(edges, seeds, hops, test_size, preprocess_seed):
    """Return unique masked (x_index, y_index) pairs and an audit without row text."""
    x = edges.x_index.to_numpy(dtype=np.int64)
    y = edges.y_index.to_numpy(dtype=np.int64)
    relation = edges.relation.astype(str).to_numpy()
    induced, reached = two_hop_induced_mask(x, y, seeds, hops)
    therapeutic = np.isin(relation, list(THERAPEUTIC))
    incident = np.isin(x, seeds) | np.isin(y, seeds)
    drug = therapeutic & incident
    budget = int(round(float(test_size) * len(edges)))
    n_drug = int(drug.sum())
    n_random = budget - n_drug
    if n_random < 0:
        raise ValueError("Area budget {} is smaller than {} therapeutic edges".format(budget, n_drug))
    candidates = np.flatnonzero(induced)
    if n_random > len(candidates):
        raise ValueError("Induced neighborhood has {} edges, need {}".format(len(candidates), n_random))
    draw = np.random.Generator(np.random.PCG64(int(preprocess_seed)))
    chosen = candidates[draw.permutation(len(candidates))[:n_random]] if n_random else np.array([], dtype=np.int64)
    selected = np.unique(np.concatenate([np.flatnonzero(drug), chosen]))
    pairs = np.stack([x[selected], y[selected]])
    order = np.lexsort((pairs[1], pairs[0]))
    pairs = pairs[:, order]
    if pairs.shape[1]:
        changed = np.any(np.diff(pairs, axis=1), axis=0)
        keep = np.empty(pairs.shape[1], dtype=bool)
        keep[0] = True
        keep[1:] = changed
        pairs = pairs[:, keep]
    audit = {"hops": int(hops), "test_size": float(test_size), "edge_rows": int(len(edges)),
             "budget": budget, "therapeutic_edges": n_drug, "random_draw": int(n_random),
             "nodes_reached": reached, "induced_edges": int(induced.sum()),
             "mask_pairs": int(pairs.shape[1]), "preprocess_seed": int(preprocess_seed)}
    return pairs, audit


def _directed_in_mask(x_index, y_index, packed_mask):
    return np.isin(_pack(x_index, y_index), packed_mask)


def _either_direction_in_mask(x_index, y_index, packed_mask):
    return _directed_in_mask(x_index, y_index, packed_mask) | _directed_in_mask(
        y_index, x_index, packed_mask)


def apply_endpoint_dedup(kg, pairs):
    """Keep one row per global endpoint pair, preferring a masked direction."""
    work = kg.reset_index(drop=True).copy()
    work["source_row"] = np.arange(len(work), dtype=np.int64)
    packed = np.unique(_pack(pairs[0], pairs[1])) if pairs.shape[1] else np.array([], dtype=np.int64)
    masked = _directed_in_mask(work.x_index.to_numpy(), work.y_index.to_numpy(), packed)
    work["rank"] = work.source_row.to_numpy() + masked.astype(np.int64) * len(work)
    work = work.sort_values("rank")
    before = len(work)
    kept = work.drop_duplicates(["x_index", "y_index"], keep="last").sort_values("source_row")
    kept = kept.drop(columns=["rank"]).reset_index(drop=True)
    audit = {"rows_before": before, "rows_after": int(len(kept)),
             "endpoint_pairs_removed": int(before - len(kept))}
    return kept, packed, audit


def assign_area_splits(indexed, packed_mask, fold_seed):
    """Test if either direction was masked; split the rest into train and valid."""
    out = indexed.copy()
    masked = _either_direction_in_mask(out.x_index.to_numpy(), out.y_index.to_numpy(), packed_mask)
    out["split"] = np.where(masked, "test", "train")
    pieces = [out[out.split == "test"]]
    pool = out[out.split == "train"]
    for _relation, part in pool.groupby("relation", sort=False):
        facts = part.drop_duplicates(FACT_COLUMNS).sort_values(FACT_COLUMNS + ["source_row"]).copy()
        n_valid = int(round(VALID_FRACTION * len(facts)))
        labels = np.full(len(facts), "train", dtype=object)
        if n_valid:
            draw = np.random.Generator(np.random.PCG64(int(fold_seed)))
            labels[draw.permutation(len(facts))[:n_valid]] = "valid"
        facts["split"] = labels
        labeled = part.drop(columns=["split"]).merge(
            facts[FACT_COLUMNS + ["split"]], on=FACT_COLUMNS, how="left")
        if labeled["split"].isna().any():
            raise ValueError("Area split missed rows")
        pieces.append(labeled)
    assigned = pd.concat(pieces, ignore_index=True).sort_values("source_row").reset_index(drop=True)
    splits = {name: assigned[assigned.split == name].reset_index(drop=True)
              for name in ("train", "valid", "test")}
    summary = {"base_rows": {name: int(len(frame)) for name, frame in splits.items()}}
    return splits, summary


def evaluation_view(test_frame, nodes, disease_ids):
    """Drop reverse therapeutic test rows whose disease is outside the released list."""
    if test_frame.empty:
        return test_frame.copy(), {"dropped_reverse_therapeutic": 0, "kept": 0}
    lookup = {}
    diseases = nodes[nodes.node_type == "disease"]
    for reference_id, local_idx in diseases[["reference_id", "local_idx"]].itertuples(index=False, name=None):
        lookup[str(reference_id)] = int(local_idx)
        if "_" in str(reference_id):
            for piece in str(reference_id).split("_"):
                try:
                    lookup[str(float(piece))] = int(local_idx)
                except ValueError:
                    lookup[str(piece)] = int(local_idx)
    allowed = set()
    for raw in disease_ids:
        key = convert2str(raw)
        if key in lookup:
            allowed.add(lookup[key])
    reverse = test_frame.relation.isin(REVERSE_THERAPEUTIC)
    outside = ~test_frame.x_idx.isin(allowed)
    drop = reverse & outside
    kept = test_frame.loc[~drop].reset_index(drop=True)
    return kept, {"dropped_reverse_therapeutic": int(drop.sum()), "kept": int(len(kept)),
                  "disease_local_ids": len(allowed)}


def prepare_autoimmune(kg, edges, node_file, mondo_xref, group_map, obo_path, disease_ids,
                       fold_seed=42, preprocess_seed=42, hops=2, test_size=0.05):
    children = obo_children(obo_path)
    doids = descendant_ids(children, AREA_ROOT)
    seeds, seed_audit = disease_global_indices(node_file, mondo_xref, group_map, doids)
    pairs, mask_audit = area_edge_pairs(edges, seeds, hops, test_size, preprocess_seed)
    deduped, packed, dedup_audit = apply_endpoint_dedup(kg, pairs)
    base, orientation = representative_edges(deduped)
    indexed, nodes, crosswalk, collisions = assign_local_indices(base)
    splits, split_summary = assign_area_splits(indexed, packed, fold_seed)
    augmented, reverse_map, _triples = add_reverse_edges(indexed, splits)
    schema = schema_for_splits(sorted(nodes.node_type.unique()), augmented, reverse_map)
    membership, group_audit = disease_membership(nodes, group_map)
    evaluation, evaluation_audit = evaluation_view(augmented["test"], nodes, disease_ids)
    checks = {
        "protocol": PROTOCOL, "doid_root": AREA_ROOT, "doid_closure": len(doids),
        "seeds": seed_audit, "mask": mask_audit, "endpoint_dedup": dedup_audit,
        "orientation": orientation, "normalization_collision_groups": len(collisions),
        "normalization_collisions": collisions, "split": split_summary,
        "group_map": group_audit, "evaluation": evaluation_audit,
        "base_rows": int(len(indexed)),
        "augmented_rows": {name: int(len(frame)) for name, frame in augmented.items()},
        "preprocess_seed_consumed": True, "crosswalk": crosswalk_audit(crosswalk),
    }
    return {"base": indexed, "nodes": nodes, "crosswalk": crosswalk, "membership": membership,
            "splits": augmented, "schema": schema, "checks": checks, "mask_pairs": pairs,
            "evaluation": evaluation}
