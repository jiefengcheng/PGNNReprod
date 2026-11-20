#!/usr/bin/env python3
"""Local, explicit orchestration; never stages generated outputs into git."""
import argparse
import json
from pathlib import Path
from txgnn_stage1.assets import acquire_assets
from txgnn_stage1.bundle import prepare_bundle, verify_bundle

PIN = "0e0203c3939f7d12da27aaf10e55daaff5424023"

def main():
    p = argparse.ArgumentParser(description="TxGNN Step 1 reproducible preparation")
    sub = p.add_subparsers(dest="command", required=True)
    fetch = sub.add_parser("fetch", help="Acquire raw KG into ignored data/raw")
    fetch.add_argument("--raw-dir", type=Path, default=Path("data/raw"))
    ref = sub.add_parser("reference", help="Export pinned upstream snapshot")
    ref.add_argument("--raw-dir", type=Path, default=Path("data/raw"))
    ref.add_argument("--upstream-repo", type=Path, default=Path("reference/TxGNN"))
    ref.add_argument("--work-dir", type=Path, default=Path("artifacts/local"))
    ref.add_argument("--split", default="complex_disease")
    ref.add_argument("--fold-seed", type=int, default=42)
    ref.add_argument("--preprocess-seed", type=int, default=42)
    ref.add_argument("--input-seed", type=int, default=0)
    ref.add_argument("--input-dim", type=int, default=100)
    ref.add_argument("--test-size", type=float, default=0.05)
    ref.add_argument("--mask-ratio", type=float, default=0.1)
    ref.add_argument("--one-hop", action="store_true")
    comp = sub.add_parser("compare", help="Archival: graph/input parity against a reference snapshot")
    comp.add_argument("--snapshot", required=True, type=Path)
    comp.add_argument("--out", type=Path, default=Path("artifacts/local"))
    prep = sub.add_parser("prepare", help="Independent NumPy/Pandas raw-to-bundle preparation")
    prep.add_argument("--raw-dir", type=Path, default=Path("data/raw"))
    prep.add_argument("--out", required=True, type=Path)
    prep.add_argument("--group-map", type=Path, default=None)
    prep.add_argument("--fold-seed", type=int, default=42)
    prep.add_argument("--preprocess-seed", type=int, default=42)
    prep.add_argument("--input-seed", type=int, default=0)
    prep.add_argument("--input-dim", type=int, default=100)
    prep.add_argument("--split", default="complex_disease")
    prep.add_argument("--obo", type=Path, default=None)
    prep.add_argument("--mondo-xref", type=Path, default=None)
    prep.add_argument("--disease-list", type=Path, default=None)
    prep.add_argument("--hops", type=int, default=2)
    prep.add_argument("--test-size", type=float, default=0.05)
    verify = sub.add_parser("verify", help="Reload a bundle in-process and in a fresh process")
    verify.add_argument("--bundle", required=True, type=Path)
    args = p.parse_args()
    if args.command == "fetch":
        records = acquire_assets(args.raw_dir)
        print(json.dumps({k: {"sha256": v["sha256"], "bytes": v["bytes"]} for k,v in records.items()}, indent=2))
    elif args.command == "prepare":
        path = prepare_bundle(args.raw_dir, args.out, group_map_path=args.group_map,
                              fold_seed=args.fold_seed, preprocess_seed=args.preprocess_seed,
                              input_seed=args.input_seed, input_dim=args.input_dim, split=args.split,
                              obo_path=args.obo, mondo_xref_path=args.mondo_xref,
                              disease_list_path=args.disease_list, hops=args.hops,
                              test_size=args.test_size)
        print("Owned bundle:", path)
    elif args.command == "verify":
        print(json.dumps(verify_bundle(args.bundle), indent=2))
    elif args.command == "reference":
        from txgnn_stage1.reference import capture_reference
        config = {
            "reference_sha": PIN, "split": args.split, "fold_seed": args.fold_seed,
            "preprocess_seed": args.preprocess_seed, "input_seed": args.input_seed,
            "input_dim": args.input_dim, "test_size": args.test_size,
            "mask_ratio": args.mask_ratio, "one_hop": args.one_hop, "no_kg": False,
        }
        path = capture_reference(args.raw_dir, args.upstream_repo, args.work_dir, config)
        print("Reference snapshot:", path)
    else:
        from txgnn_stage1.compare import compare_real_snapshot
        path, report = compare_real_snapshot(args.snapshot, args.out)
        print("Comparison bundle:", path)
        print(json.dumps(report, indent=2))
if __name__ == "__main__": main()
