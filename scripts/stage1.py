#!/usr/bin/env python3
"""Local, explicit orchestration; never stages generated outputs into git."""
import argparse
import json
import subprocess
from pathlib import Path
from txgnn_stage1.assets import acquire_assets
from txgnn_stage1.reference import capture_reference
from txgnn_stage1.compare import compare_real_snapshot

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
    comp = sub.add_parser("compare", help="Graph/input parity against reference snapshot")
    comp.add_argument("--snapshot", required=True, type=Path)
    comp.add_argument("--out", type=Path, default=Path("artifacts/local"))
    args = p.parse_args()
    if args.command == "fetch":
        records = acquire_assets(args.raw_dir)
        print(json.dumps({k: {"sha256": v["sha256"], "bytes": v["bytes"]} for k,v in records.items()}, indent=2))
    elif args.command == "reference":
        config = {
            "reference_sha": PIN, "split": args.split, "fold_seed": args.fold_seed,
            "preprocess_seed": args.preprocess_seed, "input_seed": args.input_seed,
            "input_dim": args.input_dim, "test_size": args.test_size,
            "mask_ratio": args.mask_ratio, "one_hop": args.one_hop, "no_kg": False,
        }
        path = capture_reference(args.raw_dir, args.upstream_repo, args.work_dir, config)
        print("Reference snapshot:", path)
    else:
        path, report = compare_real_snapshot(args.snapshot, args.out)
        print("Comparison bundle:", path)
        print(json.dumps(report, indent=2))
if __name__ == "__main__": main()
