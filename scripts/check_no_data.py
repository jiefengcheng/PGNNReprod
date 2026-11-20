#!/usr/bin/env python3
"""Fail if the staged or tracked repo would accidentally include data or large artifacts."""
import subprocess
import sys
from pathlib import Path

BANNED_DIRS = {"data", "datasets", "raw", "processed", "checkpoints", "models", "model_ckpt", "saved_models", "artifacts/local", "reference/TxGNN"}
BANNED_EXTS = {".npy", ".npz", ".pt", ".pth", ".bin", ".ckpt", ".parquet", ".sqlite", ".db", ".h5", ".hdf5", ".safetensors", ".pkl", ".pickle", ".zip"}
MAX_FILE_BYTES = 2_000_000

def tracked():
    out = subprocess.check_output(["git", "ls-files", "--cached", "-z"])
    return [Path(x.decode()) for x in out.split(b"\0") if x]

def main():
    failures=[]
    for path in tracked():
        posix=path.as_posix()
        if posix.startswith("data/") or posix.startswith("datasets/") or posix.startswith("artifacts/local/") or posix.startswith("reference/TxGNN/"):
            failures.append((posix,"prohibited data/generated/reference path")); continue
        if path.suffix.lower() in BANNED_EXTS:
            failures.append((posix,"prohibited binary/data extension")); continue
        if path.exists() and path.stat().st_size > MAX_FILE_BYTES:
            failures.append((posix,"file exceeds repository safety threshold"))
    if failures:
        for name,reason in failures:print(f"BLOCKED: {name}: {reason}",file=sys.stderr)
        raise SystemExit(1)
    print(f"PASS: {len(tracked())} staged/tracked files contain no prohibited data or artifacts")
if __name__ == "__main__": main()
