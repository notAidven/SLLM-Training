#!/usr/bin/env python3
"""
exclude_eval_docs.py — the training/leakage guard every data-prep script should
consult before including a document in the training set.

Compares by file content hash (SHA-256), not filename or path string, so a
renamed or re-copied duplicate of a golden-set PDF is still caught.

Usage as a library:
    from exclude_eval_docs import is_eval_document, load_eval_hashes
    if is_eval_document(candidate_path):
        skip it

Usage as a CLI — scan a directory of candidate training documents and report
which are eval documents (must be excluded) vs. available for training:
    python3 scripts/training_data/exclude_eval_docs.py data/public_samples/
"""

import argparse
import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent
MANIFEST_PATH = ROOT / "data" / "golden_set" / "manifest.json"


def sha256_of(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def load_eval_hashes():
    """Returns {sha256_hash: golden_set_id} for every document referenced by the
    golden-set manifest. Missing files are skipped with a warning rather than
    raising — a stale manifest entry shouldn't crash every caller of this guard."""
    manifest = json.loads(MANIFEST_PATH.read_text())
    hashes = {}
    for entry in manifest["entries"]:
        pdf_path = ROOT / entry["pdf_path"]
        if not pdf_path.exists():
            print(f"WARNING: manifest entry {entry['id']} points to a missing file: {pdf_path}", file=sys.stderr)
            continue
        hashes[sha256_of(pdf_path)] = entry["id"]
    return hashes


def is_eval_document(path, eval_hashes=None):
    """True if `path`'s content matches a golden-set (eval) document, regardless
    of filename/location. Pass a pre-loaded `eval_hashes` dict (from
    load_eval_hashes()) when checking many files, to avoid re-hashing the 12
    golden-set PDFs on every call."""
    eval_hashes = eval_hashes if eval_hashes is not None else load_eval_hashes()
    return sha256_of(path) in eval_hashes


def scan_directory(candidate_dir):
    eval_hashes = load_eval_hashes()
    candidate_dir = Path(candidate_dir)
    excluded, available = [], []
    for pdf_path in sorted(candidate_dir.glob("**/*.pdf")):
        h = sha256_of(pdf_path)
        if h in eval_hashes:
            excluded.append((pdf_path, eval_hashes[h]))
        else:
            available.append(pdf_path)
    return excluded, available


def main():
    parser = argparse.ArgumentParser(description="Scan a directory for documents that overlap with golden-set eval data.")
    parser.add_argument("directory", help="Directory of candidate training PDFs to scan (recursively)")
    args = parser.parse_args()

    excluded, available = scan_directory(args.directory)

    print(f"=== Excluded (matches a golden-set eval document — DO NOT use for training) ===")
    if not excluded:
        print("  (none)")
    for path, eval_id in excluded:
        print(f"  {path}  <-- matches golden-set entry '{eval_id}'")

    print(f"\n=== Available for training ({len(available)} document(s)) ===")
    for path in available:
        print(f"  {path}")


if __name__ == "__main__":
    main()
