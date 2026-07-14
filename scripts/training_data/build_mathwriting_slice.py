#!/usr/bin/env python3
"""
build_mathwriting_slice.py — sample N (image, LaTeX) pairs from the
MathWriting-Human dataset and reformat them into training examples that match
the exact chat structure math_to_latex.py's transcribe_image() already uses
for inference (system prompt + user image + assistant LaTeX target), so
training and inference formats stay consistent.

No composition/relabeling of samples — each MathWriting formula is an
independent, standalone snippet and is used exactly as-is. This targets the
baseline's dominant, largest observed problem (raw transcription fidelity:
356 defects across 8 real documents, whole pages garbled/omitted) rather than
consistency specifically — see the training-data plan for why.

Dataset license: CC BY-NC-SA 4.0 (NonCommercial) — fine for this academic
project, relevant if the assembled dataset is published later.

Output: a directory of PNG images plus a JSONL file, one training example per
line:
    {"image": "images/000001.png", "messages": [
        {"role": "system", "content": "..."},
        {"role": "user", "content": [{"type":"text","text":"..."},{"type":"image"}]},
        {"role": "assistant", "content": "<latex target>"}
    ]}

Usage:
    python3 scripts/training_data/build_mathwriting_slice.py --n 3000 --output data/training/mathwriting_slice
"""

import argparse
import json
from pathlib import Path

from datasets import load_dataset

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from math_to_latex import SYSTEM_PROMPT

USER_TEXT = "Transcribe the handwritten math in this image into LaTeX."


def build_slice(n, output_dir, split="train", seed=0, skip=0, start_index=0):
    """skip: how many samples of the (seed-determined) shuffled stream to skip before
    collecting — using the SAME seed with different skip offsets partitions the
    stream into non-overlapping shards, so multiple parallel collectors never
    duplicate a sample. start_index offsets output filenames so shards can be
    merged into one directory without filename collisions."""
    output_dir = Path(output_dir)
    images_dir = output_dir / "images"
    images_dir.mkdir(parents=True, exist_ok=True)
    jsonl_path = output_dir / "examples.jsonl"

    ds = load_dataset("deepcopy/MathWriting-Human", split=split, streaming=True)
    ds = ds.shuffle(seed=seed, buffer_size=10_000)
    if skip:
        ds = ds.skip(skip)

    written = 0
    skipped = 0
    with open(jsonl_path, "w") as out:
        for sample in ds:
            if written >= n:
                break
            latex = (sample.get("latex") or "").strip()
            image = sample.get("image")
            if not latex or image is None:
                skipped += 1
                continue

            img_filename = f"{start_index + written:06d}.png"
            image.convert("RGB").save(images_dir / img_filename)

            example = {
                "image": f"images/{img_filename}",
                "sample_id": sample.get("sample_id"),
                "source": "MathWriting-Human",
                "messages": [
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": [
                        {"type": "text", "text": USER_TEXT},
                        {"type": "image"},
                    ]},
                    {"role": "assistant", "content": latex},
                ],
            }
            out.write(json.dumps(example) + "\n")
            written += 1

    print(f"Wrote {written} example(s) to {jsonl_path} (skipped {skipped} malformed sample(s))")
    print(f"Images saved under {images_dir}")


def main():
    parser = argparse.ArgumentParser(description="Build a MathWriting-Human raw-fidelity training slice.")
    parser.add_argument("--n", type=int, default=3000, help="Number of examples to sample")
    parser.add_argument("--output", default="data/training/mathwriting_slice", help="Output directory")
    parser.add_argument("--split", default="train", help="MathWriting-Human split to sample from")
    parser.add_argument("--seed", type=int, default=0, help="Shuffle seed, for reproducibility")
    parser.add_argument("--skip", type=int, default=0, help="Skip this many samples in the shuffled stream before collecting (use with a fixed --seed to shard across parallel collectors)")
    parser.add_argument("--start-index", type=int, default=0, help="Offset for output image filenames, so shards can be merged without collisions")
    args = parser.parse_args()
    build_slice(args.n, args.output, split=args.split, seed=args.seed, skip=args.skip, start_index=args.start_index)


if __name__ == "__main__":
    main()
