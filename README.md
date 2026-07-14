# Proofline

Proofline is a research preview for turning handwritten mathematics into
editable LaTeX without silently correcting the writer's reasoning.

[Open the browser demo](https://notaidven.github.io/SLLM-Training/)

## What is here

- a browser-only demo for upload, review, editing, and export;
- an evaluation harness for transcription fidelity and consistency;
- training-data utilities for public datasets; and
- a small, fully synthetic public fixture set.

The deployed site currently runs in an explicitly labeled demo mode with
placeholder transcription. It does not send uploads to a model backend.

## Public data policy

The original development corpus contained private academic records and has
been removed from the public repository, along with raw transcriptions and
model outputs derived from it. Git history was rewritten so those source files
are not retained on the public branch.

Only synthetic examples are published. They contain no real student work,
names, email addresses, student identifiers, signatures, or grades. See
[`data/README.md`](data/README.md) for details.

## Run the demo locally

```bash
python3 -m http.server 8080 --directory site
```

Then open <http://localhost:8080>.

## Run the public fixture evaluation

Regenerate the synthetic PDFs:

```bash
python3 scripts/create_public_fixtures.py
```

Run a candidate model against `data/golden_set/manifest.json` with
`scripts/run_golden_set.py`. Generated transcriptions and scorecards are
written under `baseline_results/` and intentionally ignored by Git.

## Status

The public repository demonstrates the product interaction and evaluation
method. A live fine-tuned checkpoint and production inference service are not
currently included.
