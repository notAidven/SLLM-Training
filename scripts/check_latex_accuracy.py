#!/usr/bin/env python3
"""
check_latex_accuracy.py — score a candidate LaTeX transcription for accuracy.

Three subcommands:
  image       — compare candidate LaTeX against the original source photo/PDF directly
                (no separate ground truth needed; used for the real coursework problem sets)
  consistency — extract every named-quantity relation across a full document and flag
                contradictions between distant restatements of the same fact
  text        — compare candidate LaTeX against a known-correct reference LaTeX string
                (SymPy first, LLM-judge fallback; used for MathWriting-style pairs later)

All three share one Finding/Report data model, JSON (+ optional markdown) output, and
a mandatory-by-default spot-check so an aggregate number is never trusted blind.

Usage:
    check_latex_accuracy.py image --transcription report.md --source problemset.pdf \\
        --judge-model gpt-5.5 --output out.json --spot-check 5

    check_latex_accuracy.py consistency --transcription report.md \\
        --extractor-model gpt-5.5 --judge-model gpt-5.5 --output out.json

    check_latex_accuracy.py text --pairs pairs.jsonl --judge-model gpt-5.5 --output out.json
"""

import argparse

from latex_eval import consistency, image_mode, judge as judge_lib, text_mode
from latex_eval.report import spot_check


def add_common_args(p):
    p.add_argument("--output", required=True, help="Path to write the JSON report")
    p.add_argument("--output-md", default=None, help="Optional path to write a human-readable markdown report")
    p.add_argument("--spot-check", type=int, default=5, help="Number of findings to print for manual review (default 5; pass 0 to skip)")
    p.add_argument("--seed", type=int, default=None, help="Random seed for spot-check sampling (for reproducible review)")
    p.add_argument("--judge-base-url", default=None, help="Override the judge model's API base URL")
    p.add_argument("--judge-api-key", default=None, help="Override the judge model's API key (defaults to OPENAI_API_KEY)")


def build_parser():
    parser = argparse.ArgumentParser(description="Score LaTeX transcriptions for accuracy and cross-document consistency.")
    sub = parser.add_subparsers(dest="mode", required=True)

    p_image = sub.add_parser("image", help="Compare candidate LaTeX against the original source image/PDF")
    p_image.add_argument("--transcription", required=True, help="math_to_latex.py-style markdown report")
    p_image.add_argument("--source", required=True, help="Original PDF or image file(s) the transcription was produced from")
    p_image.add_argument("--judge-model", default="gpt-5.5")
    p_image.add_argument("--dpi", type=int, default=250)
    p_image.add_argument("--exclude-pages", default=None, help="Comma-separated 1-indexed page number(s) to skip (must match whatever --exclude-pages was used when transcribing, e.g. a trailing code screenshot)")
    add_common_args(p_image)

    p_cons = sub.add_parser("consistency", help="Flag cross-document contradictions in named-quantity relations")
    p_cons.add_argument("--transcription", required=True, help="math_to_latex.py-style markdown report")
    p_cons.add_argument("--extractor-model", default="gpt-5.5")
    p_cons.add_argument("--judge-model", default="gpt-5.5")
    add_common_args(p_cons)

    p_text = sub.add_parser("text", help="Compare candidate LaTeX against known-correct reference LaTeX")
    p_text.add_argument("--pairs", default=None, help="JSONL file of {id, candidate_latex, reference_latex}")
    p_text.add_argument("--reference-report", default=None, help="Alternative to --pairs: a reference math_to_latex.py report")
    p_text.add_argument("--candidate-report", default=None, help="Paired with --reference-report")
    p_text.add_argument("--judge-model", default="gpt-5.5")
    add_common_args(p_text)

    return parser


def main():
    parser = build_parser()
    args = parser.parse_args()

    client = judge_lib.make_client(base_url=args.judge_base_url, api_key=args.judge_api_key)

    if args.mode == "image":
        exclude_pages = {int(p.strip()) for p in args.exclude_pages.split(",") if p.strip()} if args.exclude_pages else None
        report = image_mode.run(args.transcription, args.source, args.judge_model, client, dpi=args.dpi, exclude_pages=exclude_pages)
    elif args.mode == "consistency":
        report = consistency.run(args.transcription, args.extractor_model, args.judge_model, client)
    elif args.mode == "text":
        report = text_mode.run(args, client)
    else:
        parser.error("Unknown mode")
        return

    report.write_json(args.output)
    counts = report.counts()
    print(f"Wrote {len(report.findings)} finding(s) to {args.output}")
    print(f"Counts: EXACT={counts['EXACT']}  MINOR_ERROR={counts['MINOR_ERROR']}  MAJOR_ERROR={counts['MAJOR_ERROR']}")

    if args.output_md:
        report.write_markdown(args.output_md)
        print(f"Wrote markdown report to {args.output_md}")

    if args.spot_check > 0:
        spot_check(report.findings, n=args.spot_check, seed=args.seed)


if __name__ == "__main__":
    main()
