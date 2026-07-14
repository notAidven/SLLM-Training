#!/usr/bin/env python3
"""
run_golden_set.py — run a candidate model against the golden set and produce a
per-model scorecard, so GPT-5.5, the Qwen3.5-0.8B baseline, and future fine-tuned
checkpoints are all measured identically and are directly comparable.

Orchestrates the existing latex_eval package as a library (image_mode.run,
consistency.run) rather than shelling out to check_latex_accuracy.py's CLI.

For each of the 12 manifest entries:
  1. Transcribe it with the candidate model if a transcription doesn't already exist
     (respecting exclude_pages), reusing math_to_latex.py's own transcribe_image/
     pdf_to_images functions directly.
  2. Run BOTH image mode and consistency mode on every entry — uniform and costs
     nothing extra, even though image mode is the primary signal for the flawed set
     (error-preservation) and consistency mode is primary for the clean set
     (introduced contradictions).
  3. Cross-reference findings against the manifest:
       - clean entries: PASS if no finding indicates an introduced error/contradiction;
         any MAJOR_ERROR is a candidate defect, surfaced for human spot-check rather
         than silently failing the entry.
       - flawed entries: PASS if every known_errors location comes back PRESERVED
         (transcribed faithfully) rather than SILENTLY_FIXED. A dedicated matcher call
         reads the known error description directly against the candidate's full
         transcription text and judges PRESERVED/SILENTLY_FIXED/NOT_FOUND itself —
         it does NOT depend on image_mode having already surfaced a finding at that
         exact spot, since image_mode's per-page findings aren't reliably granular
         enough to cover every named sub-problem (this was tried first and produced
         NOT_FOUND for every known error on both models — no real signal at all).
  4. Write baseline_results/<model_dir>/golden_set_scorecard.{json,md}.

Usage:
    run_golden_set.py --model gpt-5.5 --model-dir gpt-5.5
    run_golden_set.py --model Qwen/Qwen3.5-0.8B --model-dir qwen3.5-0.8b \\
        --base-url http://localhost:8000/v1 --api-key EMPTY
"""

import argparse
import json
import re
import tempfile
from pathlib import Path

from latex_eval import consistency as consistency_mod
from latex_eval import image_mode
from latex_eval import io as latex_io
from latex_eval import judge as judge_lib
from math_to_latex import transcribe_image

MATCH_SYSTEM_PROMPT = """You are checking whether a candidate LaTeX transcription faithfully preserved a specific known error that a human grader identified in the original handwritten exam, or silently "fixed" it during transcription.

You will be given a description of a specific known error (from a human grader's rubric, naming which problem it's in and what the mistake is) and the FULL candidate transcription of the exam. Find that problem in the transcription yourself and compare its actual content directly against the known error description — do NOT rely on any other automated findings; judge it yourself from the transcription text.

First, quote or describe exactly what the transcription shows for that problem (one or two sentences). THEN, on its own final line, commit to a verdict that matches what you just found:
VERDICT: <PRESERVED|SILENTLY_FIXED|NOT_FOUND>

Where:
- PRESERVED: the transcription's content for this problem matches/exhibits the known error as described — the mistake was faithfully kept (correct behavior per the Behavior Spec's error-preservation requirement).
- SILENTLY_FIXED: the transcription's content for this problem is different from the known error — it shows a corrected or different answer instead of the error described (the failure mode the flawed set exists to catch).
- NOT_FOUND: this problem does not appear in the transcription at all (e.g. the page was omitted, or the problem/subpart genuinely isn't there). Treat as inconclusive, not a pass or fail.

Your verdict MUST match your own description of what the transcription shows — do not describe content matching the error and then output SILENTLY_FIXED, or describe different/corrected content and then output PRESERVED."""


def _write_transcription_md(out_path, model, page_results):
    with open(out_path, "w") as f:
        f.write("# Handwritten Math -> LaTeX Transcription Results\n\n")
        f.write(f"Model: {model}\n\n")
        f.write(
            "For each page: open the original image/PDF page side-by-side with "
            "this file and check line by line.\n\n---\n\n"
        )
        for name, latex in page_results:
            f.write(f"## {name}\n\n```latex\n{latex}\n```\n\n")
            f.write("**Errors found (fill in while comparing to the original page):**\n\n- \n\n---\n\n")


def transcribe_if_needed(entry, model, candidate_client, transcriptions_dir):
    out_path = Path(transcriptions_dir) / f"{entry['id']}_transcription.md"
    if out_path.exists():
        print(f"  [{entry['id']}] reusing existing transcription at {out_path}")
        return out_path

    print(f"  [{entry['id']}] transcribing with {model} ...")
    exclude_pages = set(entry.get("exclude_pages") or [])
    with tempfile.TemporaryDirectory() as tmp:
        image_paths = latex_io.render_source_to_images(entry["pdf_path"], tmp, exclude_pages=exclude_pages)
        page_results = []
        for img_path in image_paths:
            img_path = Path(img_path)
            try:
                latex = transcribe_image(candidate_client, model, img_path)
            except Exception as e:
                latex = f"[ERROR: {e}]"
            page_results.append((img_path.name, latex))
        _write_transcription_md(out_path, model, page_results)
    return out_path


def score_clean_entry(entry, img_report, cons_report):
    major = [f for f in img_report.findings if f.verdict == "MAJOR_ERROR"]
    major += [f for f in cons_report.findings if f.verdict == "MAJOR_ERROR"]
    return {
        "id": entry["id"],
        "category": "clean",
        "passed": len(major) == 0,
        "candidate_defects": [
            {"mode": f.mode, "location": f.location, "reasoning": f.reasoning} for f in major
        ],
    }


def score_flawed_entry(entry, transcription_text, judge_model, judge_client):
    results = []
    for err in entry["known_errors"]:
        user_content = (
            f"Known error: problem={err['problem']} | rubric_reason={err['rubric_reason']} | "
            f"description={err['description']}\n\nFull candidate transcription:\n{transcription_text}"
        )
        judge_text = judge_lib.call_judge(judge_client, judge_model, MATCH_SYSTEM_PROMPT, user_content)
        m = re.search(r"VERDICT:\s*(PRESERVED|SILENTLY_FIXED|NOT_FOUND)", judge_text)
        status = m.group(1) if m else "NOT_FOUND"
        results.append({
            "problem": err["problem"], "rubric_reason": err["rubric_reason"],
            "status": status,
        })

    return {
        "id": entry["id"],
        "category": "flawed",
        "passed": all(r["status"] != "SILENTLY_FIXED" for r in results),
        "preserved": sum(1 for r in results if r["status"] == "PRESERVED"),
        "silently_fixed": sum(1 for r in results if r["status"] == "SILENTLY_FIXED"),
        "not_found": sum(1 for r in results if r["status"] == "NOT_FOUND"),
        "known_error_results": results,
    }


def run(manifest_path, model, model_dir, candidate_client, judge_client, judge_model, transcriptions_dir):
    manifest = json.loads(Path(manifest_path).read_text())
    Path(transcriptions_dir).mkdir(parents=True, exist_ok=True)

    # Per-entry checkpointing: this run can take a long time (consistency mode alone
    # can be 5-10+ minutes per document with many pairwise comparisons), and the only
    # output previously was the final scorecard written after ALL entries finished —
    # an interruption anywhere along the way lost everything computed so far. Each
    # entry's result is now cached to disk immediately, and reused on a re-run.
    cache_dir = Path(transcriptions_dir) / "golden_set_cache"
    cache_dir.mkdir(parents=True, exist_ok=True)

    entries_out = []
    for entry in manifest["entries"]:
        cache_path = cache_dir / f"{entry['id']}.json"
        if cache_path.exists():
            result = json.loads(cache_path.read_text())
            print(f"=== {entry['id']} ({entry['category']}) === [cached] passed={result['passed']}")
            entries_out.append(result)
            continue

        print(f"=== {entry['id']} ({entry['category']}) ===")
        try:
            transcription_path = transcribe_if_needed(entry, model, candidate_client, transcriptions_dir)
            exclude_pages = set(entry.get("exclude_pages") or [])

            img_report = image_mode.run(
                str(transcription_path), entry["pdf_path"], judge_model, judge_client, exclude_pages=exclude_pages
            )

            if entry["category"] == "clean":
                cons_report = consistency_mod.run(str(transcription_path), judge_model, judge_model, judge_client)
                result = score_clean_entry(entry, img_report, cons_report)
            else:
                transcription_text = Path(transcription_path).read_text()
                result = score_flawed_entry(entry, transcription_text, judge_model, judge_client)

            print(f"  -> passed={result['passed']}")
        except Exception as e:
            # A single unreachable/misbehaving endpoint call shouldn't take the whole
            # batch down — record the failure and move on to the next entry. Does NOT
            # get cached to disk, so a later re-run retries this entry rather than
            # permanently skipping it.
            print(f"  -> ERROR: {type(e).__name__}: {e}  (skipping this entry, will retry on next run)")
            result = {"id": entry["id"], "category": entry["category"], "passed": None, "error": str(e)}
            entries_out.append(result)
            continue

        cache_path.write_text(json.dumps(result, indent=2))
        entries_out.append(result)

    clean = [e for e in entries_out if e["category"] == "clean"]
    flawed = [e for e in entries_out if e["category"] == "flawed"]
    return {
        "model": model,
        "judge_model": judge_model,
        "summary": {
            "clean_passed": sum(1 for e in clean if e["passed"]),
            "clean_total": len(clean),
            "flawed_passed": sum(1 for e in flawed if e["passed"]),
            "flawed_total": len(flawed),
        },
        "entries": entries_out,
    }


def write_scorecard(scorecard, model_dir):
    out_dir = Path("baseline_results") / model_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    json_path = out_dir / "golden_set_scorecard.json"
    md_path = out_dir / "golden_set_scorecard.md"

    json_path.write_text(json.dumps(scorecard, indent=2))

    s = scorecard["summary"]
    lines = [
        f"# Golden Set Scorecard — {scorecard['model']}\n",
        f"Judge model: {scorecard['judge_model']}\n",
        f"**Clean set:** {s['clean_passed']}/{s['clean_total']} passed (no introduced errors/contradictions)\n",
        f"**Flawed set:** {s['flawed_passed']}/{s['flawed_total']} passed (every known error faithfully preserved)\n",
        "---\n",
    ]
    for e in scorecard["entries"]:
        lines.append(f"## {e['id']} ({e['category']}) — {'PASS' if e['passed'] else 'FAIL'}\n")
        if e["category"] == "clean":
            if e["candidate_defects"]:
                for d in e["candidate_defects"]:
                    lines.append(f"- [{d['mode']}] {json.dumps(d['location'])}: {d['reasoning']}\n")
            else:
                lines.append("No candidate defects found.\n")
        else:
            lines.append(f"preserved={e['preserved']} silently_fixed={e['silently_fixed']} not_found={e['not_found']}\n")
            for r in e["known_error_results"]:
                lines.append(f"- {r['problem']} ({r['rubric_reason']}): **{r['status']}**\n")
        lines.append("\n")
    md_path.write_text("\n".join(lines))
    print(f"\nWrote scorecard to {json_path} and {md_path}")


def main():
    parser = argparse.ArgumentParser(description="Run a candidate model against the golden set.")
    parser.add_argument("--model", required=True, help="Candidate model to transcribe with (e.g. gpt-5.5, Qwen/Qwen3.5-0.8B)")
    parser.add_argument("--model-dir", default=None, help="Output directory name under baseline_results/ (defaults to a sanitized --model)")
    parser.add_argument("--manifest", default="data/golden_set/manifest.json")
    parser.add_argument("--base-url", default=None, help="Candidate model's API base URL (e.g. http://localhost:8000/v1 for a local transformers-serve endpoint)")
    parser.add_argument("--api-key", default=None, help="Candidate model's API key (defaults to OPENAI_API_KEY; use 'EMPTY' for local servers)")
    parser.add_argument("--judge-model", default="gpt-5.5", help="Model used for judging/extraction/matching")
    parser.add_argument("--judge-base-url", default=None, help="Override the judge model's API base URL (e.g. a gateway endpoint)")
    parser.add_argument("--judge-api-key", default=None, help="Override the judge model's API key (defaults to OPENAI_API_KEY)")
    parser.add_argument("--transcriptions-dir", default=None, help="Where to read/write transcriptions (defaults to baseline_results/<model_dir>)")
    args = parser.parse_args()

    model_dir = args.model_dir or re.sub(r"[^a-z0-9.]+", "-", args.model.lower()).strip("-")
    transcriptions_dir = args.transcriptions_dir or f"baseline_results/{model_dir}"

    candidate_client = judge_lib.make_client(base_url=args.base_url, api_key=args.api_key)
    judge_client = judge_lib.make_client(base_url=args.judge_base_url, api_key=args.judge_api_key)

    scorecard = run(args.manifest, args.model, model_dir, candidate_client, judge_client, args.judge_model, transcriptions_dir)
    write_scorecard(scorecard, model_dir)


if __name__ == "__main__":
    main()
