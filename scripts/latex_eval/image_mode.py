"""Source-image mode: judge a candidate LaTeX transcription against the original photo.

There is no separate ground-truth transcript for the real coursework problem sets —
the only ground truth is the handwritten page itself. This mode sends the judge model
both the page image and the candidate LaTeX for that page and asks it to flag ANY
transcription error (not just consistency-related ones): misread symbols, dropped
steps, wrong fraction grouping, flattened multi-line work, misplaced matrix entries,
etc. This is the general accuracy check that consistency.py's contradiction-hunting
is layered on top of, not a replacement for.
"""

import tempfile
from pathlib import Path

from . import io as latex_io
from . import judge as judge_lib
from .report import Finding, Report

IMAGE_JUDGE_SYSTEM_PROMPT = """You are an expert grader comparing a candidate LaTeX transcription against a photo of the original handwritten math page it is supposed to represent.

Compare them carefully, line by line, covering the ENTIRE page — every symbol, every step, every structural element (fractions, exponents, matrices, summations, multi-line alignment). Do not stop at the first issue; report every distinct issue you find.

The candidate transcription follows these conventions — do NOT flag correct use of them as errors:
- `[ILLEGIBLE: best guess]` marks a genuinely hard-to-read symbol/word instead of silently guessing. This is correct, intended behavior, not an unwanted insertion — only flag it if the bracketed guess is a clearly wrong reading of a part that is NOT actually illegible.
- `[CROSSED OUT: ...]` preserves work the student crossed out rather than skipping it. This is correct, intended behavior, not an error.
- The task is transcribing MATHEMATICAL CONTENT AND STRUCTURE only. Do NOT flag missing reproduction of color, highlighting, underlining, or other visual/decorative styling — only flag a mark if it is itself mathematically meaningful (e.g. a strikethrough indicating retracted work, which should appear as [CROSSED OUT: ...]).

For each issue, first work out what actually differs, THEN commit to a verdict — output one line in this exact format (no other text on the line), with your reasoning BEFORE the verdict so you decide only after reasoning it through:
FINDING: location=<short pointer, e.g. "line 3" or "Problem 2b"> | reasoning=<one or two sentences citing exactly what differs> | verdict=<VERDICT>

Where <VERDICT> is one of:
- MAJOR_ERROR: a symbol, structure, or step is wrong in a way that changes the mathematical meaning (misread symbol, wrong sign, dropped term, flattened multi-line work, misplaced matrix entry, wrong fraction grouping, etc.); OR the transcription silently "corrected" a genuine mistake that was actually present in the original handwritten work instead of faithfully reproducing it.
- MINOR_ERROR: a cosmetic or notational slip that does NOT change the mathematical meaning (e.g. missing italics, spacing, a dropped label like "Problem 3" that doesn't affect the math itself).
- EXACT: use this ONLY to explicitly confirm a passage is correct, including confirming the transcription faithfully reproduced a genuine error that was actually present in the original source (do not flag faithful reproduction of the source's own mistake as a transcription error).

Your verdict MUST match the conclusion of your own reasoning — if your reasoning text ends up describing a match/no difference, the verdict must be EXACT, not MAJOR_ERROR or MINOR_ERROR.

If, after reviewing the entire page, you find nothing to flag, output exactly one line:
FINDING: location=whole page | reasoning=No errors found; transcription matches the source. | verdict=EXACT

Output ONLY FINDING lines. No preamble, no summary, no markdown fences."""


def run(transcription_path, source_path, judge_model, client, dpi=250, exclude_pages=None):
    from math_to_latex import get_mime_type, encode_image  # scripts/ is on sys.path via the entrypoint

    pages = latex_io.parse_report(transcription_path)

    with tempfile.TemporaryDirectory() as tmp:
        image_paths = latex_io.render_source_to_images(source_path, tmp, dpi=dpi, exclude_pages=exclude_pages)
        if len(image_paths) != len(pages):
            print(
                f"WARNING: {len(pages)} transcription page(s) but {len(image_paths)} source image(s) — "
                f"matching by index up to the shorter length ({min(len(pages), len(image_paths))})."
            )

        findings = []
        n = min(len(pages), len(image_paths))
        for i in range(n):
            page_label, latex = pages[i]
            img_path = Path(image_paths[i])
            mime = get_mime_type(img_path)
            b64 = encode_image(img_path)
            user_content = [
                {"type": "text", "text": f"Candidate LaTeX transcription for this page:\n\n{latex}"},
                {"type": "image_url", "image_url": {"url": f"data:{mime};base64,{b64}"}},
            ]
            judge_text = judge_lib.call_judge(client, judge_model, IMAGE_JUDGE_SYSTEM_PROMPT, user_content)
            parsed = judge_lib.parse_findings(judge_text)
            if not parsed:
                parsed = [{
                    "verdict": "MAJOR_ERROR",
                    "location": "unparseable judge output",
                    "reasoning": f"Judge output did not match the expected FINDING: format: {judge_text[:300]!r}",
                }]
            for j, p in enumerate(parsed):
                findings.append(Finding(
                    id=f"{page_label}#{j}",
                    mode="image",
                    verdict=p["verdict"],
                    method="llm_judge",
                    location={"page": page_label, "detail": p["location"]},
                    candidate_snippet=latex[:1000],
                    reasoning=p["reasoning"],
                    source_pointer=f"{Path(source_path).name} page {i + 1}",
                ))

    return Report(
        mode="image",
        metadata={
            "judge_model": judge_model,
            "transcription": str(transcription_path),
            "source": str(source_path),
            "pages_compared": n,
        },
        findings=findings,
    )
