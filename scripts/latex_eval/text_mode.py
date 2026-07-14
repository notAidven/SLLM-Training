"""Text ground-truth mode: compare candidate LaTeX against a known-correct reference
LaTeX string. Tries SymPy symbolic equivalence first for cleanly-extractable
expressions (a \\boxed{} or a short, environment-free equation/inequality), falling
back to an LLM judge for anything else — expected to be the common case against this
project's real multi-step derivations (matrices, \\mathbb{E}, align* blocks). SymPy
mainly earns its keep later against MathWriting's isolated single expressions.
"""

import json
import re
from pathlib import Path

from . import io as latex_io
from . import judge as judge_lib
from .report import Finding, Report

TEXT_JUDGE_SYSTEM_PROMPT = """You are grading whether a candidate LaTeX transcription matches a known-correct reference LaTeX transcription of the same mathematical content.

Compare them for mathematical meaning, not verbatim text — different but equivalent notation (e.g. reordered terms, \\ge vs \\geq) is not an error.

First, briefly explain your reasoning (one or two sentences). Then, on its own final line, output:
VERDICT: <EXACT|MINOR_ERROR|MAJOR_ERROR>

Where:
- MAJOR_ERROR: the candidate changes the mathematical meaning versus the reference (wrong symbol, sign, structure, dropped/added step).
- MINOR_ERROR: a cosmetic/notational difference that does not change meaning.
- EXACT: mathematically equivalent to the reference."""

_BOXED_RE = re.compile(r"\\boxed\{(.+)\}", re.DOTALL)
_ENV_RE = re.compile(r"\\begin\{(align\*?|array|bmatrix|pmatrix|matrix|cases)\}")


def _extractable(latex_str):
    """Only attempt SymPy on a single boxed answer or a short, environment-free
    equation/inequality — full derivations (align*/array/matrices/prose) reliably
    fail LaTeX parsing, so skip straight to the LLM judge for those instead of
    wasting a doomed parse attempt."""
    s = latex_str.strip()
    if _ENV_RE.search(s):
        return None
    m = _BOXED_RE.search(s)
    if m:
        return m.group(1).strip()
    if len(s.splitlines()) == 1 and len(s) < 200:
        return s
    return None


def _get_latex_parser():
    """Prefer the lark backend (pure-Python, installed via the `lark` package) over
    the antlr backend (needs antlr4-python3-runtime) — whichever is available."""
    try:
        from sympy.parsing.latex import parse_latex_lark
        return parse_latex_lark
    except ImportError:
        pass
    from sympy.parsing.latex import parse_latex
    return parse_latex


def _sympy_equivalent(candidate_expr, reference_expr):
    """Strict symbolic equivalence. Returns True/False, or None if either side fails
    to parse (caller falls back to the LLM judge in that case)."""
    try:
        import sympy
        parse = _get_latex_parser()
    except ImportError:
        return None
    try:
        c = parse(candidate_expr)
        r = parse(reference_expr)
    except Exception:
        return None
    try:
        if isinstance(c, sympy.Eq) or isinstance(r, sympy.Eq):
            if not (isinstance(c, sympy.Eq) and isinstance(r, sympy.Eq)):
                return False
            same_order = sympy.simplify((c.lhs - r.lhs) - (c.rhs - r.rhs)) == 0
            swapped = sympy.simplify((c.lhs - r.rhs) - (c.rhs - r.lhs)) == 0
            return same_order or swapped
        return sympy.simplify(c - r) == 0
    except Exception:
        return None


def _load_pairs(args):
    if args.pairs:
        pairs = []
        for line in Path(args.pairs).read_text().splitlines():
            line = line.strip()
            if line:
                pairs.append(json.loads(line))
        return pairs
    if args.reference_report and args.candidate_report:
        ref_pages = latex_io.parse_report(args.reference_report)
        cand_pages = latex_io.parse_report(args.candidate_report)
        if len(ref_pages) != len(cand_pages):
            print(
                f"WARNING: reference has {len(ref_pages)} page(s), candidate has {len(cand_pages)} — "
                f"matching by index up to the shorter length."
            )
        n = min(len(ref_pages), len(cand_pages))
        return [
            {"id": cand_pages[i][0], "candidate_latex": cand_pages[i][1], "reference_latex": ref_pages[i][1]}
            for i in range(n)
        ]
    raise ValueError("text mode needs either --pairs, or both --reference-report and --candidate-report")


def run(args, client):
    pairs = _load_pairs(args)
    findings = []
    sympy_hits = 0

    for idx, pair in enumerate(pairs):
        pid = pair.get("id", f"pair{idx}")
        candidate, reference = pair["candidate_latex"], pair["reference_latex"]

        cand_expr, ref_expr = _extractable(candidate), _extractable(reference)
        if cand_expr is not None and ref_expr is not None:
            result = _sympy_equivalent(cand_expr, ref_expr)
            if result is not None:
                sympy_hits += 1
                findings.append(Finding(
                    id=pid, mode="text", verdict="EXACT" if result else "MAJOR_ERROR", method="sympy",
                    location={"pair_id": pid}, candidate_snippet=candidate, reference_snippet=reference,
                    reasoning="SymPy symbolic equivalence check." if result else "SymPy found the expressions are not equivalent.",
                ))
                continue

        user_content = f"Reference (known correct):\n{reference}\n\nCandidate:\n{candidate}"
        judge_text = judge_lib.call_judge(client, args.judge_model, TEXT_JUDGE_SYSTEM_PROMPT, user_content)
        verdict, reasoning = judge_lib.parse_verdict(judge_text)
        if verdict is None:
            verdict = "MAJOR_ERROR"
            reasoning = f"Judge output did not match expected VERDICT format: {judge_text[:300]!r}"
        findings.append(Finding(
            id=pid, mode="text", verdict=verdict, method="llm_judge",
            location={"pair_id": pid}, candidate_snippet=candidate, reference_snippet=reference, reasoning=reasoning,
        ))

    return Report(
        mode="text",
        metadata={"judge_model": args.judge_model, "pairs": len(pairs), "sympy_hits": sympy_hits},
        findings=findings,
    )
