# Rubric: EXACT / MINOR_ERROR / MAJOR_ERROR

This is the shared, citable contract behind every verdict the eval harness (`scripts/check_latex_accuracy.py` + `scripts/latex_eval/`) produces. It's extracted and reconciled from the three judge prompts already embedded in `image_mode.py`, `consistency.py`, and `text_mode.py` — this document doesn't redesign the rubric, it states what's already being enforced in one place, citable by a human reviewing judge output (including for the final "defend" deliverable).

## The core contract, tied directly to the Behavior Spec

The Behavior Spec (`behavior_spec.md`) requires a transcription to do two things simultaneously:

1. **Never introduce a new cross-line contradiction that wasn't in the source.**
2. **Faithfully preserve any genuine error already in the source**, rather than silently "fixing" it.

Both halves matter for every verdict below — a transcription that reads well but violates either one is not EXACT.

## Verdict definitions

### MAJOR_ERROR
A transcription-quality problem that changes mathematical meaning, or a violation of either half of the Behavior Spec. Specifically:
- A symbol, sign, structure, or step is wrong in a way that changes the mathematical meaning (misread symbol, wrong sign, dropped term, flattened multi-line work, misplaced matrix entry, wrong fraction grouping, etc.).
- The transcription silently "corrected" a genuine mistake that was actually present in the original handwritten source, instead of faithfully reproducing it (violates Behavior Spec requirement #2).
- Two mentions of what is meant to be the same fact restate it with the relation direction or equality flipped, with no legitimate sandwich/combination argument shown that would make both directions independently valid (violates Behavior Spec requirement #1 — this is the `consistency` mode's `SAME_CLAIM_CONTRADICTION` label).
- In `text` mode: the candidate changes the mathematical meaning versus a known-correct reference (wrong symbol, sign, structure, dropped/added step).

### MINOR_ERROR
A cosmetic or notational difference that does **not** change mathematical meaning. Examples: missing italics, spacing, a dropped non-mathematical label (e.g. "Problem 3") that doesn't affect the math itself, reordered-but-equivalent terms, `\ge` vs `\geq`.

Also used for `consistency` mode's `DIFFERENT_CLAIMS_NOT_COMPARABLE` label — two claims that got compared but turned out not to actually be restatements of the same fact (a clustering error, not a transcription or consistency defect). This is logged for transparency but is explicitly **not** counted toward the headline contradiction metric — see `BASELINE_SUMMARY.md`'s caveats for why.

### EXACT
Correct, with two things this project treats as EXACT that a naive line-by-line diff would not:
- **Faithful reproduction of a genuine error present in the original source.** A transcription that keeps a student's actual mistake exactly as written is EXACT, not an error — silently fixing it would be the actual MAJOR_ERROR (see above). This directly operationalizes Behavior Spec requirement #2, and is why the `image` mode judge prompt explicitly instructs: don't flag faithful reproduction of the source's own mistake as a transcription error.
- **Two mentions of the same fact that agree**, including after legitimate algebraic rearrangement (`consistency` mode's `SAME_CLAIM_CONSISTENT`) or where two different-direction bounds are legitimately independent (a sandwich/squeeze argument), not a contradiction.

### Annotation conventions that are NOT errors
The candidate transcription convention (from `math_to_latex.py`'s own system prompt) uses two inline markers that judges must recognize as intentional, correct behavior rather than defects:
- `[ILLEGIBLE: best guess]` — marks a genuinely hard-to-read symbol instead of silently guessing. Only flag this if the bracketed guess is a clearly wrong reading of a part that is NOT actually illegible.
- `[CROSSED OUT: ...]` — preserves work the student crossed out rather than skipping it.

Similarly, the transcription task targets **mathematical content and structure only** — missing reproduction of color, highlighting, or underlining (as opposed to mathematically meaningful marks like a strikethrough) is not an error. Both of these were added to the `image` mode judge prompt after an early validation run flagged them incorrectly — see `BASELINE_SUMMARY.md`.

## Per-mode application

| Mode | What's being compared | Primary defect this mode catches |
|---|---|---|
| `image` | Candidate LaTeX vs. the original source photo/PDF directly | General transcription fidelity (any symbol/structure error) + the error-preservation half of the Behavior Spec |
| `consistency` | Every restatement of a named-quantity relation against every other restatement, across the full document | The introduced-contradiction half of the Behavior Spec — this is the project's actual differentiating metric |
| `text` | Candidate LaTeX vs. a known-correct reference LaTeX string (SymPy first, LLM-judge fallback) | Symbolic/mathematical equivalence — used for MathWriting-style ground-truth pairs, not the real coursework problem sets (which have no separate reference transcript) |

## Known reliability caveat — read before trusting any aggregate count

Manual verification against actual source images (done during baseline validation, documented in `BASELINE_SUMMARY.md`) found that roughly 40-50% of `image` mode's MAJOR_ERROR findings on dense, cluttered pages did not hold up under human review — the judge itself sometimes misreads the handwriting (e.g. "q" read as "9") or invents a discrepancy that isn't there. This rubric describes what a verdict is *supposed* to mean; it does not guarantee the judge always applies it correctly. Every score produced by this harness should be treated as provisional until spot-checked (`--spot-check N`, default 5, mandatory-by-default) — see `BASELINE_SUMMARY.md` for the concrete false-positive rate found so far.
