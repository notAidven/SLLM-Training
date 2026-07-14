# Project Handoff: Handwritten Math → LaTeX SLLM Fine-Tuning

## Context
One-week assignment ("Train Your Own Small Learning Model," Alpha AI bootcamp): fine-tune a small open model (0.6B-4B params, QLoRA) to reliably perform ONE narrow behavior that a well-prompted frontier model can't already do reliably (the "litmus test"). The dataset is the deliverable, not the model. One-week arc: Day 1 = research + BrainLift (done). Day 2 = Behavior Spec + eval harness + data-gen pipeline + 50-example smoke test (starting now). Day 3 = v1 dataset + first training run. Day 4 = iterate on failure modes. Day 5 = ship + defend.

Mentor's key guidance, already internalized: don't look for a domain that wholesale fails — frontier models are broadly 80-90%+ competent almost everywhere. Find the narrow, specific, low-frequency slice within a domain of real expertise where reliability breaks down (the "15 out of 100" pattern), and prove it empirically, not by assertion.

**Extensive testing already ruled out** (all passed cleanly against Claude Sonnet 5 and/or GPT-5.5, meaning they fail the litmus test): general algebra/discrete-math error diagnosis, MCAT-style physics (both multiple-choice and free-response with inverse reasoning), AP Physics C problem generation, propositional logic Peer/Teacher/Judge classification (including harder long-chain state-tracking variants), trivialized classic logic puzzles (DeepMind's UNPUZZLES dataset), and a flashcard-quality rubric grader (untested but deprioritized). Don't re-litigate these — they were tested rigorously and failed to show a gap.

## Locked Decisions — Do Not Relitigate
- **Behavior domain**: Handwritten mathematical work → LaTeX transcription.
- **Target skill**: NOT raw symbol/character-level OCR (this is now largely solved by frontier models — see Confirmed Finding below). The target is **long-range cross-line/cross-page self-consistency** — does the model maintain a consistent relationship for the same quantity across a long, multi-step derivation, not just get each line individually plausible.
- **Architecture**: Option A — full pipeline, image → LaTeX end-to-end, done by the fine-tuned small model itself (vision + consistency together). This was an explicit choice over the cheaper/simpler alternative (a text-only downstream consistency-checking pass on a frontier model's draft transcription), made with full awareness that Option A requires the small model to also reach adequate baseline OCR competence, which is a second, separate problem from the actual novel target behavior.
- **Base model**: `Qwen/Qwen3.5-0.8B` (Alibaba, Apache 2.0, native vision+language, released Feb 2026). Confirmed real and accessible on Hugging Face.
- **Training data source**: MathWriting dataset (Google Research, 2024) — 230k human-written handwritten math expressions. Pre-rendered version available directly via `datasets.load_dataset("deepcopy/MathWriting-Human")` (image + LaTeX pairs, train/val/test splits) — no need to parse raw InkML stroke data.
  - **License caveat**: CC BY-NC-SA 4.0 — NonCommercial. Relevant to the "publish your dataset" final deliverable; fine for academic/portfolio use.

## Confirmed Empirical Finding (the project's primary evidence — already verified against the real source, not just observed)
Tested GPT-5.5 (frontier, for comparison) on two real, personal, previously-unseen handwritten problem sets via a custom transcription script:
- **Private clean document A** (linear algebra): dense multi-page work with no confirmed transcription errors.
- **Private clean document B** (probability): dense multi-page work with one confirmed model-introduced error. The same inequality relating two expectations changed direction when it was restated many lines later, even though the source remained consistent. The private source has since been removed from the public repository.

This connects to a published, general mechanism: Malek et al. (Google DeepMind, 2025), "Frontier LLMs Still Struggle with Simple Reasoning Tasks" — documents "reasoning delirium," where model performance degrades measurably as the amount of state to track simultaneously increases, even on fundamentally simple tasks. Maintaining one quantity's relation consistently across dozens of lines is a direct instance of this.

Also important context: a 2026 large-scale classroom study found general frontier models (GPT-4.1-mini) now **beat** specialist OCR tools (Mathpix) at handwritten math transcription (84% vs 55% acceptable on a hard subset) — confirming the original 2023-era hypothesis ("AI is bad at reading messy handwriting") is stale. The real, current gap is specifically the long-range consistency dimension, which no existing published HME benchmark (CROHME, im2latexv2, MathWriting) even measures, since they all score isolated single expressions, not multi-step derivations.

Full literature review and citations: see `HME_BrainLift_Evan_Cabrera.md` (already written, already user-edited — read this file first for full context and source list).

## Already Built (scripts likely in the user's local Downloads folder — check before rewriting)
- `handwriting_to_latex.py` — takes image(s) or a PDF (auto-converts pages via PyMuPDF), calls a vision-capable model via the OpenAI-compatible chat completions API, outputs LaTeX transcriptions to a markdown report.
- `check_latex_accuracy.py` — scores transcriptions against ground truth two ways: (1) strict symbolic equivalence via SymPy for extractable final answers, (2) LLM-as-judge for full multi-step transcriptions, classified EXACT/MINOR_ERROR/MAJOR_ERROR. Has a `--spot-check N` flag that prints N random judge verdicts for manual human verification of the judge itself — treat this as mandatory, not optional, before trusting any aggregate accuracy number.
- Several scripts from abandoned hypotheses (physics batteries, logic classification, unpuzzles, flashcard grading) — not directly reusable for this direction, but demonstrate the established working pattern: formally verify ground truth before building a test (SAT solvers for logic, independent symbolic recomputation for physics), prefer real academic datasets over fully synthetic ones, and always spot-check automated judges against human review.

## Immediate Next Steps (Day 2)

1. **Write the Behavior Spec** (falsifiable, specific):
   > *Given a photo of a multi-step handwritten mathematical derivation, the model produces a LaTeX transcription in which every named quantity, relation, and inequality direction stays consistent with how it was first introduced across the entire document — never contradicting itself between distant lines — while faithfully preserving any genuine error present in the original source material rather than silently correcting or newly introducing one.*

2. **Get a real baseline number for Qwen3.5-0.8B before touching training.** Setup:
   ```
   pip install "transformers[serving] @ git+https://github.com/huggingface/transformers.git@main" torch torchvision pillow
   transformers serve --force-model Qwen/Qwen3.5-0.8B
   ```
   This exposes a local OpenAI-compatible endpoint at `localhost:8000/v1`. Point the existing `openai.OpenAI(base_url="http://localhost:8000/v1", api_key="EMPTY")` at it and rerun `handwriting_to_latex.py` + `check_latex_accuracy.py` against the SAME two real problem sets already tested on GPT-5.5. Report symbol-level accuracy and consistency-check results as two SEPARATE numbers, not one blended score — this determines whether Option A's dual-problem risk (raw OCR competence + consistency) is as bad as expected.
   Note: Qwen3.5 GGUF currently does not work with Ollama (separate vision mmproj files) — `transformers serve` is the correct path, not Ollama.

3. **Extend the eval harness for cross-document consistency checking specifically.** Current `check_latex_accuracy.py` compares line-by-line / whole-document holistically via LLM judge, but doesn't have a dedicated pass that explicitly extracts every named quantity + its stated relation each time it recurs, and flags any pairwise contradiction across the FULL document (not just adjacent lines). Build this — it doubles as both the eval metric and (repurposed) a synthetic-error injector for data generation.

4. **Build the MathWriting composition pipeline** (not yet started):
   - Load samples via `datasets.load_dataset("deepcopy/MathWriting-Human")`.
   - Known problem to solve: individual MathWriting samples are independent standalone Wikipedia formulas — they don't share variables/quantities with each other by default. Naively stacking N images vertically produces a page that *looks* like a derivation but has no real cross-line relationship to test.
   - Approach: sample 2+ (image, latex) pairs, compose into one "page" image (e.g., vertical stacking via PIL), and post-process/relabel a shared variable so it's used consistently across the composed page for positive/correct training examples. This is the main open engineering task.

5. **Run a 50-example smoke test** of the full generate → transcribe → score loop before scaling to hundreds/thousands of examples.

6. **First QLoRA fine-tuning run** on Qwen3.5-0.8B once the smoke test passes clean. Note: for Mac-local training (no CUDA), MLX-LM (Apple Silicon) is the viable local path; Unsloth via Modal/RunPod/Colab is the alternative if cloud GPU is preferred.

## Working Conventions Established So Far
- User is on macOS (zsh, `/usr/bin/python3`), prefers ready-to-run scripts over setup instructions.
- Scripts should read API keys from environment variables by default (`OPENAI_API_KEY`) — a key was hardcoded into one script earlier at the user's explicit request for local-testing convenience only; that key should be treated as already compromised (it was pasted in plaintext) and rotated before any repo goes public, and hardcoding should not be repeated as a default pattern.
- Always independently verify ground truth (via a solver, symbolic math check, or manual source comparison) before building a test around an assumed-correct answer key — this project has caught real self-authored errors this way multiple times.
- Prefer testing against real, novel content (the user's own past coursework) over synthetic or benchmark data where possible, specifically to rule out training-data memorization as a confound.
