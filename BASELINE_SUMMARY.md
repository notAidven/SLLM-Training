# Historical baseline summary - redacted

The original evaluation compared a frontier model with a small local model on
a private handwritten-math corpus. That corpus included academic records, so
the 44 source PDFs, raw transcriptions, detailed grading-derived annotations,
and per-document model outputs have been removed.

The useful methodological conclusions were:

- transcription fidelity must be checked separately from mathematical
  correctness;
- a model should preserve visible mistakes instead of silently repairing them;
- an independent judge is preferable to a candidate model grading itself;
- judge prompts should request reasoning before a final verdict; and
- document-level pass/fail hides the difference between isolated symbol errors
  and missing or malformed pages.

Historical numeric results are intentionally omitted because they cannot be
reproduced from the public fixtures. New public scorecards should be generated
from `data/golden_set/manifest.json` and reported separately.
