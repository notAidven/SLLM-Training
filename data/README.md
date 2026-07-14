# Public data

The public repository contains synthetic examples only.

The original development corpus comprised 44 PDFs across 231 pages. It was
removed because some files contained identifying academic records, including
names, contact details, student identifiers, signatures, and grades. Raw
transcriptions and model outputs derived from those documents were removed as
well.

## Published fixtures

`public_samples/` contains two one-page PDFs generated from the project's
synthetic handwriting illustration:

- `synthetic-clean-derivative.pdf` contains a self-consistent derivative;
- `synthetic-preserved-error.pdf` contains one intentional visible error for
  testing whether a model transcribes rather than corrects the source.

Neither document represents a real person, course, assignment, or grade.

Regenerate both files with:

```bash
python3 scripts/create_public_fixtures.py
```

The evaluation manifest is `golden_set/manifest.json`.
