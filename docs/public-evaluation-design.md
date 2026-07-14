# Public evaluation design

Proofline separates two behaviors:

1. **Clean-source fidelity:** does a transcription introduce a symbol error,
   omission, or contradiction that was not present in the source?
2. **Error preservation:** when a synthetic source deliberately contains a
   mistake, does the transcription preserve it instead of acting like a grader?

The public golden set uses generated examples only. It is deliberately small
and tests pipeline behavior, not production model quality. Expand it only with
synthetic, openly licensed, or explicitly consented documents whose provenance
is recorded alongside the fixture.

Generated transcriptions and scorecards must remain local unless they have been
reviewed for copied source content and personal information.
