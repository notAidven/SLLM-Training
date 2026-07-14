"""Shared Finding/Report data model, JSON + markdown writers, and spot-check printing."""

import json
import random
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Optional


@dataclass
class Finding:
    id: str
    mode: str  # "text" | "image" | "consistency"
    verdict: str  # EXACT | MINOR_ERROR | MAJOR_ERROR
    method: str  # "sympy" | "llm_judge"
    location: dict
    candidate_snippet: str
    reasoning: str = ""
    reference_snippet: Optional[str] = None
    source_pointer: Optional[str] = None


@dataclass
class Report:
    mode: str
    metadata: dict
    findings: list = field(default_factory=list)

    def counts(self):
        c = {"EXACT": 0, "MINOR_ERROR": 0, "MAJOR_ERROR": 0}
        for f in self.findings:
            c[f.verdict] = c.get(f.verdict, 0) + 1
        return c

    def to_dict(self):
        return {
            "mode": self.mode,
            "metadata": self.metadata,
            "counts": self.counts(),
            "findings": [asdict(f) for f in self.findings],
        }

    def write_json(self, path):
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        Path(path).write_text(json.dumps(self.to_dict(), indent=2))

    def write_markdown(self, path):
        counts = self.counts()
        lines = [
            f"# {self.mode} eval report\n",
            f"Metadata: `{json.dumps(self.metadata)}`\n",
            f"**Counts:** EXACT={counts['EXACT']}  MINOR_ERROR={counts['MINOR_ERROR']}  MAJOR_ERROR={counts['MAJOR_ERROR']}\n",
            "---\n",
        ]
        for f in self.findings:
            lines.append(f"## {f.id} — {f.verdict} ({f.method})\n")
            lines.append(f"Location: `{json.dumps(f.location)}`\n")
            lines.append("```latex\n" + f.candidate_snippet + "\n```\n")
            if f.reference_snippet:
                lines.append("Reference:\n```latex\n" + f.reference_snippet + "\n```\n")
            if f.source_pointer:
                lines.append(f"Source: {f.source_pointer}\n")
            lines.append(f"Reasoning: {f.reasoning}\n")
            lines.append("---\n")
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        Path(path).write_text("\n".join(lines))


def spot_check(findings, n=5, seed=None):
    """Print N randomly-sampled findings with full context for manual review.

    Mandatory-by-default (called with n=5 unless the caller passes --spot-check 0):
    an aggregate number from an unverified judge is not trustworthy on its own.
    """
    if not findings:
        print("No findings to spot-check.")
        return
    rng = random.Random(seed)
    sample = rng.sample(findings, min(n, len(findings)))
    print(f"\n=== SPOT CHECK: {len(sample)} of {len(findings)} findings (seed={seed}) ===\n")
    for f in sample:
        print(f"--- {f.id} | verdict={f.verdict} | method={f.method} ---")
        print(f"Location: {f.location}")
        print(f"Candidate: {f.candidate_snippet}")
        if f.reference_snippet:
            print(f"Reference: {f.reference_snippet}")
        if f.source_pointer:
            print(f"Source: {f.source_pointer}")
        print(f"Judge reasoning: {f.reasoning}")
        print()
