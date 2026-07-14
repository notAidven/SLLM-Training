#!/usr/bin/env python3
"""Generate the synthetic, privacy-safe PDF fixtures used by the public eval."""

from pathlib import Path
import re

import fitz


ROOT = Path(__file__).resolve().parent.parent
SOURCE_SVG = ROOT / "site" / "sample-handwriting.svg"
OUTPUT_DIR = ROOT / "data" / "public_samples"


def write_pdf(svg_text: str, output_path: Path, title: str) -> None:
    svg_doc = fitz.open(stream=svg_text.encode("utf-8"), filetype="svg")
    pdf_doc = fitz.open("pdf", svg_doc.convert_to_pdf())
    pdf_doc.set_metadata(
        {
            "title": title,
            "author": "",
            "subject": "Synthetic public Proofline evaluation fixture",
            "keywords": "synthetic, handwritten math, latex",
            "creator": "scripts/create_public_fixtures.py",
            "producer": "PyMuPDF",
        }
    )
    output_path.unlink(missing_ok=True)
    pdf_doc.save(output_path, garbage=4, deflate=True)
    pdf_doc.close()
    svg_doc.close()


def main() -> None:
    source = SOURCE_SVG.read_text(encoding="utf-8")
    # MuPDF does not implement SVG feDropShadow consistently and can render
    # the paper rectangle black. The shadow is decorative, so strip it before
    # producing the portable PDF fixtures.
    source = re.sub(r"<filter id=\"paperShadow\".*?</filter>", "", source, flags=re.DOTALL)
    source = source.replace(' filter="url(#paperShadow)"', "")
    source = source.replace(
        '    <rect x="72" y="38" width="656" height="524" rx="8" fill="url(#minorGrid)" opacity=".82" />\n',
        "",
    )
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    flawed = (
        source.replace("Sample handwritten derivative", "Synthetic derivative with intentional error")
        .replace("Blue ruled paper with a handwritten derivative that contains a visible plus two term.",
                 "Synthetic ruled paper with an intentional derivative error and no personal information.")
        .replace("Optimization — scratch work", "Synthetic derivative - preserved error")
        .replace("x = −1", "x = -1")
        .replace("check sign?", "intentional error")
    )
    write_pdf(
        flawed,
        OUTPUT_DIR / "synthetic-preserved-error.pdf",
        "Synthetic preserved-error fixture",
    )

    clean = (
        source.replace("Sample handwritten derivative", "Synthetic clean derivative")
        .replace("Blue ruled paper with a handwritten derivative that contains a visible plus two term.",
                 "Synthetic ruled paper with a self-consistent derivative and no personal information.")
        .replace("Optimization — scratch work", "Synthetic derivative - clean")
        .replace("f′(x) = 2x + 2", "f′(x) = 2x + 3")
        .replace("0 = 2x + 2", "0 = 2x + 3")
        .replace("x = −1", "x = -3/2")
        .replace("check sign?", "synthetic sample")
    )
    write_pdf(
        clean,
        OUTPUT_DIR / "synthetic-clean-derivative.pdf",
        "Synthetic clean derivative fixture",
    )

    print(f"Wrote 2 synthetic PDF fixtures to {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
