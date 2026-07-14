"""Parse math_to_latex.py markdown reports; render source PDFs/images to page images."""

import re
from pathlib import Path

# Matches "## <page label>" followed by a blank line and a ```latex ... ``` fence,
# as produced by math_to_latex.py's report writer.
REPORT_SECTION_RE = re.compile(r"^## (?P<label>.+?)\s*$\n\n```latex\n(?P<latex>.*?)\n```", re.MULTILINE | re.DOTALL)


def parse_report(report_path):
    """Return an ordered list of (page_label, latex) tuples from a math_to_latex.py-style report.

    Page order in the file is preserved and is the matching key used against rendered
    source images — page *labels* embed a filename stem that may not exactly match a
    freshly re-rendered image's filename, so order (not string matching) is the
    robust invariant.
    """
    text = Path(report_path).read_text()
    matches = REPORT_SECTION_RE.findall(text)
    if not matches:
        raise ValueError(
            f"No '## <page>' + ```latex fenced sections found in {report_path}. "
            "Expected the format produced by math_to_latex.py."
        )
    return matches


def render_source_to_images(source_path, out_dir, dpi=250, exclude_pages=None):
    """Render a PDF to page images (in page order), or pass through a list of image paths.

    Reuses math_to_latex.py's own PDF renderer so page ordering/DPI/exclusions match
    exactly what was used to produce the original transcription — if a page (e.g. a
    trailing code screenshot) was excluded when transcribing, it must be excluded
    here too or page indices between the transcription and the freshly rendered
    source images will drift out of alignment.
    """
    source_path = Path(source_path)
    if source_path.suffix.lower() == ".pdf":
        from math_to_latex import pdf_to_images  # scripts/ is on sys.path via the entrypoint script
        return pdf_to_images(source_path, Path(out_dir), dpi=dpi, exclude_pages=exclude_pages)
    return [source_path]
