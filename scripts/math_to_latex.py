#!/usr/bin/env python3
"""
Handwritten Math to LaTeX — tests whether a vision-capable model can
accurately transcribe REAL handwritten math work into LaTeX. Uses your own
photos or PDFs, so there's zero risk of the model having memorized a
benchmark image.

Setup:
    pip install openai pymupdf
    export OPENAI_API_KEY="..."

Usage:
    python3 math_to_latex.py page1.jpg page2.jpg --model gpt-5.5
    python3 math_to_latex.py homework.pdf --model gpt-5.5
    python3 math_to_latex.py homework.pdf --output my_results.md

    # Point at a local OpenAI-compatible server (e.g. `transformers serve`)
    python3 math_to_latex.py homework.pdf --model Qwen/Qwen3.5-0.8B \\
        --base-url http://localhost:8000/v1 --api-key EMPTY
"""

import argparse
import base64
import os
import sys
import tempfile
from pathlib import Path

try:
    import openai
except ImportError:
    print("Missing dependency. Install with: pip install openai")
    sys.exit(1)

sys.path.insert(0, str(Path(__file__).resolve().parent))
from latex_eval import judge as judge_lib

SYSTEM_PROMPT = """You are transcribing a photo of handwritten mathematical work into LaTeX.

Rules:
- Transcribe everything written in the image, preserving the exact structure and order, top to bottom.
- Use proper LaTeX math notation: fractions as \\frac{}{}, exponents as ^{}, subscripts as _{}, square roots as \\sqrt{}, integrals as \\int, summations as \\sum, Greek letters spelled out (\\alpha, \\beta, etc.).
- Use an align* environment if there are multiple lines or steps that should line up.
- If any part is illegible or you are genuinely unsure, write [ILLEGIBLE: your best guess] inline rather than silently guessing and moving on.
- If work is crossed out, transcribe it anyway wrapped as [CROSSED OUT: ...] rather than skipping it.
- Output ONLY the LaTeX code. No explanation before or after, no markdown code fences."""


def get_mime_type(path: Path) -> str:
    ext = path.suffix.lower()
    mapping = {
        ".jpg": "image/jpeg", ".jpeg": "image/jpeg",
        ".png": "image/png", ".gif": "image/gif", ".webp": "image/webp",
    }
    if ext not in mapping:
        raise ValueError(f"Unsupported image type: {ext}. Use jpg, jpeg, png, gif, or webp.")
    return mapping[ext]


def encode_image(path: Path) -> str:
    with open(path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")


def pdf_to_images(pdf_path: Path, out_dir: Path, dpi: int = 250, exclude_pages=None) -> list:
    """Render each page of a PDF to a PNG image and return their paths.

    exclude_pages: optional set/list of 1-indexed page numbers to skip (e.g. a
    trailing page that's a screenshot of code rather than handwritten math).
    """
    try:
        import fitz  # PyMuPDF
    except ImportError:
        print("PDF support needs PyMuPDF. Install with: pip install pymupdf")
        sys.exit(1)

    exclude_pages = set(exclude_pages or [])
    doc = fitz.open(pdf_path)
    zoom = dpi / 72  # PDF default is 72 DPI
    matrix = fitz.Matrix(zoom, zoom)
    image_paths = []
    skipped = []
    for i, page in enumerate(doc):
        page_num = i + 1
        if page_num in exclude_pages:
            skipped.append(page_num)
            continue
        pix = page.get_pixmap(matrix=matrix)
        out_path = out_dir / f"{pdf_path.stem}_page{page_num}.png"
        pix.save(str(out_path))
        image_paths.append(out_path)
    doc.close()
    skip_note = f" (skipped page(s) {sorted(skipped)})" if skipped else ""
    print(f"  Converted {pdf_path.name} -> {len(image_paths)} page image(s){skip_note}")
    return image_paths


def _parse_page_list(spec):
    """Parse a comma-separated 1-indexed page list like '6' or '1,7' into a set of ints."""
    if not spec:
        return set()
    return {int(p.strip()) for p in spec.split(",") if p.strip()}


def transcribe_image(client, model: str, image_path: Path) -> str:
    mime_type = get_mime_type(image_path)
    b64_image = encode_image(image_path)

    user_content = [
        {"type": "text", "text": "Transcribe the handwritten math in this image into LaTeX."},
        {
            "type": "image_url",
            "image_url": {"url": f"data:{mime_type};base64,{b64_image}"},
        },
    ]
    return judge_lib.call_judge(client, model, SYSTEM_PROMPT, user_content)


def main():
    parser = argparse.ArgumentParser(
        description="Transcribe handwritten math images or PDFs into LaTeX using a vision-capable model."
    )
    parser.add_argument("files", nargs="+", help="Path(s) to image file(s) and/or PDF(s) of your handwritten math work")
    parser.add_argument("--model", default="gpt-5.5", help="Vision-capable model to use (e.g. gpt-5.5, Qwen/Qwen3.5-0.8B)")
    parser.add_argument("--output", default="latex_results.md", help="Output markdown file")
    parser.add_argument("--dpi", type=int, default=250, help="Resolution for PDF page rendering (default 250)")
    parser.add_argument("--base-url", default=None, help="OpenAI-compatible base URL (e.g. http://localhost:8000/v1 for a local transformers-serve endpoint). Defaults to the real OpenAI API.")
    parser.add_argument("--api-key", default=None, help="API key. Defaults to the OPENAI_API_KEY environment variable. Use 'EMPTY' for local servers that don't check it.")
    parser.add_argument("--exclude-pages", default=None, help="Comma-separated 1-indexed page number(s) to skip when converting a PDF (e.g. a trailing page that's a code screenshot, not handwritten math). Applies to every PDF passed in this invocation, so pass one PDF per run when using this.")
    args = parser.parse_args()
    exclude_pages = _parse_page_list(args.exclude_pages)

    api_key = args.api_key or os.environ.get("OPENAI_API_KEY")
    if not api_key:
        print("No API key found. Set OPENAI_API_KEY in your environment or pass --api-key.")
        sys.exit(1)
    client = judge_lib.make_client(base_url=args.base_url, api_key=api_key)

    with tempfile.TemporaryDirectory() as tmp_dir_str:
        tmp_dir = Path(tmp_dir_str)
        image_paths = []
        for file_str in args.files:
            file_path = Path(file_str)
            if not file_path.exists():
                print(f"WARNING: {file_path} not found, skipping.")
                continue
            if file_path.suffix.lower() == ".pdf":
                print(f"Converting PDF: {file_path.name} ...")
                image_paths.extend(pdf_to_images(file_path, tmp_dir, dpi=args.dpi, exclude_pages=exclude_pages))
            else:
                image_paths.append(file_path)

        results = []
        for img_path in image_paths:
            print(f"Transcribing: {img_path.name} ...")
            try:
                latex = transcribe_image(client, args.model, img_path)
            except Exception as e:
                latex = f"[ERROR: {e}]"
            results.append((img_path.name, latex))
            print(f"--- Result for {img_path.name} ---")
            print(latex)
            print()

        with open(args.output, "w") as f:
            f.write("# Handwritten Math -> LaTeX Transcription Results\n\n")
            f.write(f"Model: {args.model}\n\n")
            f.write(
                "For each page: open the original image/PDF page side-by-side with "
                "this file and check line by line. Note every misread symbol, dropped "
                "step, or structural error (wrong fraction grouping, missed exponent, "
                "flattened multi-line work, misplaced matrix entry, etc.) — not just "
                "whether the final answer happens to match.\n\n---\n\n"
            )
            for name, latex in results:
                f.write(f"## {name}\n\n")
                f.write("```latex\n")
                f.write(latex)
                f.write("\n```\n\n")
                f.write("**Errors found (fill in while comparing to the original page):**\n\n- \n\n---\n\n")

    print(f"\nAll results saved to {args.output}")
    print("Open your original pages side-by-side with that file and check line-by-line for errors.")


if __name__ == "__main__":
    main()
