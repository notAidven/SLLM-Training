#!/usr/bin/env python3
"""
generate_dashboard.py — build dashboard.html from every baseline_results/<model>/
golden_set_scorecard.json found, plus data/golden_set/manifest.json for labels.

Self-contained output: all data is embedded directly in the HTML (not fetched at
runtime), so dashboard.html opens correctly via file:// with no local server —
browsers block same-origin fetches from file:// pages, so a runtime-fetch design
would only work over http(s).

Re-run this after scoring a new checkpoint (baseline_results/<name>/golden_set_scorecard.json)
to pick it up automatically — no hardcoded model list to edit beyond BASELINE_MODEL_DIRS below.

Usage:
    python3 scripts/generate_dashboard.py
"""

import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
BASELINE_RESULTS_DIR = ROOT / "baseline_results"
MANIFEST_PATH = ROOT / "data" / "golden_set" / "manifest.json"
OUTPUT_PATH = ROOT / "dashboard.html"

# Known "before" models. Anything else found under baseline_results/ (e.g. a
# fine-tuned checkpoint's output directory) is grouped as "after" automatically.
BASELINE_MODEL_DIRS = {"gpt-5.5", "qwen3.5-0.8b"}

MODEL_LABELS = {
    "gpt-5.5": "GPT-5.5",
    "qwen3.5-0.8b": "Qwen3.5-0.8B",
}


def humanize_doc_id(doc_id, course):
    suffix = doc_id[len(course) + 1:] if doc_id.startswith(course + "_") else doc_id
    suffix = re.sub(r"([a-zA-Z])(\d)", r"\1 \2", suffix)
    suffix = suffix.replace("_", " ")
    return f"{course} {suffix.title()}"


def load_manifest_labels():
    manifest = json.loads(MANIFEST_PATH.read_text())
    labels = {}
    for entry in manifest["entries"]:
        labels[entry["id"]] = humanize_doc_id(entry["id"], entry["course"])
    return labels


def build_model_entry(model_dir, labels):
    scorecard_path = model_dir / "golden_set_scorecard.json"
    scorecard = json.loads(scorecard_path.read_text())
    model_id = model_dir.name

    clean_docs = []
    flawed_docs = []
    for e in scorecard["entries"]:
        label = labels.get(e["id"], e["id"])
        if e["category"] == "clean":
            defects = e.get("candidate_defects", [])
            image_defects = sum(1 for d in defects if d["mode"] == "image")
            consistency_defects = sum(1 for d in defects if d["mode"] == "consistency")
            clean_docs.append({
                "id": e["id"], "label": label, "passed": e.get("passed"),
                "image_defects": image_defects, "consistency_defects": consistency_defects,
                "defects": defects,
            })
        elif e["category"] == "flawed":
            flawed_docs.append({
                "id": e["id"], "label": label, "passed": e.get("passed"),
                "preserved": e.get("preserved", 0),
                "silently_fixed": e.get("silently_fixed", 0),
                "not_found": e.get("not_found", 0),
                "known_errors": e.get("known_error_results", []),
            })

    return {
        "id": model_id,
        "label": MODEL_LABELS.get(model_id, model_id),
        "group": "before" if model_id in BASELINE_MODEL_DIRS else "after",
        "judge_model": scorecard.get("judge_model"),
        "summary": scorecard["summary"],
        "clean_docs": clean_docs,
        "flawed_docs": flawed_docs,
    }


def discover_models(labels):
    models = []
    if not BASELINE_RESULTS_DIR.exists():
        return models
    for model_dir in sorted(BASELINE_RESULTS_DIR.iterdir()):
        scorecard_path = model_dir / "golden_set_scorecard.json"
        if model_dir.is_dir() and scorecard_path.exists():
            models.append(build_model_entry(model_dir, labels))
    return models


HTML_TEMPLATE = r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<title>Golden Set Dashboard — Handwritten Math &rarr; LaTeX</title>
<style>
  :root {
    --before: #4f6df5;
    --after: #16a34a;
    --preserved: #16a34a;
    --silently-fixed: #dc2626;
    --not-found: #9ca3af;
    --image-defect: #f59e0b;
    --consistency-defect: #7c3aed;
    --bg: #f8fafc;
    --card-bg: #ffffff;
    --border: #e2e8f0;
    --text: #1e293b;
    --text-dim: #64748b;
  }
  * { box-sizing: border-box; }
  body {
    font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Helvetica, Arial, sans-serif;
    background: var(--bg);
    color: var(--text);
    margin: 0;
    padding: 0 0 4rem 0;
  }
  header {
    background: var(--card-bg);
    border-bottom: 1px solid var(--border);
    padding: 1.5rem 2rem;
  }
  header h1 { margin: 0 0 0.25rem 0; font-size: 1.5rem; }
  header .meta { color: var(--text-dim); font-size: 0.9rem; }
  header .links { margin-top: 0.75rem; font-size: 0.9rem; }
  header .links a { margin-right: 1rem; }
  a { color: var(--before); text-decoration: none; }
  a:hover { text-decoration: underline; }
  main { max-width: 1100px; margin: 0 auto; padding: 0 2rem; }
  section { margin-top: 2.5rem; }
  section > h2 { font-size: 1.2rem; border-bottom: 1px solid var(--border); padding-bottom: 0.5rem; }
  .group-label { font-size: 0.8rem; text-transform: uppercase; letter-spacing: 0.05em; color: var(--text-dim); margin: 1.5rem 0 0.5rem 0; }
  .cards { display: flex; gap: 1rem; flex-wrap: wrap; }
  .card {
    background: var(--card-bg);
    border: 1px solid var(--border);
    border-radius: 10px;
    padding: 1rem 1.25rem;
    min-width: 240px;
    flex: 1;
  }
  .card.placeholder { color: var(--text-dim); font-style: italic; border-style: dashed; }
  .card h3 { margin: 0 0 0.5rem 0; font-size: 1.05rem; }
  .card .badge {
    display: inline-block;
    font-size: 0.7rem;
    font-weight: 600;
    padding: 0.1rem 0.5rem;
    border-radius: 999px;
    margin-left: 0.5rem;
    vertical-align: middle;
  }
  .badge.before { background: #e8ecfe; color: var(--before); }
  .badge.after { background: #dcfce7; color: var(--after); }
  .card dl { margin: 0.5rem 0 0 0; display: grid; grid-template-columns: auto 1fr; gap: 0.15rem 0.75rem; font-size: 0.9rem; }
  .card dt { color: var(--text-dim); }
  .card dd { margin: 0; text-align: right; font-weight: 600; }
  .doc-row { border: 1px solid var(--border); border-radius: 8px; margin-bottom: 0.6rem; background: var(--card-bg); overflow: hidden; }
  .doc-row-header { display: flex; align-items: center; gap: 0.75rem; padding: 0.6rem 0.9rem; cursor: pointer; }
  .doc-row-header:hover { background: #f1f5f9; }
  .doc-label { flex: 0 0 200px; font-size: 0.9rem; }
  .doc-model { flex: 0 0 130px; font-size: 0.8rem; color: var(--text-dim); }
  .bar-track { flex: 1; height: 18px; background: #f1f5f9; border-radius: 4px; overflow: hidden; display: flex; }
  .bar-seg { height: 100%; }
  .doc-count { flex: 0 0 40px; text-align: right; font-size: 0.8rem; color: var(--text-dim); }
  .caret { flex: 0 0 14px; color: var(--text-dim); font-size: 0.75rem; transition: transform 0.15s; }
  .doc-row.open .caret { transform: rotate(90deg); }
  .doc-detail { display: none; padding: 0 0.9rem 0.9rem 0.9rem; font-size: 0.85rem; }
  .doc-row.open .doc-detail { display: block; }
  .finding { padding: 0.4rem 0; border-top: 1px solid var(--border); }
  .finding:first-child { border-top: none; }
  .finding .tag { font-weight: 600; font-size: 0.75rem; text-transform: uppercase; margin-right: 0.4rem; }
  .tag.image { color: var(--image-defect); }
  .tag.consistency { color: var(--consistency-defect); }
  .tag.PRESERVED { color: var(--preserved); }
  .tag.SILENTLY_FIXED { color: var(--silently-fixed); }
  .tag.NOT_FOUND { color: var(--not-found); }
  .finding .loc { color: var(--text-dim); font-size: 0.8rem; }
  .legend { font-size: 0.8rem; color: var(--text-dim); margin-bottom: 0.75rem; }
  .legend span { display: inline-flex; align-items: center; margin-right: 1rem; }
  .legend .swatch { width: 10px; height: 10px; border-radius: 2px; margin-right: 0.35rem; display: inline-block; }
  .caveats { background: var(--card-bg); border: 1px solid var(--border); border-radius: 10px; padding: 1rem 1.25rem; }
  .caveats ol { margin: 0; padding-left: 1.25rem; }
  .caveats li { margin-bottom: 0.75rem; font-size: 0.9rem; }
  footer { max-width: 1100px; margin: 3rem auto 0; padding: 1rem 2rem; border-top: 1px solid var(--border); font-size: 0.85rem; color: var(--text-dim); }
  footer a { margin-right: 1rem; }
  .empty-note { color: var(--text-dim); font-size: 0.85rem; }
</style>
</head>
<body>

<header>
  <h1>Golden Set Dashboard</h1>
  <div class="meta">Handwritten math &rarr; LaTeX transcription &middot; Behavior Spec: cross-document consistency + error preservation &middot; Judge: <strong>__JUDGE_MODEL__</strong></div>
  <div class="links">
    <a href="BASELINE_SUMMARY.md">Baseline Summary</a>
    <a href="RUBRIC.md">Rubric</a>
    <a href="behavior_spec.md">Behavior Spec</a>
  </div>
</header>

<main>

<section id="summary">
  <h2>Summary</h2>
  <div class="group-label">Before (baseline)</div>
  <div class="cards" id="cards-before"></div>
  <div class="group-label">After (fine-tuned)</div>
  <div class="cards" id="cards-after"></div>
</section>

<section id="clean-set">
  <h2>Clean set &mdash; does the model introduce a NEW error or contradiction?</h2>
  <div class="legend">
    <span><span class="swatch" style="background:var(--image-defect)"></span>image-mode defect</span>
    <span><span class="swatch" style="background:var(--consistency-defect)"></span>consistency-mode defect</span>
  </div>
  <div id="clean-docs"></div>
</section>

<section id="flawed-set">
  <h2>Flawed set &mdash; does the model faithfully PRESERVE a known genuine error?</h2>
  <div class="legend">
    <span><span class="swatch" style="background:var(--preserved)"></span>preserved</span>
    <span><span class="swatch" style="background:var(--silently-fixed)"></span>silently fixed</span>
    <span><span class="swatch" style="background:var(--not-found)"></span>not found</span>
  </div>
  <div id="flawed-docs"></div>
</section>

<section id="caveats">
  <h2>Read before citing these numbers</h2>
  <div class="caveats">
    <ol>
      <li><strong>Judge independence.</strong> Both models are scored by <code>claude-opus-4-8</code>, a model unrelated to either candidate &mdash; not by GPT-5.5 judging itself, which was tried first and found to be a real bias risk, not a theoretical one.</li>
      <li><strong>Binary pass/fail is the wrong headline number for the clean set.</strong> A "pass" requires zero MAJOR_ERROR findings across an entire multi-page document &mdash; on dense real coursework, a single wrong digit anywhere fails the whole thing. The defect count/severity per document is the informative comparison, not the pass rate.</li>
      <li><strong>NOT_FOUND is not evidence of a pass on the flawed set.</strong> It means the known error's location wasn't found in the transcription at all &mdash; often because the transcription is too incomplete to evaluate, not because the model behaved well. Check the not-found count alongside any "passed" flawed document.</li>
      <li><strong>Judge reliability caveat carries forward.</strong> An earlier manual-verification pass found roughly 40-50% of dense-page MAJOR_ERROR findings didn't hold up under human review, under the earlier self-graded setup. Not re-measured at the same rigor under the current judge. Treat every count here as provisional pending spot-check.</li>
    </ol>
    <p>Full detail, including the verdict-ordering judge-prompt bug that was found and fixed mid-baseline: see <a href="BASELINE_SUMMARY.md">BASELINE_SUMMARY.md</a>.</p>
  </div>
</section>

</main>

<footer>
  <a href="RUBRIC.md">Rubric</a>
  <a href="behavior_spec.md">Behavior Spec</a>
  <a href="data/golden_set/known_errors_worksheet.md">Known-errors worksheet</a>
  <a href="data/golden_set/manifest.json">Golden set manifest</a>
  <a href="BASELINE_SUMMARY.md">Baseline Summary</a>
</footer>

<script>
const DATA = __DATA_JSON__;

function el(tag, attrs, children) {
  const e = document.createElement(tag);
  for (const [k, v] of Object.entries(attrs || {})) {
    if (k === "class") e.className = v;
    else if (k === "html") e.innerHTML = v;
    else e.setAttribute(k, v);
  }
  for (const c of (children || [])) e.appendChild(c);
  return e;
}
function text(tag, className, str) {
  const e = document.createElement(tag);
  if (className) e.className = className;
  e.textContent = str;
  return e;
}

function renderCards() {
  const before = document.getElementById("cards-before");
  const after = document.getElementById("cards-after");
  const beforeModels = DATA.models.filter(m => m.group === "before");
  const afterModels = DATA.models.filter(m => m.group === "after");

  if (!beforeModels.length) {
    before.appendChild(text("div", "empty-note", "No baseline results found under baseline_results/."));
  }
  for (const m of beforeModels) before.appendChild(renderCard(m));

  if (!afterModels.length) {
    const card = el("div", { class: "card placeholder" });
    card.appendChild(text("h3", null, "Awaiting fine-tuned checkpoint"));
    card.appendChild(text("div", null, "Run scripts/run_golden_set.py against the trained checkpoint into a new baseline_results/<name>/ directory, then re-run scripts/generate_dashboard.py to populate this."));
    after.appendChild(card);
  } else {
    for (const m of afterModels) after.appendChild(renderCard(m));
  }
}

function renderCard(m) {
  const card = el("div", { class: "card" });
  const h3 = text("h3", null, m.label);
  h3.appendChild(el("span", { class: "badge " + m.group, html: m.group === "before" ? "BEFORE" : "AFTER" }));
  card.appendChild(h3);

  const totalCleanDefects = m.clean_docs.reduce((s, d) => s + d.image_defects + d.consistency_defects, 0);
  const totalPreserved = m.flawed_docs.reduce((s, d) => s + d.preserved, 0);
  const totalSilentlyFixed = m.flawed_docs.reduce((s, d) => s + d.silently_fixed, 0);
  const totalNotFound = m.flawed_docs.reduce((s, d) => s + d.not_found, 0);
  const totalKnownErrors = totalPreserved + totalSilentlyFixed + totalNotFound;

  const dl = el("dl", {});
  const rows = [
    ["Clean docs passed", m.summary.clean_passed + "/" + m.summary.clean_total],
    ["Clean-set defects", String(totalCleanDefects)],
    ["Flawed docs passed", m.summary.flawed_passed + "/" + m.summary.flawed_total],
    ["Errors preserved", totalPreserved + "/" + totalKnownErrors],
    ["Silently fixed", String(totalSilentlyFixed)],
    ["Not found", String(totalNotFound)],
  ];
  for (const [k, v] of rows) {
    dl.appendChild(text("dt", null, k));
    dl.appendChild(text("dd", null, v));
  }
  card.appendChild(dl);
  return card;
}

function renderCleanDocs() {
  const container = document.getElementById("clean-docs");
  const docIds = [...new Set(DATA.models.flatMap(m => m.clean_docs.map(d => d.id)))];
  const maxDefects = Math.max(1, ...DATA.models.flatMap(m => m.clean_docs.map(d => d.image_defects + d.consistency_defects)));

  for (const docId of docIds) {
    for (const m of DATA.models) {
      const doc = m.clean_docs.find(d => d.id === docId);
      if (!doc) continue;
      container.appendChild(renderCleanRow(m, doc, maxDefects));
    }
  }
}

function renderCleanRow(model, doc, maxDefects) {
  const total = doc.image_defects + doc.consistency_defects;
  const row = el("div", { class: "doc-row" });
  const header = el("div", { class: "doc-row-header" });
  header.appendChild(text("div", "caret", "▶"));
  header.appendChild(text("div", "doc-label", doc.label));
  header.appendChild(text("div", "doc-model", model.label));

  const track = el("div", { class: "bar-track" });
  if (total > 0) {
    const imgPct = (doc.image_defects / maxDefects) * 100;
    const consPct = (doc.consistency_defects / maxDefects) * 100;
    if (imgPct > 0) track.appendChild(el("div", { class: "bar-seg", style: `width:${imgPct}%;background:var(--image-defect)` }));
    if (consPct > 0) track.appendChild(el("div", { class: "bar-seg", style: `width:${consPct}%;background:var(--consistency-defect)` }));
  }
  header.appendChild(track);
  header.appendChild(text("div", "doc-count", String(total)));
  row.appendChild(header);

  const detail = el("div", { class: "doc-detail" });
  if (doc.defects.length === 0) {
    detail.appendChild(text("div", "finding", "No defects found."));
  } else {
    for (const d of doc.defects) {
      const f = el("div", { class: "finding" });
      const tag = text("span", "tag " + d.mode, d.mode);
      f.appendChild(tag);
      f.appendChild(text("span", "loc", JSON.stringify(d.location)));
      f.appendChild(document.createElement("br"));
      f.appendChild(text("span", null, d.reasoning));
      detail.appendChild(f);
    }
  }
  row.appendChild(detail);

  header.addEventListener("click", () => row.classList.toggle("open"));
  return row;
}

function renderFlawedDocs() {
  const container = document.getElementById("flawed-docs");
  const docIds = [...new Set(DATA.models.flatMap(m => m.flawed_docs.map(d => d.id)))];

  for (const docId of docIds) {
    for (const m of DATA.models) {
      const doc = m.flawed_docs.find(d => d.id === docId);
      if (!doc) continue;
      container.appendChild(renderFlawedRow(m, doc));
    }
  }
}

function renderFlawedRow(model, doc) {
  const total = Math.max(1, doc.preserved + doc.silently_fixed + doc.not_found);
  const row = el("div", { class: "doc-row" });
  const header = el("div", { class: "doc-row-header" });
  header.appendChild(text("div", "caret", "▶"));
  header.appendChild(text("div", "doc-label", doc.label));
  header.appendChild(text("div", "doc-model", model.label));

  const track = el("div", { class: "bar-track" });
  const segs = [
    [doc.preserved, "var(--preserved)"],
    [doc.silently_fixed, "var(--silently-fixed)"],
    [doc.not_found, "var(--not-found)"],
  ];
  for (const [count, color] of segs) {
    if (count > 0) track.appendChild(el("div", { class: "bar-seg", style: `width:${(count / total) * 100}%;background:${color}` }));
  }
  header.appendChild(track);
  header.appendChild(text("div", "doc-count", doc.preserved + "/" + (doc.preserved + doc.silently_fixed + doc.not_found)));
  row.appendChild(header);

  const detail = el("div", { class: "doc-detail" });
  for (const ke of doc.known_errors) {
    const f = el("div", { class: "finding" });
    f.appendChild(text("span", "tag " + ke.status, ke.status));
    f.appendChild(text("span", "loc", ke.problem + " — " + ke.rubric_reason));
    detail.appendChild(f);
  }
  row.appendChild(detail);

  header.addEventListener("click", () => row.classList.toggle("open"));
  return row;
}

renderCards();
renderCleanDocs();
renderFlawedDocs();
</script>

</body>
</html>
"""


def main():
    labels = load_manifest_labels()
    models = discover_models(labels)
    judge_models = {m["judge_model"] for m in models if m["judge_model"]}
    judge_model_display = ", ".join(sorted(judge_models)) if judge_models else "unknown"

    data = {"models": models}
    html = HTML_TEMPLATE.replace("__DATA_JSON__", json.dumps(data))
    html = html.replace("__JUDGE_MODEL__", judge_model_display)

    OUTPUT_PATH.write_text(html)
    print(f"Found {len(models)} model(s): {', '.join(m['id'] for m in models)}")
    print(f"Wrote {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
