"""Cross-document consistency checking: extract every named-quantity relation across
a full multi-page transcription and flag contradictions between distant restatements
of the same fact — this project's actual differentiating eval metric.

Exposes extract_claims/cluster_claims/check_contradictions as plain, importable
functions so a future synthetic-error injector can reuse them directly without
redesign (given a clean composed document, pick a cluster, flip one member's
relation at its recorded location).
"""

import re

from . import io as latex_io
from . import judge as judge_lib
from .report import Finding, Report

CLAIM_RE = re.compile(
    r"CLAIM:\s*id=(?P<id>\S+?)\s*\|\s*page=(?P<page>.*?)\s*\|\s*subject=(?P<subject>.*?)\s*\|\s*"
    r"lhs=(?P<lhs>.*?)\s*\|\s*relation=(?P<relation>\\geq|\\ge|>=|\\leq|\\le|<=|\\neq|\\ne|!=|=|>|<)\s*\|\s*"
    r"rhs=(?P<rhs>.*?)\s*\|\s*raw=(?P<raw>.*)"
)

_REL_NORMALIZE = {
    "\\geq": ">=", "\\ge": ">=", ">=": ">=",
    "\\leq": "<=", "\\le": "<=", "<=": "<=",
    "\\neq": "!=", "\\ne": "!=", "!=": "!=",
    "=": "=", ">": ">", "<": "<",
}
_FLIP = {">=": "<=", "<=": ">=", ">": "<", "<": ">", "=": "=", "!=": "!="}

CLAIM_EXTRACTION_SYSTEM_PROMPT = """You are analyzing a full multi-page LaTeX transcription of a handwritten mathematical derivation. Page boundaries are marked with lines like "%% PAGE: <label> %%".

Your job: extract every assertion in the document that states a relation (an equation, inequality, or bound) about a PERSISTENT NAMED QUANTITY — a variable, expression, probability, or expectation that is referenced more than once across the document (e.g. "P(X >= n/10)", "E[Y]", "Var(X)"). Do NOT extract routine one-off algebraic substitution steps that are never referenced again.

For every such relation, wherever it appears (including every restatement, re-derivation, or reuse of it later in the document), output one line in this exact format:
CLAIM: id=<unique short id like c1, c2, ...> | page=<page label from the nearest %% PAGE %% marker above it> | subject=<short human-readable tag for what persistent quantity/fact this is about> | lhs=<left-hand side LaTeX> | relation=<one of: >=, <=, =, >, <, !=> | rhs=<right-hand side LaTeX> | raw=<the literal line(s) this came from, condensed to one line>

Use the SAME subject tag for every occurrence of the same underlying fact, even if restated in a different algebraic form later — this is critical, since the entire point is tracking one fact across the whole document.

Output ONLY CLAIM lines, no other text."""

CLUSTER_REVIEW_SYSTEM_PROMPT = """You are reviewing a proposed grouping of claims from a math derivation, where each group is supposed to contain every restatement of ONE persistent fact.

You will be given the groups (by subject tag) with each member's lhs/relation/rhs. Two claims belong in the same group only if they assert the same underlying relationship (algebraically, allowing rearrangement), not merely because they involve the same variable name.

Output zero or more directives, one per line:
MERGE: <tag1>,<tag2>          (these two groups are actually the same fact and should be combined)
SPLIT: <tag> ids=<id1>,<id2>  (these listed ids do NOT belong with the rest of this group and should be pulled out into their own group)

If no changes are needed, output exactly:
NO_CHANGES

Output ONLY directive lines or NO_CHANGES, no other text."""

CONTRADICTION_JUDGE_SYSTEM_PROMPT = """You are checking whether two mentions of what is claimed to be the SAME underlying mathematical fact, appearing at different points in a long derivation, actually contradict each other.

IMPORTANT: two claims with opposite-direction relations (e.g. A >= B and A <= B) are NOT automatically a contradiction — they can be two independently-valid bounds later combined in a legitimate squeeze/sandwich argument. A genuine contradiction is specifically: the SAME fact, restated later as if unchanged, but with the relation direction or equality flipped, in a way inconsistent with the derivation's own logic (not part of a sandwich/combination step).

You will be given both claims with their location and surrounding derivation context. First, write one or two sentences of reasoning working through whether they agree. THEN, on its own final line, commit to a verdict that matches the conclusion you just reasoned to:
VERDICT: <SAME_CLAIM_CONSISTENT|SAME_CLAIM_CONTRADICTION|DIFFERENT_CLAIMS_NOT_COMPARABLE>

- SAME_CLAIM_CONSISTENT: both claims assert the same fact and agree (including if algebraically equivalent after rearrangement).
- SAME_CLAIM_CONTRADICTION: both claims assert the same fact but disagree (e.g. flipped inequality direction with no legitimate sandwich/combination logic shown).
- DIFFERENT_CLAIMS_NOT_COMPARABLE: on inspection these are not actually the same underlying fact (a clustering error) — do not judge them against each other.

Your verdict MUST match the conclusion of your own reasoning above it — do not reason your way to "they agree" and then output SAME_CLAIM_CONTRADICTION, or vice versa."""


def _normalize(s):
    s = re.sub(r"\\(left|right|!|,|;|:)", "", s)
    return re.sub(r"\s+", "", s)


def _dedup_key(claim):
    return (_normalize(claim["lhs"]), claim["relation"], _normalize(claim["rhs"]))


def _flipped_key(claim):
    return (_normalize(claim["rhs"]), _FLIP[claim["relation"]], _normalize(claim["lhs"]))


def extract_claims(transcription_path, extractor_model, client):
    """Single whole-document pass: emit every named-quantity relation as a structured claim."""
    pages = latex_io.parse_report(transcription_path)
    full_doc = "\n\n".join(f"%% PAGE: {label} %%\n{latex}" for label, latex in pages)

    judge_text = judge_lib.call_judge(client, extractor_model, CLAIM_EXTRACTION_SYSTEM_PROMPT, full_doc)
    claims = []
    for m in CLAIM_RE.finditer(judge_text):
        claims.append({
            "id": m.group("id"),
            "page": m.group("page").strip(),
            "subject": m.group("subject").strip(),
            "lhs": m.group("lhs").strip(),
            "relation": _REL_NORMALIZE.get(m.group("relation").strip(), m.group("relation").strip()),
            "rhs": m.group("rhs").strip(),
            "raw": m.group("raw").strip(),
        })
    return claims, full_doc


def _dedup_within_group(claims_in_group):
    """Collapse literal or sides-swapped-equivalent duplicates within one subject group
    with zero LLM calls (A>=B and B<=A are the same statement, not a contradiction)."""
    seen = {}
    for c in claims_in_group:
        key = _dedup_key(c)
        flipped = _flipped_key(c)
        match_key = key if key in seen else (flipped if flipped in seen else None)
        if match_key is not None:
            seen[match_key].setdefault("duplicate_ids", []).append(c["id"])
        else:
            c = dict(c)
            c["duplicate_ids"] = []
            seen[key] = c
    return list(seen.values())


def _union_find(claims):
    parent = {c["id"]: c["id"] for c in claims}

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(x, y):
        rx, ry = find(x), find(y)
        if rx != ry:
            parent[ry] = rx

    return find, union


def cluster_claims(claims, extractor_model, client):
    """Union claims sharing EITHER the extractor-assigned subject tag OR an identical
    normalized left-hand side, then run one LLM review call to merge/split groups for
    the harder case (same fact restated with a genuinely different LHS, e.g. after
    algebraic rearrangement). Avoids all-pairs comparison: only within-cluster pairs
    get judged downstream.

    The LHS-match union matters because the extractor can (and did, in validation
    against the known pset8 contradiction) assign two mentions of the exact same
    left-hand expression different subject tags — e.g. describing one occurrence as
    a "lower bound" and the later, flipped-direction occurrence as an "upper bound"
    on the same quantity, rather than recognizing them as the same restated claim.
    Relying on subject tags alone would silently miss exactly this case, which is
    close to the project's own target failure mode. An identical LHS is a strong,
    free (zero-LLM-call) signal that two claims are talking about the same thing,
    independent of how the extractor happened to label them.
    """
    if not claims:
        return {}

    find, union = _union_find(claims)
    by_subject, by_lhs = {}, {}
    for c in claims:
        by_subject.setdefault(c["subject"], []).append(c["id"])
        by_lhs.setdefault(_normalize(c["lhs"]), []).append(c["id"])
    for id_group in list(by_subject.values()) + list(by_lhs.values()):
        for other_id in id_group[1:]:
            union(id_group[0], other_id)

    raw_groups = {}
    for c in claims:
        raw_groups.setdefault(find(c["id"]), []).append(c)

    def _label(members):
        from collections import Counter
        return Counter(m["subject"] for m in members).most_common(1)[0][0]

    groups = {_label(members): members for members in raw_groups.values()}
    deduped_groups = {tag: _dedup_within_group(members) for tag, members in groups.items()}

    if len(deduped_groups) <= 1:
        return deduped_groups

    summary = "\n".join(
        f"tag={tag} | id={m['id']} | lhs={m['lhs']} | relation={m['relation']} | rhs={m['rhs']}"
        for tag, members in deduped_groups.items()
        for m in members
    )
    review_text = judge_lib.call_judge(client, extractor_model, CLUSTER_REVIEW_SYSTEM_PROMPT, summary)

    for line in review_text.splitlines():
        line = line.strip()
        merge_m = re.match(r"MERGE:\s*(\S+?),(\S+)", line)
        split_m = re.match(r"SPLIT:\s*(\S+?)\s+ids=(.+)", line)
        if merge_m:
            tag1, tag2 = merge_m.group(1), merge_m.group(2)
            if tag1 in deduped_groups and tag2 in deduped_groups and tag1 != tag2:
                deduped_groups[tag1].extend(deduped_groups.pop(tag2))
        elif split_m:
            tag, ids_str = split_m.group(1), split_m.group(2)
            ids = {i.strip() for i in ids_str.split(",")}
            if tag in deduped_groups:
                group = deduped_groups[tag]
                remaining_lhs = {_normalize(c["lhs"]) for c in group if c["id"] not in ids}
                # Never let the semantic reviewer split apart claims that share an
                # EXACT left-hand side — that deterministic signal is stronger than
                # its judgment call, and overriding it is exactly how the known
                # pset8 contradiction (same LHS, flipped relation, tagged "lower
                # bound" vs "upper bound" by the extractor) got missed in practice.
                safe_pulled_ids = {c["id"] for c in group if c["id"] in ids and _normalize(c["lhs"]) not in remaining_lhs}
                if safe_pulled_ids and len(safe_pulled_ids) < len(group):
                    pulled = [c for c in group if c["id"] in safe_pulled_ids]
                    deduped_groups[tag] = [c for c in group if c["id"] not in safe_pulled_ids]
                    deduped_groups[f"{tag}__split__{'_'.join(sorted(safe_pulled_ids))}"] = pulled

    return deduped_groups


def _find_context(full_doc, claim, window=400):
    """Best-effort: locate the claim's raw text in the full document for surrounding
    derivation context (needed to distinguish a real contradiction from a legitimate
    sandwich/squeeze argument). Falls back to the bare claim if not found verbatim
    (the extractor may have condensed whitespace when copying `raw`)."""
    for needle in (claim["raw"][:60], _normalize(claim["lhs"])):
        if not needle:
            continue
        idx = full_doc.find(needle)
        if idx != -1:
            start, end = max(0, idx - window), min(len(full_doc), idx + len(needle) + window)
            return full_doc[start:end]
    return f"{claim['lhs']} {claim['relation']} {claim['rhs']}"


def check_contradictions(clusters, full_doc, judge_model, client):
    findings = []
    for tag, members in clusters.items():
        if len(members) < 2:
            # Singleton — nothing to compare, but recorded explicitly so it can be
            # spot-checked: a missed merge here silently undercounts contradictions,
            # which is the failure mode that matters more than a false-positive flag.
            if members:
                findings.append(Finding(
                    id=f"singleton::{tag}::{members[0]['id']}",
                    mode="consistency",
                    verdict="EXACT",
                    method="singleton_cluster",
                    location={"subject": tag, "claim_ids": [m["id"] for m in members]},
                    candidate_snippet=members[0]["raw"],
                    reasoning="Only one distinct claim found for this subject — no restatement to compare against.",
                ))
            continue

        for i in range(len(members)):
            for j in range(i + 1, len(members)):
                a, b = members[i], members[j]
                user_content = (
                    f"Claim A (page {a['page']}): {a['lhs']} {a['relation']} {a['rhs']}\n"
                    f"Context around Claim A:\n{_find_context(full_doc, a)}\n\n"
                    f"Claim B (page {b['page']}): {b['lhs']} {b['relation']} {b['rhs']}\n"
                    f"Context around Claim B:\n{_find_context(full_doc, b)}"
                )
                judge_text = judge_lib.call_judge(client, judge_model, CONTRADICTION_JUDGE_SYSTEM_PROMPT, user_content)
                m = re.search(
                    r"VERDICT:\s*(SAME_CLAIM_CONSISTENT|SAME_CLAIM_CONTRADICTION|DIFFERENT_CLAIMS_NOT_COMPARABLE)",
                    judge_text,
                )
                pair_id = f"{tag}::{a['id']}_vs_{b['id']}"
                snippet = f"{a['lhs']} {a['relation']} {a['rhs']}  vs  {b['lhs']} {b['relation']} {b['rhs']}"
                location = {"subject": tag, "claim_a": a["id"], "claim_b": b["id"], "page_a": a["page"], "page_b": b["page"]}
                if not m:
                    findings.append(Finding(
                        id=pair_id, mode="consistency", verdict="MAJOR_ERROR", method="llm_judge",
                        location=location, candidate_snippet=snippet,
                        reasoning=f"Judge output did not match expected VERDICT format: {judge_text[:300]!r}",
                    ))
                    continue
                label = m.group(1)
                reasoning_text = judge_text[:m.start()].strip() or judge_text[m.end():].strip()
                verdict_map = {
                    "SAME_CLAIM_CONSISTENT": "EXACT",
                    "SAME_CLAIM_CONTRADICTION": "MAJOR_ERROR",
                    "DIFFERENT_CLAIMS_NOT_COMPARABLE": "MINOR_ERROR",
                }
                method = "llm_judge" if label != "DIFFERENT_CLAIMS_NOT_COMPARABLE" else "clustering_error"
                findings.append(Finding(
                    id=pair_id, mode="consistency", verdict=verdict_map[label], method=method,
                    location=location, candidate_snippet=snippet, reasoning=reasoning_text,
                ))
    return findings


def run(transcription_path, extractor_model, judge_model, client):
    claims, full_doc = extract_claims(transcription_path, extractor_model, client)
    if not claims:
        return Report(
            mode="consistency",
            metadata={
                "extractor_model": extractor_model, "judge_model": judge_model,
                "transcription": str(transcription_path), "claims_extracted": 0,
            },
            findings=[],
        )
    clusters = cluster_claims(claims, extractor_model, client)
    findings = check_contradictions(clusters, full_doc, judge_model, client)
    return Report(
        mode="consistency",
        metadata={
            "extractor_model": extractor_model, "judge_model": judge_model,
            "transcription": str(transcription_path),
            "claims_extracted": len(claims), "clusters": len(clusters),
            # Persisted for debuggability: without these, a singleton or an
            # under-merged cluster can't be spot-checked after the fact — only the
            # final findings would be visible, not what the extractor/clusterer saw.
            "claims": claims,
            "cluster_assignment": {tag: [m["id"] for m in members] for tag, members in clusters.items()},
        },
        findings=findings,
    )
