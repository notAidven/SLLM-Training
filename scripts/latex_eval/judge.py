"""OpenAI client construction, judge prompt calls, and VERDICT/FINDING line parsing.

Uses plain parseable text lines (FINDING: .../VERDICT: ...) rather than a JSON
response schema, matching this project's existing convention (see
run_physics_battery_openai.py's FINAL VERDICT: line) — lower parsing risk than
asking a model to emit valid JSON for open-ended reasoning output.
"""

import os
import re
import time

import openai

FINDING_RE = re.compile(
    r"FINDING:\s*location=(?P<location>.*?)\s*\|\s*reasoning=(?P<reasoning>.*?)\s*\|\s*verdict=(?P<verdict>EXACT|MINOR_ERROR|MAJOR_ERROR)"
)
VERDICT_RE = re.compile(r"VERDICT:\s*(EXACT|MINOR_ERROR|MAJOR_ERROR)")

# A gateway can hold a connection ESTABLISHED for hours without ever responding or
# erroring (observed directly: two multi-hour runs stuck on a single request with a
# live-but-silent TCP connection, well past the SDK's default ~600s read timeout not
# firing). An explicit, shorter timeout plus a bounded retry means a single dead
# request fails fast and gets a fresh connection, instead of hanging the whole job.
# 90s was tried first but is too tight for legitimately slow (not hung) calls — e.g.
# a single whole-document claim-extraction call over a long transcript can genuinely
# take over 90s to finish generating; that's normal latency, not a hang. 180s gives
# real long calls room to finish while still escaping multi-hour hangs within minutes.
DEFAULT_TIMEOUT_SECONDS = 180
MAX_ATTEMPTS = 3
RETRY_BACKOFF_SECONDS = 5


def make_client(base_url=None, api_key=None, timeout=DEFAULT_TIMEOUT_SECONDS):
    key = api_key or os.environ.get("OPENAI_API_KEY")
    if not key:
        raise RuntimeError("No API key found. Set OPENAI_API_KEY in your environment or pass --judge-api-key.")
    return openai.OpenAI(base_url=base_url, api_key=key, timeout=timeout, max_retries=0)


def call_judge(client, model, system_prompt, user_content, max_attempts=MAX_ATTEMPTS):
    """user_content: a str, or a list of content blocks (text + image_url) for vision calls.

    Retries on timeout/connection errors with a fresh request (new connection) rather
    than letting the SDK's own retry logic reuse a possibly-stuck connection. Raises
    the last error if every attempt fails, so the caller can decide how to record a
    genuinely unreachable endpoint rather than hanging indefinitely.
    """
    last_error = None
    for attempt in range(1, max_attempts + 1):
        try:
            response = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_content},
                ],
            )
            return response.choices[0].message.content or ""
        except (openai.APITimeoutError, openai.APIConnectionError, openai.InternalServerError) as e:
            # InternalServerError covers 5xx (e.g. a transient 502 from the gateway/
            # upstream provider) — a transient server-side failure, worth retrying,
            # unlike a 4xx (bad request/auth) which retrying would never fix.
            last_error = e
            print(f"    [judge call attempt {attempt}/{max_attempts} failed: {type(e).__name__}: {e}]")
            if attempt < max_attempts:
                time.sleep(RETRY_BACKOFF_SECONDS * attempt)
    raise last_error


def parse_findings(judge_text):
    """Parse zero or more `FINDING: VERDICT | location=... | reasoning=...` lines."""
    results = []
    for m in FINDING_RE.finditer(judge_text):
        results.append({
            "verdict": m.group("verdict"),
            "location": m.group("location").strip(),
            "reasoning": m.group("reasoning").strip(),
        })
    return results


def parse_verdict(judge_text):
    """Parse a single `VERDICT: ...` line. Returns (verdict, reasoning_text_before_it)."""
    m = VERDICT_RE.search(judge_text)
    if not m:
        return None, judge_text.strip()
    return m.group(1), judge_text[: m.start()].strip()
