"""Answer solver.

Strategy for SPEED:
  1. If an LLM API key is configured, send ONLY the extracted text (never the
     image) to Gemini or DeepSeek with a tiny prompt and max_tokens=16.
     Text round-trips are ~0.3-1.5s, image uploads are 5-15s+ - that's why we
     never upload pictures.
  2. If no key / API fails / times out, fall back to a local heuristic so the
     app still answers instantly and offline.
Results are cached by question hash so re-scanning the same screen is free.
"""

from __future__ import annotations

import re
import time
from dataclasses import dataclass

import requests

from mcq_parser import MCQ

LETTER_RE = re.compile(r"\b([A-E])\b")

# 3.x flash models first: the API told us 2.x models are "no longer available
# to new users" and to use gemini-3.5-flash-lite / gemini-3.8-flash instead.
GEMINI_MODEL_PREFS = (
    "gemini-3.5-flash-lite", "gemini-3.8-flash", "gemini-3.5-flash",
    "gemini-2.5-flash-lite", "gemini-2.5-flash", "gemini-2.0-flash-lite",
    "gemini-2.0-flash", "gemini-1.5-flash-latest", "gemini-1.5-flash",
)

# model families that can't answer text MCQs via generateContent
NON_TEXT_MARKERS = ("tts", "imagen", "veo", "embed", "aqa", "native-audio")


@dataclass
class Answer:
    letter: str | None
    text: str
    source: str          # "gemini" | "deepseek" | "local" | "none"
    latency_ms: int
    raw: str = ""


# --------------------------------------------------------------------- #
# Local heuristic fallback (offline, ~0 ms)
# --------------------------------------------------------------------- #
NEGATION_RE = re.compile(r"\b(not|except|least|never|incorrect|false|wrong)\b", re.I)


def solve_local(mcq: MCQ) -> Answer:
    """Offline heuristic. Fast (~0 ms) but far from perfect - the LLM is the
    primary solver. Handles negation questions ("NOT a mammal") by elimination:
    the answer is the option LEAST related to the question's words."""
    start = time.perf_counter()
    q = mcq.question.lower()
    q_words = set(re.findall(r"[a-z]+", q))
    negated = bool(NEGATION_RE.search(mcq.question))

    best, best_score = None, None
    for letter, opt in mcq.options.items():
        o_words = set(re.findall(r"[a-z]+", opt.lower()))
        if not o_words:
            continue
        overlap = len(q_words & o_words)
        rare = len([w for w in (q_words & o_words) if len(w) > 5])
        if negated:
            score = -(rare * 2 + overlap) + len(opt) / 500.0
        else:
            score = rare * 2 - overlap * 0.5 + len(opt) / 200.0
        if best_score is None or score > best_score:
            best, best_score = letter, score
    latency = int((time.perf_counter() - start) * 1000)
    if best is None:
        return Answer(None, "could not parse options", "local", latency)
    return Answer(best, mcq.options[best], "local", latency)


# --------------------------------------------------------------------- #
# LLM APIs - TEXT ONLY, short prompt, tiny token budget, short timeout
# --------------------------------------------------------------------- #
PROMPT = (
    "You are answering a multiple-choice question. "
    "Reply with ONLY the single option letter (A, B, C, D or E). Nothing else.\n"
    "{mcq}"
)


def _extract_letter(text: str, mcq: MCQ) -> str | None:
    m = LETTER_RE.search(text.strip().upper())
    if m and m.group(1) in mcq.options:
        return m.group(1)
    return None


def gemini_model_candidates(api_key: str, timeout: float = 6.0) -> list[str]:
    """Models this key can actually list, fastest text model first. Falls back
    to a static list if the listing call itself fails."""
    try:
        r = requests.get(f"https://generativelanguage.googleapis.com/v1beta/models?key={api_key}",
                         timeout=timeout)
        r.raise_for_status()
        names = [m["name"].split("/")[-1] for m in r.json().get("models", [])]
        ordered = [p for p in GEMINI_MODEL_PREFS if p in names]
        ordered += [n for n in names
                    if ("flash" in n or "gemini-3" in n)
                    and not any(x in n for x in NON_TEXT_MARKERS)
                    and n not in ordered]
        if ordered:
            return ordered
    except Exception as e:
        print(f"[solver] model listing failed ({e}); trying static list")
    return list(GEMINI_MODEL_PREFS)



def _gemini_text(data: dict) -> str:
    """Extract text from a generateContent response tolerantly (content.parts
    may be absent for blocked/odd responses)."""
    try:
        parts = data["candidates"][0]["content"]["parts"]
        return "".join(p.get("text", "") for p in parts)
    except Exception:
        return str(data)[:200]


def solve_gemini(api_key: str, model: str, mcq: MCQ, timeout: float = 6.0) -> Answer:
    """Try every candidate model x both API versions (v1 first - v1beta timed
    out from some networks). Each failure is logged WITH THE RESPONSE BODY so
    a 404 reveals the real reason (retired model, bad key, wrong project...)."""
    start = time.perf_counter()
    body = {
        "contents": [{"parts": [{"text": PROMPT.format(mcq=mcq.to_prompt())}]}],
        "generationConfig": {"maxOutputTokens": 16, "temperature": 0},
    }
    candidates = [model] if model else []
    candidates += [m for m in gemini_model_candidates(api_key, timeout) if m not in candidates]
    last_err = None
    for m in candidates[:5]:
        for api_version in ("v1", "v1beta"):
            url = (f"https://generativelanguage.googleapis.com/{api_version}/models/"
                   f"{m}:generateContent?key={api_key}")
            try:
                r = None
                for attempt in range(2):  # retry once on 503 / read timeout
                    try:
                        r = requests.post(url, json=body, timeout=timeout)
                        r.raise_for_status()
                        break
                    except requests.HTTPError as e:
                        if e.response is not None and e.response.status_code == 503 and attempt == 0:
                            time.sleep(0.5)
                            continue
                        raise
                if r is None:
                    raise RuntimeError("no response")
                data = r.json()
                text = _gemini_text(data)
                latency = int((time.perf_counter() - start) * 1000)
                print(f"[solver] gemini OK with {m} ({api_version})")
                return Answer(_extract_letter(text, mcq),
                              mcq.options.get(_extract_letter(text, mcq) or "", text),
                              "gemini", latency, raw=text)
            except requests.HTTPError as e:
                resp_body = e.response.text[:300] if e.response is not None else ""
                last_err = f"{m}@{api_version}: HTTP {e.response.status_code} body={resp_body}"
                print(f"[solver] {last_err}")
            except Exception as e:
                last_err = f"{m}@{api_version}: {e}"
                print(f"[solver] {last_err}")
    raise RuntimeError(last_err or "no gemini model worked")


def solve_deepseek(api_key: str, model: str, mcq: MCQ, timeout: float = 6.0) -> Answer:
    start = time.perf_counter()
    url = "https://api.deepseek.com/chat/completions"
    headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
    body = {
        "model": model,
        "messages": [{"role": "user", "content": PROMPT.format(mcq=mcq.to_prompt())}],
        "max_tokens": 16,
        "temperature": 0,
    }
    r = requests.post(url, json=body, headers=headers, timeout=timeout)
    r.raise_for_status()
    data = r.json()
    text = data["choices"][0]["message"]["content"]
    latency = int((time.perf_counter() - start) * 1000)
    return Answer(_extract_letter(text, mcq), mcq.options.get(_extract_letter(text, mcq) or "", text),
                  "deepseek", latency, raw=text)


# --------------------------------------------------------------------- #
# Public entry point with caching
# --------------------------------------------------------------------- #
class Solver:
    def __init__(self, provider: str = "none", api_key: str = "",
                 model: str = "", timeout: float = 6.0):
        self.provider = provider.lower()
        self.api_key = api_key
        self.model = model or {"gemini": "", "deepseek": "deepseek-chat"}.get(self.provider, "")
        self.timeout = timeout
        self._cache: dict[str, Answer] = {}

    def solve(self, mcq: MCQ) -> Answer:
        key = mcq.to_prompt()
        if key in self._cache:
            cached = self._cache[key]
            return Answer(cached.letter, cached.text, cached.source + " (cached)", 0, cached.raw)
        answer = None
        if self.provider in ("gemini", "deepseek") and self.api_key:
            try:
                fn = solve_gemini if self.provider == "gemini" else solve_deepseek
                answer = fn(self.api_key, self.model, mcq, self.timeout)
            except Exception as e:
                print(f"[solver] LLM failed ({e}); using local fallback")
        if answer is None or answer.letter is None:
            answer = solve_local(mcq)
        self._cache[key] = answer
        return answer
