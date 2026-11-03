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
            # elimination: pick the option with the LEAST overlap
            score = -(rare * 2 + overlap) + len(opt) / 500.0
        else:
            # positive question: prefer distinctive shared words, slightly
            # prefer longer/more informative options
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


def solve_gemini(api_key: str, model: str, mcq: MCQ, timeout: float = 6.0) -> Answer:
    start = time.perf_counter()
    body = {
        "contents": [{"parts": [{"text": PROMPT.format(mcq=mcq.to_prompt())}]}],
        "generationConfig": {"maxOutputTokens": 16, "temperature": 0},
    }
    # try v1beta first, fall back to v1 (404 happens when the model name is
    # only available on one of the endpoints for a given key)
    last_err = None
    for api_version in ("v1beta", "v1"):
        url = (f"https://generativelanguage.googleapis.com/{api_version}/models/"
               f"{model}:generateContent?key={api_key}")
        try:
            r = requests.post(url, json=body, timeout=timeout)
            r.raise_for_status()
            data = r.json()
            text = data["candidates"][0]["content"]["parts"][0]["text"]
            latency = int((time.perf_counter() - start) * 1000)
            return Answer(_extract_letter(text, mcq),
                          mcq.options.get(_extract_letter(text, mcq) or "", text),
                          "gemini", latency, raw=text)
        except Exception as e:
            last_err = e
    raise last_err


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
        self.model = model or {"gemini": "gemini-2.0-flash",
                               "deepseek": "deepseek-chat"}.get(self.provider, "")
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
