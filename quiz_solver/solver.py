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
def solve_local(mcq: MCQ) -> Answer:
    start = time.perf_counter()
    q = mcq.question.lower()
    q_words = set(re.findall(r"[a-z]+", q))
    best, best_score = None, -1.0
    for letter, opt in mcq.options.items():
        o_words = set(re.findall(r"[a-z]+", opt.lower()))
        if not o_words:
            continue
        # options sharing distinctive words with the question are usually wrong;
        # MCQ answers often relate to what the question asks about -> use overlap
        # of rare words, plus prefer longer, more informative options slightly.
        overlap = len(q_words & o_words)
        rare = len([w for w in (q_words & o_words) if len(w) > 5])
        score = rare * 2 - overlap * 0.5 + len(opt) / 200.0
        if score > best_score:
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
    url = (f"https://generativelanguage.googleapis.com/v1beta/models/"
           f"{model}:generateContent?key={api_key}")
    body = {
        "contents": [{"parts": [{"text": PROMPT.format(mcq=mcq.to_prompt())}]}],
        "generationConfig": {"maxOutputTokens": 16, "temperature": 0},
    }
    r = requests.post(url, json=body, timeout=timeout)
    r.raise_for_status()
    data = r.json()
    text = data["candidates"][0]["content"]["parts"][0]["text"]
    latency = int((time.perf_counter() - start) * 1000)
    return Answer(_extract_letter(text, mcq), mcq.options.get(_extract_letter(text, mcq) or "", text),
                  "gemini", latency, raw=text)


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
