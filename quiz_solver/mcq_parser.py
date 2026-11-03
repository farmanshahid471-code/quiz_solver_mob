"""Parse raw OCR text into (question, options) for an MCQ."""

from __future__ import annotations

import re
from dataclasses import dataclass, field

# Matches "A)", "(B)", "c.", "D:", "E -" at the start of a line
OPTION_RE = re.compile(r"^\s*\(?([A-Ea-e])\s*[\.\)\:\-]\s*(.+)$")
# Matches inline options inside one line: "A) foo B) bar C) baz"
INLINE_RE = re.compile(r"\(?([A-Ea-e])\s*[\)\.\:]\s*")


@dataclass
class MCQ:
    question: str
    options: dict[str, str] = field(default_factory=dict)
    raw: str = ""

    def to_prompt(self) -> str:
        lines = [f"Question: {self.question}"]
        for letter in sorted(self.options):
            lines.append(f"{letter}) {self.options[letter]}")
        return "\n".join(lines)


def _clean(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


def parse_mcq(text: str) -> MCQ | None:
    """Return an MCQ if the text looks like one, else None."""
    if not text or len(text) < 8:
        return None
    lines = [l.strip() for l in text.splitlines() if l.strip()]

    options: dict[str, str] = {}
    question_lines: list[str] = []

    # Pass 1: option-per-line, keeping contiguous blocks so we can pick the
    # biggest/best block (the camera often sees browser junk around the MCQ)
    blocks: list[tuple[list[str], dict[str, str]]] = []
    cur_q: list[str] = []
    cur_o: dict[str, str] = {}
    for line in lines:
        m = OPTION_RE.match(line)
        if m:
            cur_o[m.group(1).upper()] = _clean(m.group(2))
        else:
            if cur_o:
                blocks.append((cur_q, cur_o))
                cur_q, cur_o = [], {}
            cur_q.append(line)
    if cur_o:
        blocks.append((cur_q, cur_o))
    if not blocks:
        question_lines = lines  # no per-line options -> maybe inline options

    if blocks:
        # prefer the block with the most options (real MCQ, not stray "A)" junk)
        question_lines, options = max(blocks, key=lambda b: len(b[1]))

    # Pass 2: inline options on a single line, e.g. "A) x B) y C) z D) w"
    if len(options) < 2 and question_lines:
        joined = " ".join(question_lines)
        matches = list(INLINE_RE.finditer(joined))
        if len(matches) >= 2:
            options = {}
            question_lines = [joined[: matches[0].start()].strip()]
            for i, m in enumerate(matches):
                end = matches[i + 1].start() if i + 1 < len(matches) else len(joined)
                options[m.group(1).upper()] = _clean(joined[m.end():end])

    if len(options) < 2:
        return None

    # normalize common OCR confusions for single-character options (O vs 0, l vs 1)
    for letter, opt in list(options.items()):
        if len(opt) == 1:
            options[letter] = {"O": "0", "o": "0", "l": "1", "I": "1", "|": "1"}.get(opt, opt)

    question = _clean(" ".join(question_lines))
    if not question:
        question = "(question not recognized)"
    return MCQ(question=question, options=options, raw=text)
