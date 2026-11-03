"""Offline tests for the MCQ parser and solver (no camera / no API key needed)."""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from mcq_parser import parse_mcq
from solver import Solver, solve_local


# ------------------------------- parser ------------------------------- #

def test_multiline_options():
    m = parse_mcq("3. Which planet is known as the Red Planet?\n"
                  "A) Venus\nB) Mars\nC) Jupiter\nD) Mercury")
    assert m.question.endswith("Red Planet?")
    assert m.options == {"A": "Venus", "B": "Mars", "C": "Jupiter", "D": "Mercury"}


def test_inline_options():
    m = parse_mcq("What is 2+2? A) 3 B) 4 C) 5 D) 6")
    assert m.question == "What is 2+2?"
    assert m.options["B"] == "4"


def test_lowercase_and_paren_style():
    m = parse_mcq("Which gas do plants absorb?\n(a) Oxygen\n(b) Nitrogen\n"
                  "(c) Carbon dioxide\n(d) Hydrogen")
    assert m.options["C"] == "Carbon dioxide"


def test_multiline_question():
    m = parse_mcq("In Python, which keyword\nis used to define a function?\n"
                  "A) func\nB) def\nC) lambda\nD) class")
    assert "which keyword" in m.question and "define a function?" in m.question
    assert m.options["B"] == "def"


def test_not_an_mcq():
    assert parse_mcq("just some random sentence without options") is None
    assert parse_mcq("") is None
    assert parse_mcq("A) only one option") is None


# ------------------------------- solver ------------------------------- #

def test_local_solver_returns_valid_letter():
    m = parse_mcq("Which planet is known as the Red Planet?\n"
                  "A) Venus\nB) Mars\nC) Jupiter\nD) Mercury")
    a = solve_local(m)
    assert a.letter in m.options
    assert a.source == "local"


def test_negation_flips_to_elimination():
    # "NOT" questions: the answer is usually the option LEAST related to the
    # question words -> local solver must not pick the most-overlapping option.
    m = parse_mcq("Which of the following is NOT a mammal?\n"
                  "A) whale\nB) dolphin\nC) bat\nD) crocodile")
    a = solve_local(m)
    assert a.letter == "D", a


def test_solver_falls_back_when_api_unreachable():
    m = parse_mcq(open(os.path.join(os.path.dirname(__file__), "..", "sample_ocr.txt")).read())
    s = Solver("gemini", api_key="fake-key", model="gemini-2.0-flash", timeout=2)
    a = s.solve(m)
    assert a.source == "local"  # API unreachable in tests -> local fallback


def test_solver_cache_is_zero_latency():
    m = parse_mcq("Capital of France?\nA) London\nB) Paris\nC) Rome\nD) Madrid")
    s = Solver("none")
    first = s.solve(m)
    second = s.solve(m)
    assert second.latency_ms == 0 and "cached" in second.source
    assert second.letter == first.letter
