"""Draw the answer on the live frame + keep a small always-on-top window."""

from __future__ import annotations

import cv2
import numpy as np

from solver import Answer


def draw_on_frame(frame: np.ndarray, answer: Answer | None,
                  status: str, fps: float) -> np.ndarray:
    out = frame.copy()
    h, w = out.shape[:2]
    bar_h = 64 if answer else 34
    overlay = out.copy()
    cv2.rectangle(overlay, (0, h - bar_h), (w, h), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.65, out, 0.35, 0, out)
    if answer and answer.letter:
        txt = f"ANSWER: {answer.letter}   ({answer.source}, {answer.latency_ms} ms)"
        color = (0, 255, 0)
    elif answer:
        txt = f"NO CLEAR ANSWER ({answer.source}, {answer.latency_ms} ms)"
        color = (0, 200, 255)
    else:
        txt = status
        color = (200, 200, 200)
    cv2.putText(out, txt, (12, h - bar_h + (24 if answer else 22)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.65, color, 2, cv2.LINE_AA)
    if answer:
        cv2.putText(out, answer.text[:90], (12, h - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
    cv2.putText(out, f"{fps:.1f} FPS", (w - 110, 24),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 1, cv2.LINE_AA)
    return out


class AnswerWindow:
    """Small always-on-top Tk window so the answer stays visible while you move the phone."""

    def __init__(self, title: str = "Quiz Solver - ANSWER"):
        import tkinter as tk
        self.root = tk.Tk()
        self.root.title(title)
        self.root.attributes("-topmost", True)
        self.label = tk.Label(self.root, text="waiting for question...",
                              font=("Helvetica", 22, "bold"), fg="green",
                              bg="black", padx=20, pady=14, justify="left")
        self.label.pack()
        self.root.update()

    def show(self, answer: Answer | None) -> None:
        if answer and answer.letter:
            self.label.config(text=f"ANSWER: {answer.letter}\n{answer.text[:120]}\n({answer.source}, {answer.latency_ms} ms)",
                              fg="green")
        elif answer:
            self.label.config(text="no clear answer", fg="orange")
        self.root.update()

    def close(self) -> None:
        try:
            self.root.destroy()
        except Exception:
            pass
