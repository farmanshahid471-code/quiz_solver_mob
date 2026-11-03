#!/usr/bin/env python3
"""Quiz Solver - scan MCQs on your laptop screen with your phone camera.

Usage examples:
  python main.py --stream-url http://192.168.1.50:8080/video \
      --provider gemini --api-key $GEMINI_API_KEY

  python main.py --stream-url http://192.168.1.50:8080/video --provider none   # offline
  python main.py --demo sample_ocr.txt --provider deepseek --api-key $DEEPSEEK_API_KEY

Keys in the live window:
  SPACE  solve the current frame now
  A      toggle auto-solve (OCR triggers when the screen changes & settles)
  Q/ESC  quit
"""

from __future__ import annotations

import argparse
import hashlib
import os
import sys
import threading
import time

import cv2
import numpy as np

from mcq_parser import parse_mcq
from ocr_engine import build_ocr
from overlay import AnswerWindow, draw_on_frame
from phone_stream import PhoneCamera
from solver import Answer, Solver


def load_env(path: str = ".env") -> None:
    if not os.path.exists(path):
        return
    for line in open(path):
        line = line.strip()
        if line and not line.startswith("#") and "=" in line:
            k, v = line.split("=", 1)
            os.environ.setdefault(k.strip(), v.strip().strip('"').strip("'"))


class QuizSolverApp:
    def __init__(self, args):
        self.args = args
        self.ocr = build_ocr(args.ocr)
        self.solver = Solver(args.provider,
                             args.api_key or os.environ.get(f"{args.provider.upper()}_API_KEY", ""),
                             args.model, args.timeout)
        self.auto = args.auto
        self.answer: Answer | None = None
        self.status = "ready - press SPACE to scan"
        self._last_small = None
        self._stable = 0
        self._changed = False
        self._busy = False
        self._last_scan = 0.0
        self.cooldown = 3.0  # min seconds between auto scans (OCR is the slow part)
        self._lock = threading.Lock()
        self._last_hash = ""

    # ------------------------------------------------------------------ #
    def solve_frame(self, frame: np.ndarray) -> None:
        """OCR + solve in a worker thread so the preview never freezes."""
        with self._lock:
            if self._busy:
                return
            self._busy = True
        self.status = "scanning..."

        def work():
            t0 = time.perf_counter()
            # OCR on a downscaled copy - much faster, text stays readable
            h, w = frame.shape[:2]
            if w > 640:
                frame = cv2.resize(frame, (640, int(h * 640 / w)), interpolation=cv2.INTER_AREA)
            text = self.ocr.read_text(frame)
            t_ocr = (time.perf_counter() - t0) * 1000
            mcq = parse_mcq(text)
            if mcq is None:
                ans = Answer(None, "no MCQ detected - aim camera at the question", "none", int(t_ocr))
            else:
                h = hashlib.md5(mcq.to_prompt().encode()).hexdigest()
                if h == self._last_hash and self.answer is not None:
                    ans = self.answer
                else:
                    self._last_hash = h
                    ans = self.solver.solve(mcq)
            with self._lock:
                self.answer = ans
                self.status = f"OCR {t_ocr:.0f} ms"
                self._busy = False
            if self.win:
                try:
                    self.win.show(ans)
                except Exception:
                    pass
            print(f"[result] {ans.letter or '?'} | {ans.source} | {ans.latency_ms} ms | OCR {t_ocr:.0f} ms")
            if mcq:
                print(f"[q] {mcq.question}\n[opts] {mcq.options}")

        threading.Thread(target=work, daemon=True).start()

    # ------------------------------------------------------------------ #
    def run(self) -> None:
        cam = PhoneCamera(stream_url=self.args.stream_url if self.args.source is None else None,
                          snapshot_url=self.args.snapshot_url,
                          source=self.args.source,
                          width=960, height=540)
        if not self.args.no_window:
            try:
                self.win = AnswerWindow()
            except Exception:
                self.win = None
        else:
            self.win = None

        print("[info] keys: SPACE=scan now, A=auto-solve on/off, Q=quit")
        prev_time = time.perf_counter()
        fps = 0.0
        while True:
            ok, frame = cam.read()
            if not ok or frame is None:
                time.sleep(0.02)
                continue

            now = time.perf_counter()
            fps = 0.9 * fps + 0.1 * (1.0 / max(now - prev_time, 1e-3))
            prev_time = now

            # auto-trigger: screen changed, then stable for a few frames
            small = cv2.resize(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY), (64, 36))
            if self.auto and self._last_small is not None:
                diff = float(np.mean(cv2.absdiff(small, self._last_small)))
                if diff > 6.0:
                    self._stable = 0
                    self._changed = True
                else:
                    self._stable += 1
                    if self._changed and self._stable >= 8 and not self._busy:
                        self._changed = False
                        self.solve_frame(frame)
            self._last_small = small

            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), 27):
                break
            elif key == ord(" "):
                self.solve_frame(frame)
            elif key == ord("a"):
                self.auto = not self.auto
                self._changed = False
                print(f"[info] auto-solve {'ON' if self.auto else 'OFF'}")

            out = draw_on_frame(frame, self.answer, self.status, fps)
            cv2.imshow("Quiz Solver - phone camera", out)

        cam.release()
        cv2.destroyAllWindows()
        if self.win:
            self.win.close()


def run_demo(path: str, provider: str, api_key: str, model: str, timeout: float) -> None:
    text = open(path).read()
    mcq = parse_mcq(text)
    if mcq is None:
        print("no MCQ found in demo file"); sys.exit(1)
    print(mcq.to_prompt())
    solver = Solver(provider, api_key, model, timeout)
    ans = solver.solve(mcq)
    print(f"\nANSWER: {ans.letter} - {ans.text}  [{ans.source}, {ans.latency_ms} ms]")


def main() -> None:
    load_env()
    p = argparse.ArgumentParser(description="Phone-camera MCQ solver")
    p.add_argument("--stream-url", default=os.environ.get("PHONE_STREAM_URL", "http://192.168.1.100:8080/video"),
                   help="MJPEG stream URL of the phone camera (IP Webcam: http://<phone-ip>:8080/video)")
    p.add_argument("--snapshot-url", default=os.environ.get("PHONE_SNAPSHOT_URL", ""),
                   help="fallback snapshot URL (IP Webcam: http://<phone-ip>:8080/shot.jpg)")
    p.add_argument("--source", type=int, default=None,
                   help="local webcam index (e.g. --source 0) when the phone appears as a PC webcam via iVCam/EpocCam/DroidCam")
    p.add_argument("--ocr", default="rapidocr", choices=["rapidocr", "tesseract"])
    p.add_argument("--provider", default="none", choices=["none", "gemini", "deepseek"])
    p.add_argument("--api-key", default="")
    p.add_argument("--model", default="")
    p.add_argument("--timeout", type=float, default=6.0, help="LLM HTTP timeout in seconds")
    p.add_argument("--auto", action="store_true", help="auto-solve when the screen changes")
    p.add_argument("--no-window", action="store_true", help="disable the always-on-top answer window")
    p.add_argument("--demo", default="", help="solve an MCQ from a text file instead of the camera")
    args = p.parse_args()

    if args.demo:
        run_demo(args.demo, args.provider,
                 args.api_key or os.environ.get(f"{args.provider.upper()}_API_KEY", ""),
                 args.model, args.timeout)
    else:
        QuizSolverApp(args).run()


if __name__ == "__main__":
    main()
