"""OCR engines. Only TEXT ever leaves the machine (never the image).

Default engine is RapidOCR (RapidAI/RapidOCR - ONNX Runtime based), which is
the fastest good-accuracy CPU OCR and ideal for the "really quick" need.
pytesseract is available as a fallback.
"""

from __future__ import annotations

import cv2
import numpy as np


def preprocess(img: np.ndarray, target_width: int = 1600) -> np.ndarray:
    """Upscale small text, enhance contrast, denoise - improves OCR a lot."""
    h, w = img.shape[:2]
    if w < target_width:
        scale = target_width / w
        img = cv2.resize(img, (target_width, int(h * scale)), interpolation=cv2.INTER_CUBIC)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    gray = clahe.apply(gray)
    gray = cv2.fastNlMeansDenoising(gray, None, h=7, templateWindowSize=7, searchWindowSize=21)
    return gray


class OCREngine:
    name = "base"

    def read_text(self, img: np.ndarray) -> str:  # pragma: no cover - interface
        raise NotImplementedError


class RapidOCREngine(OCREngine):
    name = "rapidocr"

    def __init__(self):
        from rapidocr_onnxruntime import RapidOCR  # heavy import, done lazily
        self._ocr = RapidOCR()

    def read_text(self, img: np.ndarray) -> str:
        proc = preprocess(img)
        result, _ = self._ocr(proc)
        if not result:
            return ""
        # sort top-to-bottom, left-to-right into lines
        items = sorted(result, key=lambda r: (r[0][0][1], r[0][0][0]))
        lines, last_y, cur = [], None, []
        for box, text, score in items:
            y = box[0][1]
            if last_y is None or abs(y - last_y) > 18:
                if cur:
                    lines.append(" ".join(cur))
                cur = [text]
                last_y = y
            else:
                cur.append(text)
        if cur:
            lines.append(" ".join(cur))
        return "\n".join(lines)


class TesseractEngine(OCREngine):
    name = "tesseract"

    def __init__(self):
        import pytesseract
        from PIL import Image  # noqa: F401  (pytesseract needs PIL)
        self._pytesseract = pytesseract

    def read_text(self, img: np.ndarray) -> str:
        proc = preprocess(img)
        return self._pytesseract.image_to_string(proc, config="--psm 6").strip()


def build_ocr(name: str = "rapidocr") -> OCREngine:
    name = name.lower()
    if name in ("rapidocr", "rapid", "rapidocr_onnxruntime"):
        return RapidOCREngine()
    if name in ("tesseract", "pytesseract"):
        return TesseractEngine()
    raise ValueError(f"unknown OCR engine: {name}")
