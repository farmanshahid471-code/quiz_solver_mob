# Quiz Solver — phone camera → MCQ answers on your laptop screen

Point your **phone camera** at MCQs shown on your **laptop screen**; the PC reads
the phone's video stream over Wi‑Fi, OCRs the question, and shows the answer
**instantly** — only the extracted **text** is ever sent to an LLM API, never
the picture, so nothing slow is uploaded.

## How it works

```
Phone (IP Webcam app) --Wi-Fi MJPEG--> PC (this project)
        --> OpenCV frame --> RapidOCR (fast, on-device, ONNX)
        --> MCQ parser --> LLM API (Gemini/DeepSeek, TEXT only, ~0.3-1.5 s)
                            or local heuristic fallback (offline, ~0 ms)
        --> answer overlay on live preview + always-on-top window
```

## 1. Setup the phone as a camera

1. Install **IP Webcam** on Android
   (repo: https://github.com/pavelkhlebovich/ip-webcam , Play Store: "IP Webcam").
2. Phone and PC on the **same Wi‑Fi**. Open the app → "Start server".
3. Note the phone's IP shown at the bottom, e.g. `http://192.168.1.50:8080`.
   - stream: `http://192.168.1.50:8080/video`
   - snapshot fallback: `http://192.168.1.50:8080/shot.jpg`

   (DroidCam also works: `http://<phone-ip>:4747/video`.)

## 2. Install on the PC

```bash
cd quiz_solver
python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

### Windows — keep everything on the D: drive

`git clone` puts the folder in your **current** directory, so make sure the
prompt shows `D:\>` before cloning (check the drive letter in the prompt!):

```bat
D:\> git clone -b arena/cb605cae-quiz-solver-mob https://github.com/farmanshahid471-code/quiz_solver_mob.git
D:\> cd D:\quiz_solver_mob\quiz_solver
D:\quiz_solver_mob\quiz_solver> python -m venv D:\quiz_solver_mob\.venv
D:\quiz_solver_mob\quiz_solver> D:\quiz_solver_mob\.venv\Scripts\activate
(.venv) D:\quiz_solver_mob\quiz_solver> pip install -r requirements.txt
```

Troubleshooting: if you ran `cd D:\` but the prompt still shows `C:\...`, the
clone went to the C: folder. Move it over and redo the venv in the right place:

```bat
REM only do this if D:\quiz_solver_mob contains just the stray .venv:
rmdir /s /q D:\quiz_solver_mob
move C:\Users\<you>\quiz_solver_mob D:\
cd D:\quiz_solver_mob\quiz_solver
python -m venv D:\quiz_solver_mob\.venv
```

Optional: `pip install pytesseract` + the Tesseract binary if you prefer it over RapidOCR.

## 3. Run

```bash
# Offline mode (local heuristic only, fastest, no key needed)
python main.py --stream-url http://192.168.1.50:8080/video --provider none

# With Gemini (text-only API call, no image upload)
export GEMINI_API_KEY=your_key
python main.py --stream-url http://192.168.1.50:8080/video --provider gemini --auto

# Or DeepSeek
export DEEPSEEK_API_KEY=your_key
python main.py --stream-url http://192.168.1.50:8080/video --provider deepseek --auto
```

Tip: put keys in a `.env` file (`GEMINI_API_KEY=...`, `PHONE_STREAM_URL=...`) — it is loaded automatically.

**Keys in the live window:** `SPACE` = scan now · `A` = toggle auto-solve
(OCR fires automatically when the on-screen content changes and settles) · `Q` = quit.

## 4. Test without a camera

```bash
python main.py --demo sample_ocr.txt --provider none   # solve one sample MCQ
python -m pytest tests/ -q                             # offline test suite (parser + solver)
```

## Speed notes

- RapidOCR runs locally on CPU (ONNX) — no model download per scan, ~100-400 ms/frame.
- LLM call sends a tiny text-only prompt with `max_tokens=16` and a 6 s timeout,
  so a wrong/slow API never blocks the preview; it falls back to the local solver.
- OCR runs in a worker thread — the camera preview never freezes.
- Re-scanning the same question is answered from cache in 0 ms.

## Useful GitHub repos this builds on

- Phone as webcam: `pavelkhlebovich/ip-webcam` (Android app), `home-assistant-libs/pydroid-ipcam` (Python client)
- Fast OCR: `RapidAI/RapidOCR` (default here), `PaddlePaddle/PaddleOCR`
- LLM APIs: Google Gemini `generativelanguage.googleapis.com`, DeepSeek `api.deepseek.com`

## Files

| file | purpose |
|---|---|
| `main.py` | entry point, live preview, hotkeys, auto-solve trigger |
| `phone_stream.py` | MJPEG stream client with snapshot fallback |
| `ocr_engine.py` | RapidOCR / Tesseract wrappers + preprocessing |
| `mcq_parser.py` | parses OCR text into question + options A-E |
| `solver.py` | text-only Gemini/DeepSeek calls + local fallback + cache |
| `overlay.py` | on-frame answer bar + always-on-top answer window |
