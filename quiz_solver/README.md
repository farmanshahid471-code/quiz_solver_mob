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

## 1. Connect your phone

The PC and phone must be on the **same router/LAN** (PC by cable + phone on
Wi‑Fi is fine). The video stays on your local network — it never goes to the
internet. Quick check from the PC: `ping <phone-ip>` must reply.

### Android (recommended — free, Wi‑Fi or USB)

1. Install **IP Webcam** (Play Store; repo: https://github.com/pavelkhlebovich/ip-webcam).
2. Open the app → **Start server**. It shows the phone's IP at the bottom,
   e.g. `http://192.168.1.50:8080` — **use this exact IP** in `.env` as
   `PHONE_STREAM_URL=http://192.168.1.50:8080/video` (no angle brackets!).
3. Run: `python main.py --provider gemini --auto`

   Alternatives: **DroidCam** (`http://<phone-ip>:4747/video`, also has a USB mode).

### iPhone

There is no IP Webcam for iOS, so use an app that makes the phone appear as a
**local webcam on the PC** (install the app + the PC driver):

- **iVCam** (eInfoMount) or **EpocCam** (Elgato) — after installing, the phone
  shows up as a normal webcam. Then run with the webcam index:

  ```bat
  python main.py --source 0 --provider gemini --auto
  ```

  (If the window is black, try `--source 1`, `--source 2`...)

### Troubleshooting the connection

- `Connection to tcp://... failed` / `ping` times out → wrong IP (check the IP
  shown **in the app**, not a guess), phone not on the same Wi‑Fi, or the
  router has **AP/client isolation** on (disable it in router settings).
- `Failed to create DXGI factory.` is a harmless OpenCV warning on some Windows
  setups — the preview window still opens.
- Always put the **real phone IP** in `.env` / `--stream-url`; the shipped
  `.env.example` value `192.168.1.100` is only a placeholder.

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
REM IMPORTANT: in cmd.exe use "cd /d" to switch drives — plain "cd D:\..."
REM only changes the folder on D: but keeps the prompt on C:!
D:\> git clone -b arena/cb605cae-quiz-solver-mob https://github.com/farmanshahid471-code/quiz_solver_mob.git
D:\> cd /d D:\quiz_solver_mob\quiz_solver
D:\quiz_solver_mob\quiz_solver> python -m venv D:\quiz_solver_mob\.venv
D:\quiz_solver_mob\quiz_solver> D:\quiz_solver_mob\.venv\Scripts\activate
(.venv) D:\quiz_solver_mob\quiz_solver> pip install -r requirements.txt
```

Troubleshooting: if you ran `cd D:\` but the prompt still shows `C:\...`, the
clone went to the C: folder. Easiest fix (avoids "Access is denied" from
`move`, which happens when Explorer/antivirus locks the folder): close all
File Explorer windows, remove the stray folder on D:, and re-clone directly
on D:

```bat
REM explicit destination path -> works from ANY current directory:
rmdir /s /q D:\quiz_solver_mob
REM if an old clone exists on C:, remove it first (close Explorer windows!):
rmdir /s /q C:\Users\<you>\quiz_solver_mob
git clone -b arena/cb605cae-quiz-solver-mob https://github.com/farmanshahid471-code/quiz_solver_mob.git D:\quiz_solver_mob
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
