# quiz_solver_mob

Phone-camera MCQ solver: point your phone (as a Wi‑Fi webcam) at MCQs on your
laptop screen and get instant answers on your PC — OCR on-device + optional
text-only LLM (Gemini/DeepSeek). See [`quiz_solver/README.md`](quiz_solver/README.md).

Quick start:

```bash
cd quiz_solver
pip install -r requirements.txt
python main.py --stream-url http://<phone-ip>:8080/video --provider gemini   # add your API key
```
