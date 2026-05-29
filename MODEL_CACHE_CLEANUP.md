# Model Cache And Deep Cleanup

PaddleOCR, PaddleX, Hugging Face, and llama.cpp can store large model files outside this project directory.

This guide explains where those files are saved and how to clean them safely.

## 1. Project Files

This project lives here:

```text
/Users/ayusharyansingh/Developer/Python/receipt-automation
```

Project-local model files are expected here:

```text
models/
```

Project outputs are saved here:

```text
outputs/
```

Project input samples are saved here:

```text
data/
```

## 2. PaddleOCR / PaddleX Model Cache

When you see logs like:

```text
Using official model (PP-DocLayoutV3), the model files will be automatically downloaded and saved in /Users/ayusharyansingh/.paddlex/official_models/PP-DocLayoutV3
Using official model (PaddleOCR-VL-1.6), the model files will be automatically downloaded and saved in /Users/ayusharyansingh/.paddlex/official_models/PaddleOCR-VL-1.6
```

The models are being stored here:

```text
/Users/ayusharyansingh/.paddlex/official_models/
```

Common PaddleOCR-VL cache directories:

```text
/Users/ayusharyansingh/.paddlex/official_models/PP-DocLayoutV3
/Users/ayusharyansingh/.paddlex/official_models/PaddleOCR-VL-1.6
```

Check size:

```bash
du -sh ~/.paddlex
find ~/.paddlex/official_models -maxdepth 2 -type d -print
```

Remove only PaddleOCR-VL related cached models:

```bash
rm -rf ~/.paddlex/official_models/PP-DocLayoutV3
rm -rf ~/.paddlex/official_models/PaddleOCR-VL-1.6
```

Remove all PaddleX official model cache:

```bash
rm -rf ~/.paddlex/official_models
```

Remove all PaddleX cache:

```bash
rm -rf ~/.paddlex
```

Note: after deleting these, PaddleOCR will download the models again next time you run the pipeline.

## 3. PaddleHub Cache

Some Paddle tools may also use:

```text
/Users/ayusharyansingh/.paddlehub
```

Check size:

```bash
du -sh ~/.paddlehub 2>/dev/null || true
```

Clean:

```bash
rm -rf ~/.paddlehub
```

## 4. Hugging Face Cache

CLIP, Qwen downloads, and dataset downloads can use Hugging Face cache directories.

Common locations:

```text
/Users/ayusharyansingh/.cache/huggingface
/Users/ayusharyansingh/.cache/huggingface/hub
/Users/ayusharyansingh/.cache/huggingface/datasets
```

Check size:

```bash
du -sh ~/.cache/huggingface 2>/dev/null || true
du -sh ~/.cache/huggingface/hub 2>/dev/null || true
du -sh ~/.cache/huggingface/datasets 2>/dev/null || true
```

Remove all Hugging Face cache:

```bash
rm -rf ~/.cache/huggingface
```

Remove only dataset cache:

```bash
rm -rf ~/.cache/huggingface/datasets
```

Remove only model cache:

```bash
rm -rf ~/.cache/huggingface/hub
```

Note: deleting Hugging Face cache means CLIP, datasets, or downloaded models may need to download again.

## 5. Local Project Data Cleanup

Remove generated Excel files:

```bash
rm -rf outputs/*.xlsx
```

Remove SROIE sample data:

```bash
rm -rf data/sroie_sample
```

Remove debug input data:

```bash
rm -rf data/debug
```

Remove failed/processed/review folders if you used them:

```bash
rm -rf data/processed/*
rm -rf data/failed/*
rm -rf data/review/*
```

Keep `data/inbox` unless you intentionally want to delete your input receipts.

## 6. Python Cache Cleanup

Remove Python bytecode caches inside the project:

```bash
find . -type d -name __pycache__ -prune -exec rm -rf {} +
find . -type f -name '*.pyc' -delete
```

## 7. Virtual Environment Cleanup

Remove the virtual environment:

```bash
rm -rf .venv
```

Recreate it:

```bash
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

## 8. Full Deep Cleanup

This removes generated outputs, local samples, Python caches, and major external model caches.

Use carefully:

```bash
rm -rf outputs/*.xlsx
rm -rf data/sroie_sample
rm -rf data/debug
find . -type d -name __pycache__ -prune -exec rm -rf {} +
find . -type f -name '*.pyc' -delete
rm -rf ~/.paddlex
rm -rf ~/.paddlehub
rm -rf ~/.cache/huggingface
```

This does not delete:

```text
source code
.env
requirements.txt
models/ files inside the project
manual receipts in data/inbox
```

If you also want to remove project-local downloaded GGUF/model files:

```bash
rm -rf models/*.gguf
rm -rf models/qwen3.5-2b-gguf
```

## 9. Check Disk Usage Before And After

Before cleanup:

```bash
du -sh .
du -sh ~/.paddlex 2>/dev/null || true
du -sh ~/.paddlehub 2>/dev/null || true
du -sh ~/.cache/huggingface 2>/dev/null || true
```

After cleanup:

```bash
du -sh .
du -sh ~/.paddlex 2>/dev/null || true
du -sh ~/.paddlehub 2>/dev/null || true
du -sh ~/.cache/huggingface 2>/dev/null || true
```

## 10. Disable Paddle Model Source Check

PaddleOCR may print:

```text
Checking connectivity to the model hosters, this may take a while.
To bypass this check, set PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK to True.
```

To skip this connectivity check for a run:

```bash
PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK=True python -m app.cli data/sroie_sample \
  --ocr-mode single_vl \
  --output outputs/sroie_single_vl.xlsx
```

Or add this to `.env` only if your shell/process loads it for environment variables. Some libraries read this directly from the OS environment, so the shell prefix above is the most reliable method.
