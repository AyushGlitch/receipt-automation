# Receipt Automation

Fully local receipt-to-Excel automation for an Apple Silicon MacBook Air.

The project supports two OCR architectures:

```text
single_vl   -> PaddleOCR-VL 1.6 for every receipt
multi_model -> CLIP ViT-B/32 router -> PP-OCR for printed receipts, PaddleOCR-VL for handwritten/complex receipts
```

Recommended first version:

```text
single_vl
```

It is slower, but simpler and more reliable because there is no trained classifier and no routing mistake can send a handwritten receipt to the wrong OCR path.

## Pipeline

```text
image
  -> OpenCV/Pillow preprocessing
  -> OCR mode router
  -> PaddleOCR-VL or PP-OCR
  -> layout normalization
  -> Qwen3.5 structured extraction
  -> Pydantic validation
  -> Excel export
```

## 1. Create The Python Environment

Use Python 3.11 if possible. Python 3.13 may cause dependency issues with PaddlePaddle, PaddleOCR, Core ML tooling, or llama.cpp bindings.

```bash
cd /Users/ayusharyansingh/Developer/Python/receipt-automation
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

Copy the environment template:

```bash
cp .env.example .env
```

## 2. Choose OCR Mode

### Option A: Single VL OCR

Recommended first setup:

```env
RECEIPT_OCR_MODE=single_vl
RECEIPT_ENABLE_VL=true
RECEIPT_ENABLE_PADDLEOCR=true
```

Flow:

```text
receipt -> PaddleOCR-VL 1.6 -> Qwen extraction -> Excel
```

Use this if you want:

```text
maximum simplicity
no classifier training
same path for printed and handwritten receipts
fewer moving parts
```

### Option B: Multi Model OCR

Optional faster setup:

```env
RECEIPT_OCR_MODE=multi_model
RECEIPT_ENABLE_CLIP_ROUTER=true
RECEIPT_CLIP_MODEL_NAME=openai/clip-vit-base-patch32
RECEIPT_CLIP_MIN_CONFIDENCE=0.34
RECEIPT_ENABLE_PADDLEOCR=true
RECEIPT_ENABLE_VL=true
```

Flow:

```text
receipt
  -> CLIP ViT-B/32 zero-shot routing
  -> printed receipt: PaddleOCR PP-OCR
  -> handwritten/mixed/unknown receipt: PaddleOCR-VL 1.6
```

Use this if you want:

```text
faster printed receipt processing
no classifier training
VL fallback for uncertain receipts
```

Caution: CLIP is zero-shot. It is useful as a lightweight router, but it can misclassify subtle cases. Keep PaddleOCR-VL enabled as the fallback.

## 3. Install PaddleOCR For Printed Receipts

Required for `multi_model` printed receipt fast path.

```bash
python -m pip install -U paddleocr
python -m pip install "paddlepaddle>=3.2.1"
```

Keep enabled in `.env`:

```env
RECEIPT_ENABLE_PADDLEOCR=true
```

## 4. Install PaddleOCR-VL 1.6

Required for `single_vl`. Strongly recommended for `multi_model` fallback.

Install using the official document parser extra:

```bash
python -m pip install -U "paddleocr[doc-parser]"
```

Recommended Apple Silicon settings:

```env
RECEIPT_ENABLE_VL=true
RECEIPT_PADDLE_VL_PIPELINE_VERSION=v1.6
RECEIPT_PADDLE_VL_DEVICE=cpu
RECEIPT_PADDLE_VL_USE_LAYOUT_DETECTION=true
RECEIPT_PADDLE_VL_MAX_CONCURRENCY=1
RECEIPT_PADDLE_VL_FOR_UNKNOWN_ROUTE=true
```

The project uses PaddleOCR-VL like this:

```python
from paddleocr import PaddleOCRVL

pipeline = PaddleOCRVL(pipeline_version="v1.6", device="cpu")
output = pipeline.predict("receipt.png")
```

Official docs: https://www.paddleocr.ai/main/en/version3.x/pipeline_usage/PaddleOCR-VL.html

## 5. Optional PaddleOCR-VL llama.cpp Server Mode

Use this only if you decide to run PaddleOCR-VL recognition through a local llama.cpp server.

Start `llama-server` with the PaddleOCR-VL GGUF and mmproj files:

```bash
llama-server \
  -m /path/to/PaddleOCR-VL-1.6-GGUF.gguf \
  --mmproj /path/to/PaddleOCR-VL-1.6-GGUF-mmproj.gguf \
  --port 8111 \
  --host 127.0.0.1 \
  --temp 0
```

Then set:

```env
RECEIPT_ENABLE_VL=true
RECEIPT_PADDLE_VL_REC_BACKEND=llama-cpp-server
RECEIPT_PADDLE_VL_REC_SERVER_URL=http://localhost:8111/v1
RECEIPT_PADDLE_VL_REC_API_MODEL_NAME=PaddleOCR-VL-1.6
```

## 6. Install CLIP For Multi Model OCR

Only needed when:

```env
RECEIPT_OCR_MODE=multi_model
```

Install:

```bash
python -m pip install -U transformers torch accelerate safetensors
```

The default CLIP router uses:

```env
RECEIPT_CLIP_MODEL_NAME=openai/clip-vit-base-patch32
```

For fully offline use, run once while online so Hugging Face caches the model, or pre-download it into your local Hugging Face cache.

## 7. Install Qwen3.5 Structured Extraction

This project is configured for the Unsloth Qwen3.5-2B GGUF quant:

```text
unsloth/Qwen3.5-2B-GGUF:UD-Q4_K_XL
```

Install llama.cpp Python bindings with Metal support:

```bash
CMAKE_ARGS="-DGGML_METAL=on" python -m pip install llama-cpp-python
```

Install Hugging Face download tools:

```bash
python -m pip install huggingface_hub hf_transfer
```

Download the Unsloth GGUF file:

```bash
mkdir -p models
HF_HUB_ENABLE_HF_TRANSFER=1 hf download unsloth/Qwen3.5-2B-GGUF \
  --include "*UD-Q4_K_XL*.gguf" \
  --local-dir models/qwen3.5-2b-gguf
```

Then either update `.env` to the downloaded file path, or move/rename the file to:

```text
models/qwen3.5-2b-gguf/Qwen3.5-2B-UD-Q4_K_XL.gguf
```

Required `.env` setting:

```env
RECEIPT_ENABLE_LLM=true
RECEIPT_QWEN_MODEL_PATH=models/qwen3.5-2b-gguf/Qwen3.5-2B-UD-Q4_K_XL.gguf
RECEIPT_LLM_CONTEXT_SIZE=4096
RECEIPT_LLM_GPU_LAYERS=-1
RECEIPT_LLM_TEMPERATURE=0.1
```

## 8. Optional Transformers/Hugging Face Setup For Qwen

The current extractor uses GGUF through `llama-cpp-python`, which is recommended for a 16 GB MacBook Air.

Install Transformers only if you want to experiment with a Hugging Face Qwen checkpoint instead of GGUF:

```bash
python -m pip install -U transformers accelerate safetensors sentencepiece huggingface_hub
python -m pip install torch torchvision torchaudio
```

Official Qwen Transformers docs: https://qwen.readthedocs.io/en/stable/inference/transformers.html

## 9. Add Input Receipts

Put receipt images here:

```text
data/inbox/
```

Supported image types:

```text
.jpg .jpeg .png .tif .tiff .bmp .webp
```

## 10. Run The Pipeline

For PaddleOCR-VL, start with one receipt first. CPU inference can be slow on the M4 Air:

```bash
mkdir -p data/debug
cp data/inbox/your_receipt.jpg data/debug/
PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK=True python -m app.cli data/debug --ocr-mode single_vl --output outputs/debug_one.xlsx
```

Then run the full inbox:

```bash
PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK=True python -m app.cli data/inbox --output outputs/receipts.xlsx
```

Override OCR mode for one run:

```bash
python -m app.cli data/inbox --ocr-mode multi_model --output outputs/receipts.xlsx
```

Or:

```bash
./scripts/run_inbox.sh
```

The workbook is written to:

```text
outputs/receipts.xlsx
```

It contains:

```text
Receipts
Line Items
Review Queue
Processing Logs
```

## Recommended M4 Air Settings

Start with:

```env
RECEIPT_OCR_MODE=single_vl
RECEIPT_MAX_IMAGE_LONG_EDGE=1600
RECEIPT_ENABLE_VL=true
RECEIPT_PADDLE_VL_DEVICE=cpu
RECEIPT_PADDLE_VL_MAX_CONCURRENCY=1
RECEIPT_ENABLE_LLM=true
RECEIPT_LLM_CONTEXT_SIZE=4096
RECEIPT_LLM_GPU_LAYERS=-1
```

If memory pressure is high:

```env
RECEIPT_MAX_IMAGE_LONG_EDGE=1600
RECEIPT_LLM_CONTEXT_SIZE=2048
```

If PaddleOCR-VL is too slow, switch to:

```env
RECEIPT_OCR_MODE=multi_model
RECEIPT_ENABLE_CLIP_ROUTER=true
```

## SROIE Dataset Testing

For a printed-receipt benchmark workflow, see [SROIE_TEST_PLAN.md](SROIE_TEST_PLAN.md).

## Model Cache And Cleanup

For PaddleOCR/PaddleX, Hugging Face, project output, and deep cleanup commands, see [MODEL_CACHE_CLEANUP.md](MODEL_CACHE_CLEANUP.md).

## Notes

- `single_vl` mode ignores CLIP and PP-OCR routing and sends all receipts to PaddleOCR-VL.
- `multi_model` mode uses CLIP only as a router, not as an extractor.
- In `multi_model`, unknown or low-confidence routes should stay on PaddleOCR-VL.
- Low-confidence or inconsistent extracted fields are marked in the `Review Queue` sheet.
