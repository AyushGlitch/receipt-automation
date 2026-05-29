# Receipt Automation

Fully local receipt-to-Excel automation for an Apple Silicon MacBook Air.

The pipeline is designed to use cheap stages first and heavier models only when needed:

```text
image -> preprocessing -> classifier -> OCR/VL -> layout normalization -> Qwen extraction -> validation -> Excel
```

## What This Project Uses

- OpenCV + Pillow for image cleanup.
- MobileViT-XXS/Core ML for printed-vs-handwritten routing.
- PaddleOCR PP-OCR for printed receipts.
- PaddleOCR-VL 1.6 for handwritten, mixed, or complex receipts.
- Unsloth Qwen3.5-2B UD-Q4_K_XL GGUF for structured extraction through llama.cpp.
- Pydantic v2 for schema validation.
- openpyxl for Excel export.

## 1. Create The Python Environment

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

## 2. Add The MobileViT Receipt Classifier

The classifier is a lightweight MobileViT-XXS Core ML model used only for routing:

```text
printed -> PaddleOCR PP-OCR
handwritten/mixed/unknown -> PaddleOCR-VL 1.6
```

Place the trained Core ML model here:

```text
models/mobilevit_xxs_receipt.mlpackage
```

Recommended classes:

```text
printed_receipt
handwritten_receipt
mixed_receipt
invoice
non_receipt
low_quality
```

`.env` defaults:

```env
RECEIPT_ENABLE_CLASSIFIER=true
RECEIPT_CLASSIFIER_MODEL_PATH=models/mobilevit_xxs_receipt.mlpackage
RECEIPT_CLASSIFIER_INPUT_NAME=image
RECEIPT_CLASSIFIER_INPUT_TYPE=image
RECEIPT_CLASSIFIER_IMAGE_SIZE=256
```

If the model file is missing, the pipeline uses a conservative heuristic fallback and sends unknown receipts to PaddleOCR-VL when VL is enabled.

## 3. Install PaddleOCR For Printed Receipts

Install PaddleOCR using the official package path:

```bash
python -m pip install -U paddleocr
```

For PaddlePaddle, install the current package supported by PaddleOCR for your platform. On Apple Silicon this normally runs on CPU:

```bash
python -m pip install "paddlepaddle>=3.2.1"
```

Keep this enabled in `.env`:

```env
RECEIPT_ENABLE_PADDLEOCR=true
```

## 4. Install PaddleOCR-VL 1.6 For Handwritten/Complex Receipts

Install the official PaddleOCR document parser extra:

```bash
python -m pip install -U "paddleocr[doc-parser]"
```

Enable PaddleOCR-VL in `.env`:

```env
RECEIPT_ENABLE_VL=true
RECEIPT_PADDLE_VL_PIPELINE_VERSION=v1.6
RECEIPT_PADDLE_VL_DEVICE=cpu
RECEIPT_PADDLE_VL_USE_LAYOUT_DETECTION=true
RECEIPT_PADDLE_VL_MAX_CONCURRENCY=1
RECEIPT_PADDLE_VL_FOR_UNKNOWN_ROUTE=true
```

The official PaddleOCR-VL Python API used by this project is:

```python
from paddleocr import PaddleOCRVL

pipeline = PaddleOCRVL(pipeline_version="v1.6", device="cpu")
output = pipeline.predict("receipt.png")
```

Official docs: https://www.paddleocr.ai/main/en/version3.x/pipeline_usage/PaddleOCR-VL.html

## 5. Optional PaddleOCR-VL llama.cpp Server Mode

You only need this if you choose to run PaddleOCR-VL recognition through a local llama.cpp server instead of direct PaddleOCR-VL inference.

Download the PaddleOCR-VL 1.6 GGUF and mmproj files from the official PaddleOCR-VL GGUF repo referenced in the PaddleOCR docs, then start `llama-server`:

```bash
llama-server \
  -m /path/to/PaddleOCR-VL-1.6-GGUF.gguf \
  --mmproj /path/to/PaddleOCR-VL-1.6-GGUF-mmproj.gguf \
  --port 8111 \
  --host 127.0.0.1 \
  --temp 0
```

Then set `.env`:

```env
RECEIPT_ENABLE_VL=true
RECEIPT_PADDLE_VL_REC_BACKEND=llama-cpp-server
RECEIPT_PADDLE_VL_REC_SERVER_URL=http://localhost:8111/v1
RECEIPT_PADDLE_VL_REC_API_MODEL_NAME=PaddleOCR-VL-1.6
```

## 6. Install Qwen3.5 Structured Extraction

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
models/unsloth-qwen3.5-2b-ud-q4_k_xl.gguf
```

Required `.env` setting:

```env
RECEIPT_ENABLE_LLM=true
RECEIPT_QWEN_MODEL_PATH=models/unsloth-qwen3.5-2b-ud-q4_k_xl.gguf
RECEIPT_LLM_CONTEXT_SIZE=4096
RECEIPT_LLM_GPU_LAYERS=-1
RECEIPT_LLM_TEMPERATURE=0.1
```

Unsloth docs: https://unsloth.ai/docs/models/qwen3.5

## 7. Optional Transformers/Hugging Face Setup

The current pipeline uses GGUF through `llama-cpp-python`, which is recommended for this MacBook Air setup.

Install Transformers only if you want to experiment with Hugging Face checkpoints or add a non-GGUF extractor later:

```bash
python -m pip install -U transformers accelerate safetensors sentencepiece huggingface_hub
```

For PyTorch on macOS:

```bash
python -m pip install torch torchvision torchaudio
```

For Qwen models in Transformers, use the latest `transformers` version. Qwen’s docs show usage through Hugging Face `pipeline()` or `generate()` APIs.

Official Qwen Transformers docs: https://qwen.readthedocs.io/en/stable/inference/transformers.html

## 8. Add Input Receipts

Put receipt images here:

```text
data/inbox/
```

Supported image types:

```text
.jpg .jpeg .png .tif .tiff .bmp .webp
```

## 9. Run The Pipeline

```bash
python -m app.cli data/inbox --output outputs/receipts.xlsx
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

Use these defaults in `.env` first:

```env
RECEIPT_MAX_IMAGE_LONG_EDGE=2200
RECEIPT_ENABLE_PADDLEOCR=true
RECEIPT_ENABLE_VL=true
RECEIPT_PADDLE_VL_DEVICE=cpu
RECEIPT_PADDLE_VL_MAX_CONCURRENCY=1
RECEIPT_ENABLE_LLM=true
RECEIPT_LLM_CONTEXT_SIZE=4096
RECEIPT_LLM_GPU_LAYERS=-1
```

If memory pressure is high, reduce:

```env
RECEIPT_MAX_IMAGE_LONG_EDGE=1600
RECEIPT_LLM_CONTEXT_SIZE=2048
```

## Notes

- PP-OCR is the default path for printed receipts.
- PaddleOCR-VL is used for handwritten, mixed, and unknown-route receipts when enabled.
- Unknown receipts route to PaddleOCR-VL by default while the MobileViT classifier is missing or still being trained.
- Low-confidence or inconsistent extractions are marked in the `Review Queue` sheet.
