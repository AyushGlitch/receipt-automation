# Ubuntu NVIDIA T4 GPU Setup

This guide sets up the receipt automation pipeline on an Ubuntu machine with an NVIDIA T4 GPU.

The T4 is a CUDA GPU with compute capability `7.5`, so use CUDA builds and avoid Apple/Mac-specific Metal settings.

Recommended architecture on T4:

```text
printed-heavy batches -> multi_model mode
handwritten/mixed batches -> single_vl mode
```

## 1. System Requirements

Recommended:

```text
Ubuntu 22.04 or 24.04
Python 3.11
NVIDIA T4 GPU
NVIDIA driver installed
CUDA runtime available
```

Check GPU:

```bash
nvidia-smi
```

Expected: the T4 should appear in the output.

## 2. Clone Or Enter Project

```bash
cd /path/to/receipt-automation
```

If copying from Mac, make sure these folders exist:

```bash
mkdir -p data/inbox data/processed data/failed data/review outputs logs models
```

## 3. Create Python Environment

Use Python 3.11 if possible.

```bash
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip setuptools wheel
python -m pip install -r requirements.txt
```

If `python3.11` is missing on Ubuntu, install it first:

```bash
sudo apt update
sudo apt install -y python3.11 python3.11-venv python3.11-dev build-essential cmake git
```

## 4. Install PaddlePaddle GPU

Remove CPU PaddlePaddle if present:

```bash
python -m pip uninstall -y paddlepaddle paddlepaddle-gpu
```

Install PaddlePaddle GPU. For CUDA 12.6, PaddleOCR documentation commonly uses:

```bash
python -m pip install paddlepaddle-gpu==3.2.1 -i https://www.paddlepaddle.org.cn/packages/stable/cu126/
```

Then install PaddleOCR with document parser support:

```bash
python -m pip install -U "paddleocr[doc-parser]"
```

Verify Paddle GPU:

```bash
python - <<'PY'
import paddle
print("paddle", paddle.__version__)
print("cuda compiled:", paddle.device.is_compiled_with_cuda())
print("device:", paddle.device.get_device())
PY
```

Expected:

```text
cuda compiled: True
```

If this fails, check your CUDA/driver compatibility and the PaddlePaddle wheel index.

Official PaddleOCR-VL docs:

```text
https://www.paddleocr.ai/main/en/version3.x/pipeline_usage/PaddleOCR-VL.html
```

## 5. Configure PaddleOCR-VL For T4

Copy the env template:

```bash
cp .env.example .env
```

Edit `.env`:

```env
RECEIPT_ENABLE_VL=true
RECEIPT_PADDLE_VL_DEVICE=gpu:0
RECEIPT_PADDLE_VL_ENGINE=paddle
RECEIPT_PADDLE_VL_MAX_CONCURRENCY=1
RECEIPT_MAX_IMAGE_LONG_EDGE=1600
```

For T4, keep concurrency at `1` first. Increase only after checking VRAM usage with:

```bash
watch -n 1 nvidia-smi
```

## 6. Install Qwen Runtime With CUDA

The pipeline uses Unsloth Qwen3.5-2B `UD-Q4_K_XL` GGUF through `llama-cpp-python`.

Install CUDA-enabled llama.cpp Python bindings:

```bash
python -m pip uninstall -y llama-cpp-python

CMAKE_ARGS="-DGGML_CUDA=on -DCMAKE_CUDA_ARCHITECTURES=75" \
FORCE_CMAKE=1 \
python -m pip install --no-cache-dir --force-reinstall llama-cpp-python
```

T4 compute architecture is `75`.

Verify llama-cpp-python import:

```bash
python - <<'PY'
from llama_cpp import Llama
print("llama-cpp-python import ok")
PY
```

Recommended `.env`:

```env
RECEIPT_ENABLE_LLM=true
RECEIPT_LLM_GPU_LAYERS=-1
RECEIPT_LLM_CONTEXT_SIZE=4096
RECEIPT_LLM_TEMPERATURE=0.1
```

Official llama.cpp build docs:

```text
https://github.com/ggml-org/llama.cpp/blob/master/docs/build.md
```

## 7. Download Qwen GGUF

Install Hugging Face tools:

```bash
python -m pip install huggingface_hub hf_transfer
```

Download the Unsloth GGUF:

```bash
mkdir -p models
HF_HUB_ENABLE_HF_TRANSFER=1 hf download unsloth/Qwen3.5-2B-GGUF \
  --include "*UD-Q4_K_XL*.gguf" \
  --local-dir models/qwen3.5-2b-gguf
```

Set `.env`:

```env
RECEIPT_QWEN_MODEL_PATH=models/qwen3.5-2b-gguf/Qwen3.5-2B-UD-Q4_K_XL.gguf
```

Verify:

```bash
ls -lh models/qwen3.5-2b-gguf/
python - <<'PY'
from app.config.settings import settings
print(settings.qwen_model_path)
print("exists:", settings.qwen_model_path.exists())
PY
```

## 8. Optional CLIP Router For Multi Model Mode

Install CLIP dependencies:

```bash
python -m pip install -U transformers torch accelerate safetensors
```

For `multi_model` mode, set:

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
CLIP router -> printed receipts use PP-OCR
CLIP router -> handwritten/mixed/unknown use PaddleOCR-VL GPU
```

## 9. Recommended Modes

### Printed-heavy receipts

Use:

```env
RECEIPT_OCR_MODE=multi_model
RECEIPT_ENABLE_CLIP_ROUTER=true
RECEIPT_ENABLE_PADDLEOCR=true
RECEIPT_ENABLE_VL=true
RECEIPT_PADDLE_VL_DEVICE=gpu:0
```

Run:

```bash
PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK=True python -m app.cli data/inbox \
  --ocr-mode multi_model \
  --output outputs/receipts_t4_multi_model.xlsx
```

### Handwritten or mixed receipts

Use:

```env
RECEIPT_OCR_MODE=single_vl
RECEIPT_ENABLE_VL=true
RECEIPT_PADDLE_VL_DEVICE=gpu:0
```

Run:

```bash
PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK=True python -m app.cli data/inbox \
  --ocr-mode single_vl \
  --output outputs/receipts_t4_single_vl.xlsx
```

## 10. SROIE Test On T4

SROIE is printed receipts, so start with `multi_model`:

```bash
PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK=True python -m app.cli data/sroie_sample \
  --ocr-mode multi_model \
  --output outputs/sroie_t4_multi_model.xlsx
```

Then compare with full VL:

```bash
PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK=True python -m app.cli data/sroie_sample \
  --ocr-mode single_vl \
  --output outputs/sroie_t4_single_vl.xlsx
```

Monitor GPU:

```bash
watch -n 1 nvidia-smi
```

## 11. Smoke Tests

### PaddleOCR-VL Init

```bash
python - <<'PY'
from paddleocr import PaddleOCRVL
pipeline = PaddleOCRVL(pipeline_version="v1.6", device="gpu:0")
print("PaddleOCR-VL GPU init ok")
PY
```

### Qwen GGUF CUDA Load

```bash
python - <<'PY'
from llama_cpp import Llama
llm = Llama(
    model_path="models/qwen3.5-2b-gguf/Qwen3.5-2B-UD-Q4_K_XL.gguf",
    n_ctx=512,
    n_gpu_layers=-1,
    verbose=False,
)
print(llm("Return JSON only: {\"ok\": true}", max_tokens=20, temperature=0)["choices"][0]["text"])
PY
```

### One Receipt Debug

```bash
mkdir -p data/debug
cp data/inbox/your_receipt.jpg data/debug/

PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK=True python -m app.cli data/debug \
  --ocr-mode single_vl \
  --output outputs/debug_one_t4.xlsx
```

## 12. Performance Expectations

Compared with M4 Air CPU mode:

```text
PaddleOCR-VL should be much faster
CPU heat issue should disappear
Qwen extraction should be faster with CUDA offload
multi_model mode should be fastest for printed batches
```

T4 VRAM is limited, so avoid large batches/concurrency at first:

```env
RECEIPT_PADDLE_VL_MAX_CONCURRENCY=1
RECEIPT_MAX_IMAGE_LONG_EDGE=1600
RECEIPT_LLM_CONTEXT_SIZE=4096
```

If VRAM is tight:

```env
RECEIPT_MAX_IMAGE_LONG_EDGE=1200
RECEIPT_LLM_CONTEXT_SIZE=2048
```

## 13. Troubleshooting

### Paddle says CUDA is not available

Check:

```bash
nvidia-smi
python - <<'PY'
import paddle
print(paddle.device.is_compiled_with_cuda())
PY
```

If false, reinstall the correct `paddlepaddle-gpu` wheel.

### llama-cpp-python is still CPU-only

Reinstall with CUDA:

```bash
python -m pip uninstall -y llama-cpp-python
CMAKE_ARGS="-DGGML_CUDA=on -DCMAKE_CUDA_ARCHITECTURES=75" \
FORCE_CMAKE=1 \
python -m pip install --no-cache-dir --force-reinstall llama-cpp-python
```

### Out of memory

Lower:

```env
RECEIPT_MAX_IMAGE_LONG_EDGE=1200
RECEIPT_LLM_CONTEXT_SIZE=2048
RECEIPT_PADDLE_VL_MAX_CONCURRENCY=1
```

### Paddle model download checks are slow

Run with:

```bash
PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK=True python -m app.cli data/inbox \
  --ocr-mode single_vl \
  --output outputs/receipts.xlsx
```
