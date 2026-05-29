# Model Files

Place local model artifacts here:

- `mobilevit_xxs_receipt.mlpackage` or `.mlmodel`: MobileViT receipt routing classifier converted for Core ML / ANE.
- `unsloth-qwen3.5-2b-ud-q4_k_xl.gguf`: Unsloth Qwen3.5-2B UD `Q4_K_XL` quantized GGUF for structured extraction through `llama-cpp-python`.
- `paddleocr/`: optional local PaddleOCR model cache if you do not want PaddleOCR to manage downloads.

PaddleOCR-VL 1.6 is integrated through `app/pipeline/ocr.py` using PaddleOCR's `PaddleOCRVL` API. For Apple Silicon, the default config uses CPU mode. You can also run PaddleOCR-VL recognition through a local llama.cpp server by setting the `RECEIPT_PADDLE_VL_REC_*` variables in `.env`.
