# Model Files

Place local model artifacts here:

- `qwen3.5-2b-gguf/Qwen3.5-2B-UD-Q4_K_XL.gguf`: Unsloth Qwen3.5-2B UD `Q4_K_XL` quantized GGUF for structured extraction through `llama-cpp-python`.
- `paddleocr/`: optional local PaddleOCR model cache if you do not want PaddleOCR to manage downloads.
- CLIP ViT-B/32 is loaded through Hugging Face Transformers when `RECEIPT_OCR_MODE=multi_model`; cache it locally if you want fully offline startup.

PaddleOCR-VL 1.6 is integrated through `app/pipeline/ocr.py` using PaddleOCR's `PaddleOCRVL` API. For Apple Silicon, the default config uses CPU mode. You can also run PaddleOCR-VL recognition through a local llama.cpp server by setting the `RECEIPT_PADDLE_VL_REC_*` variables in `.env`.
