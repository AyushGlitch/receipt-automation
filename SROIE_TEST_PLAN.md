# SROIE Test Plan

Use this guide to test the receipt pipeline with the ICDAR 2019 SROIE printed receipt dataset.

SROIE is useful for testing:

```text
printed receipt OCR
key-field extraction
Excel export
single_vl mode
multi_model mode with CLIP + PP-OCR
```

SROIE is not a handwritten receipt dataset, so it does not fully test handwritten receipt behavior.

Dataset sources:

- https://huggingface.co/datasets/jsdnrs/ICDAR2019-SROIE
- https://huggingface.co/datasets/Voxel51/scanned_receipts

## 1. Activate The Environment

```bash
cd /Users/ayusharyansingh/Developer/Python/receipt-automation
source .venv/bin/activate
```

Use Python 3.11 if possible:

```bash
python --version
```

## 2. Install Dataset Download Tools

```bash
python -m pip install datasets huggingface_hub
```

Optional faster Hugging Face downloads:

```bash
python -m pip install hf_transfer
```

## 3. Create The Download Script

Create a script that downloads a small SROIE sample into `data/sroie_sample`:

```bash
mkdir -p scripts
cat > scripts/download_sroie_sample.py <<'PY'
from pathlib import Path
from datasets import load_dataset

out = Path("data/sroie_sample")
out.mkdir(parents=True, exist_ok=True)

dataset = load_dataset("jsdnrs/ICDAR2019-SROIE", split="test")

limit = 25
for i, row in enumerate(dataset.select(range(limit))):
    image = row["image"]
    image_path = out / f"sroie_{i:04d}.jpg"
    image.save(image_path)

    meta = {k: v for k, v in row.items() if k != "image"}
    meta_path = out / f"sroie_{i:04d}.txt"
    meta_path.write_text(str(meta), encoding="utf-8")

print(f"Saved {limit} SROIE images to {out}")
PY
```

Run it:

```bash
python scripts/download_sroie_sample.py
```

Expected output:

```text
data/sroie_sample/sroie_0000.jpg
data/sroie_sample/sroie_0000.txt
...
```

## 4. Check OCR Dependencies

Before running OCR, verify PaddleOCR-VL can see both `paddleocr` and `paddlepaddle`:

```bash
python - <<'PY'
import paddle
from paddleocr import PaddleOCRVL
print("paddle", paddle.__version__)
print("PaddleOCR-VL import ok")
PY
```

If this fails with `No module named paddle` or `paddlepaddle is not installed`, install PaddlePaddle:

```bash
python -m pip install "paddlepaddle>=3.2.1"
```

## 5. Smoke Test The Pipeline

Run the pipeline on the downloaded sample.

### Single VL Mode

This sends every receipt to PaddleOCR-VL 1.6. Test one receipt first because CPU inference can be slow:

```bash
mkdir -p data/debug
cp data/sroie_sample/sroie_0000.jpg data/debug/
PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK=True python -m app.cli data/debug \
  --ocr-mode single_vl \
  --output outputs/debug_one.xlsx
```

Then run the sample batch:

```bash
PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK=True python -m app.cli data/sroie_sample \
  --ocr-mode single_vl \
  --output outputs/sroie_single_vl.xlsx
```

Expected:

```text
no crash
outputs/sroie_single_vl.xlsx created
Processing Logs sheet shows paddleocr_vl_1.6
```

### Multi Model Mode

This uses CLIP ViT-B/32 to route printed receipts to PP-OCR:

```bash
python -m app.cli data/sroie_sample \
  --ocr-mode multi_model \
  --output outputs/sroie_multi_model.xlsx
```

Expected for SROIE:

```text
most receipts route as printed_receipt
most receipts use paddle_ppocr
uncertain receipts may use paddleocr_vl_1.6
```

## 6. Inspect The Workbooks

Open:

```text
outputs/sroie_single_vl.xlsx
outputs/sroie_multi_model.xlsx
```

Check these sheets:

```text
Receipts
Line Items
Review Queue
Processing Logs
```

In `Receipts`, inspect:

```text
merchant
date
total
confidence
needs_review
status
source_file
```

In `Processing Logs`, inspect:

```text
document_type
route
ocr_engine
classifier_confidence
ocr_confidence
processing_time_ms
status
error
```

## 7. Manual Accuracy Checklist

For each receipt image, compare against the Excel output.

Mark each row as:

```text
pass    = merchant/date/total are correct
partial = total is correct but merchant/date is wrong or missing
fail    = total is wrong or missing
```

Initial acceptance target for the first 25 samples:

```text
no crashes
80%+ have a total
60%+ have a date
merchant is plausible for most receipts
review queue catches uncertain rows
```

## 8. Compare Single VL vs Multi Model

Record:

```text
runtime
accepted count
review count
failed count
total present rate
date present rate
merchant plausibility
```

Use timing commands:

```bash
time python -m app.cli data/sroie_sample \
  --ocr-mode single_vl \
  --output outputs/sroie_single_vl_perf.xlsx
```

```bash
time python -m app.cli data/sroie_sample \
  --ocr-mode multi_model \
  --output outputs/sroie_multi_model_perf.xlsx
```

Expected tradeoff:

```text
single_vl   = slower, simpler, fewer routing mistakes
multi_model = faster for printed receipts, depends on CLIP routing quality
```

## 9. Expand The Test Size

After the 25-image smoke test passes, increase the sample size.

Edit this line in `scripts/download_sroie_sample.py`:

```python
limit = 25
```

Suggested stages:

```text
25 images
100 images
250 images
full available split
```

Run the same commands again with different output names:

```bash
python -m app.cli data/sroie_sample \
  --ocr-mode single_vl \
  --output outputs/sroie_100_single_vl.xlsx
```

```bash
python -m app.cli data/sroie_sample \
  --ocr-mode multi_model \
  --output outputs/sroie_100_multi_model.xlsx
```

## 10. Optional Ground Truth Evaluation

SROIE includes key information annotations. After confirming the exact dataset field names from the `.txt` metadata files, create a comparison script that maps:

```text
company -> merchant_name
date    -> date
total   -> total
```

Useful metrics:

```text
total_accuracy    = extracted total within 0.01 or 1%
date_accuracy     = normalized dates match
merchant_accuracy = fuzzy string ratio > 80
failure_rate      = failed / total
review_rate       = needs_review / total
avg_time_ms       = average processing time
```

Recommended Python packages for this optional step:

```bash
python -m pip install rapidfuzz pandas openpyxl
```

## 11. Debug One Receipt

If one image fails, isolate it:

```bash
mkdir -p data/debug
cp data/sroie_sample/sroie_0000.jpg data/debug/
python -m app.cli data/debug \
  --ocr-mode single_vl \
  --output outputs/debug_one.xlsx
```

Then try multi-model mode:

```bash
python -m app.cli data/debug \
  --ocr-mode multi_model \
  --output outputs/debug_one_multi_model.xlsx
```

Compare:

```text
OCR engine used
raw extraction quality
warnings
Review Queue entries
```

## 12. What SROIE Can And Cannot Tell Us

SROIE can test:

```text
printed receipt OCR
merchant/date/total extraction
Excel export
PP-OCR vs PaddleOCR-VL behavior
CLIP printed receipt routing
runtime differences
```

SROIE cannot fully test:

```text
handwritten receipts
mixed handwritten + printed receipts
phone photos with severe perspective distortion
real-world Indian GST receipt variety
```

For handwritten testing, add your own small set later:

```text
10 handwritten receipts
10 mixed receipts
10 low-quality phone photos
```

## 13. Long Running PaddleOCR-VL Notes

PaddleOCR-VL on CPU may show high CPU usage without printing logs for a while. The pipeline now prints progress per receipt and writes the workbook incrementally after each processed receipt.

If progress appears stuck, test one image first:

```bash
mkdir -p data/debug
cp data/sroie_sample/sroie_0000.jpg data/debug/
PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK=True python -m app.cli data/debug \
  --ocr-mode single_vl \
  --output outputs/debug_one.xlsx
```

Recommended M4 Air setting:

```env
RECEIPT_MAX_IMAGE_LONG_EDGE=1600
```

If it is still too slow, temporarily lower it:

```bash
RECEIPT_MAX_IMAGE_LONG_EDGE=1200 PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK=True python -m app.cli data/debug \
  --ocr-mode single_vl \
  --output outputs/debug_one_1200.xlsx
```
