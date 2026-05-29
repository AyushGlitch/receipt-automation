from __future__ import annotations

import json
import time
from pathlib import Path

from app.config.settings import settings
from app.models.schemas import ProcessingResult, ProcessingStatus, ReceiptRoute
from app.pipeline.export_excel import export_results
from app.pipeline.extract import StructuredExtractor
from app.pipeline.ingest import discover_inputs, load_image
from app.pipeline.layout import normalize_document
from app.pipeline.ocr import OCREngine
from app.pipeline.preprocess import preprocess_default, preprocess_handwritten, preprocess_printed
from app.pipeline.router import ReceiptRouter
from app.pipeline.validate import validate_extraction


class ReceiptPipeline:
    def __init__(self) -> None:
        self.router = ReceiptRouter()
        self.ocr = OCREngine()
        self.extractor = StructuredExtractor()

    def process_path(self, input_path: Path, output_path: Path) -> list[ProcessingResult]:
        receipts = discover_inputs(input_path)
        results: list[ProcessingResult] = []
        total = len(receipts)
        print(f"Found {total} receipt(s). OCR mode: {settings.ocr_mode}", flush=True)
        for index, receipt in enumerate(receipts, start=1):
            print(f"[{index}/{total}] Processing {receipt.source_path}", flush=True)
            result = self.process_one(receipt)
            results.append(result)
            print(
                f"[{index}/{total}] {result.status.value} in {result.processing_time_ms} ms"
                f" | route={result.classification.route.value}"
                f" | engine={result.document.ocr_engine if result.document else result.classification.ocr_engine}"
                f" | error={result.error or ''}",
                flush=True,
            )
            if settings.export_incremental:
                export_results([result], output_path)
                print(f"[{index}/{total}] Updated workbook: {output_path}", flush=True)
        if not settings.export_incremental:
            export_results(results, output_path)
        return results

    def process_one(self, receipt_input) -> ProcessingResult:
        started = time.perf_counter()
        classification = None
        document = None
        try:
            image = load_image(receipt_input)
            classification = self.router.route(image)
            processed = self._preprocess_for_route(image, classification.route)
            ocr_payload = self.ocr.run(processed, classification, receipt_input.source_path)
            document = normalize_document(receipt_input, classification, ocr_payload)
            if settings.debug_artifacts:
                _write_debug_artifact(document, ocr_payload)
            extraction = self.extractor.extract(document)
            extraction = validate_extraction(extraction)
            status = ProcessingStatus.NEEDS_REVIEW if extraction.needs_review else ProcessingStatus.ACCEPTED
            return ProcessingResult(
                receipt_input=receipt_input,
                classification=classification,
                document=document,
                extraction=extraction,
                status=status,
                processing_time_ms=_elapsed_ms(started),
            )
        except Exception as exc:
            from app.models.schemas import ClassificationResult

            return ProcessingResult(
                receipt_input=receipt_input,
                classification=classification or ClassificationResult(),
                document=document,
                status=ProcessingStatus.FAILED,
                error=str(exc),
                processing_time_ms=_elapsed_ms(started),
            )

    def _preprocess_for_route(self, image, route: ReceiptRoute):
        if settings.ocr_mode == "single_vl":
            return preprocess_default(image)
        if route == ReceiptRoute.PRINTED:
            return preprocess_printed(image)
        if route in {ReceiptRoute.HANDWRITTEN, ReceiptRoute.MIXED}:
            return preprocess_handwritten(image)
        return preprocess_default(image)


def _elapsed_ms(started: float) -> int:
    return int((time.perf_counter() - started) * 1000)


def _write_debug_artifact(document, ocr_payload) -> None:
    settings.debug_dir.mkdir(parents=True, exist_ok=True)
    path = settings.debug_dir / f"{document.receipt_id}.json"
    payload = {
        "receipt_id": document.receipt_id,
        "source_path": str(document.source_path),
        "ocr_engine": document.ocr_engine,
        "document_confidence": document.confidence,
        "raw_text": document.raw_text,
        "lines": [line.model_dump(mode="json") for line in document.lines],
        "candidates": document.candidates,
        "key_values": [kv.model_dump(mode="json") for kv in document.key_values],
        "ocr_metadata": ocr_payload.metadata,
    }
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False, default=str), encoding="utf-8")
