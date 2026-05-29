from __future__ import annotations

import time
from pathlib import Path

from app.models.schemas import ProcessingResult, ProcessingStatus, ReceiptRoute
from app.pipeline.router import ReceiptRouter
from app.pipeline.export_excel import export_results
from app.pipeline.extract import StructuredExtractor
from app.pipeline.ingest import discover_inputs, load_image
from app.pipeline.layout import normalize_document
from app.pipeline.ocr import OCREngine
from app.pipeline.preprocess import preprocess_default, preprocess_handwritten, preprocess_printed
from app.pipeline.validate import validate_extraction


class ReceiptPipeline:
    def __init__(self) -> None:
        self.router = ReceiptRouter()
        self.ocr = OCREngine()
        self.extractor = StructuredExtractor()

    def process_path(self, input_path: Path, output_path: Path) -> list[ProcessingResult]:
        receipts = discover_inputs(input_path)
        results = [self.process_one(receipt) for receipt in receipts]
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
        if route == ReceiptRoute.PRINTED:
            return preprocess_printed(image)
        if route in {ReceiptRoute.HANDWRITTEN, ReceiptRoute.MIXED}:
            return preprocess_handwritten(image)
        return preprocess_default(image)


def _elapsed_ms(started: float) -> int:
    return int((time.perf_counter() - started) * 1000)
