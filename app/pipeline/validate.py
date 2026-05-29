from __future__ import annotations

from app.config.settings import settings
from app.models.schemas import ReceiptExtraction


def validate_extraction(extraction: ReceiptExtraction) -> ReceiptExtraction:
    if extraction.confidence < settings.review_confidence_threshold:
        extraction.warnings.append("low_extraction_confidence")
        extraction.needs_review = True

    if extraction.total is not None:
        expected = _expected_total(extraction)
        if expected is not None:
            tolerance = max(1.0, abs(extraction.total) * settings.total_tolerance_ratio)
            if abs(expected - extraction.total) > tolerance:
                extraction.warnings.append("total_arithmetic_mismatch")
                extraction.needs_review = True

    if extraction.total is not None and extraction.subtotal is not None:
        if extraction.total < extraction.subtotal * 0.5:
            extraction.warnings.append("total_suspiciously_low")
            extraction.needs_review = True

    extraction.warnings = list(dict.fromkeys(extraction.warnings))
    return extraction


def _expected_total(extraction: ReceiptExtraction) -> float | None:
    if extraction.subtotal is None:
        return None
    value = extraction.subtotal
    value += extraction.tax or 0.0
    value += extraction.tip or 0.0
    value -= extraction.discount or 0.0
    return value
