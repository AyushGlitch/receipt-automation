from __future__ import annotations

import re
from statistics import mean

from app.models.schemas import (
    BoundingBox,
    ClassificationResult,
    KeyValueCandidate,
    NormalizedReceiptDocument,
    OCRLine,
    OCRPayload,
    OCRToken,
    ReceiptInput,
)

AMOUNT_RE = re.compile(r"(?:rs\.?|inr|₹|\$)?\s*[-+]?\d{1,6}(?:[,\s]\d{3})*(?:\.\d{1,2})?", re.I)
DATE_RE = re.compile(r"\b(?:\d{1,2}[/-]\d{1,2}[/-]\d{2,4}|\d{4}[/-]\d{1,2}[/-]\d{1,2})\b")
TOTAL_KEYWORDS = ("total", "grand total", "amount due", "net amount", "balance")
TAX_KEYWORDS = ("tax", "gst", "cgst", "sgst", "igst", "vat")
SUBTOTAL_KEYWORDS = ("subtotal", "sub total", "sub-total")


def normalize_document(
    receipt_input: ReceiptInput,
    classification: ClassificationResult,
    payload: OCRPayload,
) -> NormalizedReceiptDocument:
    lines = group_tokens_into_lines(payload.tokens)
    if not lines and payload.raw_text:
        lines = [
            OCRLine(text=line.strip(), confidence=0.75)
            for line in payload.raw_text.splitlines()
            if line.strip()
        ]
    raw_text = payload.raw_text or "\n".join(line.text for line in lines)
    candidates = extract_candidates(lines)
    key_values = extract_key_values(lines)
    token_confidence = mean([token.confidence for token in payload.tokens]) if payload.tokens else 0.0
    line_confidence = mean([line.confidence for line in lines]) if lines else 0.0
    confidence = token_confidence or line_confidence
    return NormalizedReceiptDocument(
        receipt_id=receipt_input.receipt_id,
        source_path=receipt_input.source_path,
        route=classification.route,
        ocr_engine=payload.engine or classification.ocr_engine,
        raw_text=raw_text,
        lines=lines,
        blocks=payload.blocks,
        tables=payload.tables,
        key_values=key_values,
        candidates=candidates,
        confidence=confidence,
    )


def group_tokens_into_lines(tokens: list[OCRToken]) -> list[OCRLine]:
    positioned = [token for token in tokens if token.bbox is not None and token.text.strip()]
    unpositioned = [token for token in tokens if token.bbox is None and token.text.strip()]
    positioned.sort(key=lambda token: (token.bbox.cy if token.bbox else 0, token.bbox.x1 if token.bbox else 0))

    line_groups: list[list[OCRToken]] = []
    for token in positioned:
        assert token.bbox is not None
        placed = False
        for group in line_groups:
            group_box = _merge_boxes([t.bbox for t in group if t.bbox])
            avg_height = mean([t.bbox.height for t in group if t.bbox]) if group else token.bbox.height
            if abs(token.bbox.cy - group_box.cy) <= max(8.0, avg_height * 0.55):
                group.append(token)
                placed = True
                break
        if not placed:
            line_groups.append([token])

    lines: list[OCRLine] = []
    for group in line_groups:
        group.sort(key=lambda token: token.bbox.x1 if token.bbox else 0)
        boxes = [token.bbox for token in group if token.bbox]
        lines.append(
            OCRLine(
                text=" ".join(token.text.strip() for token in group),
                tokens=group,
                bbox=_merge_boxes(boxes),
                confidence=mean([token.confidence for token in group]),
            )
        )

    for token in unpositioned:
        lines.append(OCRLine(text=token.text, tokens=[token], confidence=token.confidence))

    lines.sort(key=lambda line: (line.bbox.y1 if line.bbox else 10**9, line.bbox.x1 if line.bbox else 0))
    return lines


def extract_candidates(lines: list[OCRLine]) -> dict[str, list[str]]:
    text_lines = [line.text.strip() for line in lines if line.text.strip()]
    amounts: list[str] = []
    dates: list[str] = []
    totals: list[str] = []
    taxes: list[str] = []
    subtotals: list[str] = []

    for line in text_lines:
        lowered = line.lower()
        found_amounts = [match.group(0).strip() for match in AMOUNT_RE.finditer(line)]
        amounts.extend(found_amounts)
        dates.extend(match.group(0) for match in DATE_RE.finditer(line))
        if any(keyword in lowered for keyword in TOTAL_KEYWORDS):
            totals.extend(found_amounts)
        if any(keyword in lowered for keyword in TAX_KEYWORDS):
            taxes.extend(found_amounts)
        if any(keyword in lowered for keyword in SUBTOTAL_KEYWORDS):
            subtotals.extend(found_amounts)

    merchant_candidates = [line for line in text_lines[:5] if not AMOUNT_RE.search(line)]
    return {
        "merchant": _dedupe(merchant_candidates),
        "dates": _dedupe(dates),
        "amounts": _dedupe(amounts),
        "totals": _dedupe(totals),
        "taxes": _dedupe(taxes),
        "subtotals": _dedupe(subtotals),
    }


def extract_key_values(lines: list[OCRLine]) -> list[KeyValueCandidate]:
    key_values: list[KeyValueCandidate] = []
    for line in lines:
        lowered = line.text.lower()
        amounts = [match.group(0).strip() for match in AMOUNT_RE.finditer(line.text)]
        if not amounts:
            continue
        if any(keyword in lowered for keyword in TOTAL_KEYWORDS):
            key_values.append(KeyValueCandidate(key="total", value=amounts[-1], confidence=line.confidence, bbox=line.bbox))
        elif any(keyword in lowered for keyword in SUBTOTAL_KEYWORDS):
            key_values.append(KeyValueCandidate(key="subtotal", value=amounts[-1], confidence=line.confidence, bbox=line.bbox))
        elif any(keyword in lowered for keyword in TAX_KEYWORDS):
            key_values.append(KeyValueCandidate(key="tax", value=amounts[-1], confidence=line.confidence, bbox=line.bbox))
    return key_values


def _merge_boxes(boxes: list[BoundingBox]) -> BoundingBox:
    return BoundingBox(
        x1=min(box.x1 for box in boxes),
        y1=min(box.y1 for box in boxes),
        x2=max(box.x2 for box in boxes),
        y2=max(box.y2 for box in boxes),
    )


def _dedupe(values: list[str]) -> list[str]:
    seen: set[str] = set()
    out: list[str] = []
    for value in values:
        normalized = value.strip()
        key = normalized.lower()
        if normalized and key not in seen:
            seen.add(key)
            out.append(normalized)
    return out
