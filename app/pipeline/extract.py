from __future__ import annotations

import json
import re
from datetime import date
from pathlib import Path
from typing import Any

from dateutil import parser as date_parser

from app.config.settings import settings
from app.models.schemas import NormalizedReceiptDocument, ReceiptExtraction

NUMBER_RE = re.compile(r"[-+]?\d+(?:[,\s]\d{3})*(?:\.\d{1,2})?")


class StructuredExtractor:
    def __init__(self, model_path: Path | None = None) -> None:
        self.model_path = model_path or settings.qwen_model_path
        self._llm = None
        self._load_attempted = False

    def extract(self, document: NormalizedReceiptDocument) -> ReceiptExtraction:
        if settings.enable_llm and self.model_path.exists():
            try:
                return self._extract_with_llm(document)
            except Exception as exc:
                fallback = self._extract_with_rules(document)
                fallback.warnings.append(f"llm_fallback:{exc.__class__.__name__}")
                fallback.needs_review = True
                return fallback
        extraction = self._extract_with_rules(document)
        extraction.warnings.append("llm_model_missing_rule_fallback_used")
        extraction.needs_review = True
        return extraction

    def _load_llm(self) -> None:
        self._load_attempted = True
        if self._llm is not None:
            return
        from llama_cpp import Llama  # type: ignore

        self._llm = Llama(
            model_path=str(self.model_path),
            n_ctx=settings.llm_context_size,
            n_gpu_layers=settings.llm_gpu_layers,
            verbose=False,
        )

    def _extract_with_llm(self, document: NormalizedReceiptDocument) -> ReceiptExtraction:
        if not self._load_attempted:
            self._load_llm()
        assert self._llm is not None
        prompt = build_extraction_prompt(document)
        response = self._llm(
            prompt,
            max_tokens=900,
            temperature=settings.llm_temperature,
            stop=["</json>", "<|end|>"],
        )
        raw = response["choices"][0]["text"]
        payload = parse_json_object(raw)
        payload.setdefault("receipt_id", document.receipt_id)
        payload.setdefault("source_path", document.source_path)
        extraction = ReceiptExtraction.model_validate(payload)
        extraction.raw_model_output = raw
        return extraction

    def _extract_with_rules(self, document: NormalizedReceiptDocument) -> ReceiptExtraction:
        kv = {item.key: item.value for item in document.key_values}
        dates = document.candidates.get("dates", [])
        merchant_candidates = document.candidates.get("merchant", [])
        total = parse_amount(kv.get("total") or last_value(document.candidates.get("totals", [])))
        subtotal = parse_amount(kv.get("subtotal") or last_value(document.candidates.get("subtotals", [])))
        tax = parse_amount(kv.get("tax") or last_value(document.candidates.get("taxes", [])))
        parsed_date = parse_date(dates[0]) if dates else None
        confidence = document.confidence * 0.75
        return ReceiptExtraction(
            receipt_id=document.receipt_id,
            source_path=document.source_path,
            merchant_name=merchant_candidates[0] if merchant_candidates else None,
            date=parsed_date,
            currency=infer_currency(document.raw_text),
            subtotal=subtotal,
            tax=tax,
            total=total,
            line_items=[],
            confidence=confidence,
            needs_review=confidence < settings.review_confidence_threshold,
        )


def build_extraction_prompt(document: NormalizedReceiptDocument) -> str:
    compact = {
        "receipt_id": document.receipt_id,
        "route": document.route,
        "raw_text": document.raw_text[:6000],
        "lines": [
            {
                "text": line.text,
                "bbox": line.bbox.model_dump() if line.bbox else None,
                "confidence": line.confidence,
            }
            for line in document.lines[:120]
        ],
        "candidates": document.candidates,
        "key_values": [kv.model_dump() for kv in document.key_values],
    }
    return (
        "You extract receipt data. Return one strict JSON object only, no markdown.\n"
        "Required keys: merchant_name, date, currency, subtotal, tax, tip, discount, total, "
        "payment_method, category, line_items, confidence, needs_review, warnings.\n"
        "Use ISO date format YYYY-MM-DD. Use numbers for amounts. Use null when unknown.\n"
        "Line items must contain description, quantity, unit_price, total_price, confidence.\n"
        "<receipt>\n"
        f"{json.dumps(compact, default=str, ensure_ascii=False)}\n"
        "</receipt>\nJSON:"
    )


def parse_json_object(raw: str) -> dict[str, Any]:
    start = raw.find("{")
    end = raw.rfind("}")
    if start == -1 or end == -1 or end <= start:
        raise ValueError("No JSON object found in LLM output")
    return json.loads(raw[start : end + 1])


def parse_amount(value: str | None) -> float | None:
    if not value:
        return None
    match = NUMBER_RE.search(value.replace(",", ""))
    if not match:
        return None
    return float(match.group(0).replace(" ", ""))


def parse_date(value: str) -> date | None:
    try:
        return date_parser.parse(value, dayfirst=True).date()
    except Exception:
        return None


def infer_currency(text: str) -> str | None:
    lowered = text.lower()
    if "₹" in text or "rs" in lowered or "inr" in lowered or "gst" in lowered:
        return "INR"
    if "$" in text:
        return "USD"
    return None


def last_value(values: list[str] | None) -> str | None:
    if not values:
        return None
    return values[-1]
