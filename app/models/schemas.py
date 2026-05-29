from __future__ import annotations

from datetime import date, datetime
from enum import StrEnum
from pathlib import Path
from typing import Any
from uuid import uuid4

from pydantic import BaseModel, Field, field_validator, model_validator


class ReceiptRoute(StrEnum):
    PRINTED = "printed"
    HANDWRITTEN = "handwritten"
    MIXED = "mixed"
    UNKNOWN = "unknown"


class ProcessingStatus(StrEnum):
    ACCEPTED = "accepted"
    NEEDS_REVIEW = "needs_review"
    FAILED = "failed"


class BoundingBox(BaseModel):
    x1: float
    y1: float
    x2: float
    y2: float

    @property
    def width(self) -> float:
        return max(0.0, self.x2 - self.x1)

    @property
    def height(self) -> float:
        return max(0.0, self.y2 - self.y1)

    @property
    def cy(self) -> float:
        return (self.y1 + self.y2) / 2


class ReceiptInput(BaseModel):
    receipt_id: str = Field(default_factory=lambda: uuid4().hex)
    source_path: Path
    created_at: datetime = Field(default_factory=datetime.utcnow)
    metadata: dict[str, Any] = Field(default_factory=dict)


class ClassificationResult(BaseModel):
    route: ReceiptRoute = ReceiptRoute.UNKNOWN
    document_type: str = "unknown"
    confidence: float = 0.0
    preprocessing_profile: str = "default"
    ocr_engine: str = "paddle_ppocr"


class OCRToken(BaseModel):
    text: str
    bbox: BoundingBox | None = None
    confidence: float = 0.0


class OCRLine(BaseModel):
    text: str
    tokens: list[OCRToken] = Field(default_factory=list)
    bbox: BoundingBox | None = None
    confidence: float = 0.0


class OCRBlock(BaseModel):
    label: str = "text"
    lines: list[OCRLine] = Field(default_factory=list)
    bbox: BoundingBox | None = None
    confidence: float = 0.0


class ExtractedTable(BaseModel):
    name: str = "line_items"
    rows: list[list[str]] = Field(default_factory=list)
    confidence: float = 0.0


class KeyValueCandidate(BaseModel):
    key: str
    value: str
    confidence: float = 0.0
    bbox: BoundingBox | None = None


class OCRPayload(BaseModel):
    engine: str
    tokens: list[OCRToken] = Field(default_factory=list)
    raw_text: str = ""
    blocks: list[OCRBlock] = Field(default_factory=list)
    tables: list[ExtractedTable] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)


class NormalizedReceiptDocument(BaseModel):
    receipt_id: str
    source_path: Path
    route: ReceiptRoute
    ocr_engine: str
    raw_text: str = ""
    lines: list[OCRLine] = Field(default_factory=list)
    blocks: list[OCRBlock] = Field(default_factory=list)
    tables: list[ExtractedTable] = Field(default_factory=list)
    key_values: list[KeyValueCandidate] = Field(default_factory=list)
    candidates: dict[str, list[str]] = Field(default_factory=dict)
    confidence: float = 0.0


class LineItem(BaseModel):
    description: str
    quantity: float | None = None
    unit_price: float | None = None
    total_price: float | None = None
    confidence: float = 0.0


class ReceiptExtraction(BaseModel):
    receipt_id: str
    source_path: Path
    merchant_name: str | None = None
    date: date | None = None
    currency: str | None = None
    subtotal: float | None = None
    tax: float | None = None
    tip: float | None = None
    discount: float | None = None
    total: float | None = None
    payment_method: str | None = None
    category: str | None = None
    line_items: list[LineItem] = Field(default_factory=list)
    confidence: float = 0.0
    needs_review: bool = False
    warnings: list[str] = Field(default_factory=list)
    raw_model_output: str | None = None

    @field_validator("currency")
    @classmethod
    def normalize_currency(cls, value: str | None) -> str | None:
        if value is None:
            return None
        normalized = value.strip().upper()
        aliases = {"₹": "INR", "RS": "INR", "RS.": "INR", "$": "USD"}
        return aliases.get(normalized, normalized)

    @model_validator(mode="after")
    def flag_missing_core_fields(self) -> ReceiptExtraction:
        if not self.merchant_name:
            self.warnings.append("merchant_name_missing")
        if self.total is None:
            self.warnings.append("total_missing")
        if self.date is None:
            self.warnings.append("date_missing")
        if self.warnings:
            self.needs_review = True
        return self


class ProcessingResult(BaseModel):
    receipt_input: ReceiptInput
    classification: ClassificationResult
    document: NormalizedReceiptDocument | None = None
    extraction: ReceiptExtraction | None = None
    status: ProcessingStatus
    error: str | None = None
    processing_time_ms: int = 0
