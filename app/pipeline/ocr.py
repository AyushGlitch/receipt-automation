from __future__ import annotations

import json
import tempfile
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

from app.config.settings import settings
from app.models.schemas import (
    BoundingBox,
    ClassificationResult,
    ExtractedTable,
    OCRBlock,
    OCRLine,
    OCRPayload,
    OCRToken,
    ReceiptRoute,
)


class OCREngine:
    def __init__(self) -> None:
        self._ppocr = None
        self._vl = None

    def run(
        self,
        image: Image.Image,
        classification: ClassificationResult,
        source_path: Path | None = None,
    ) -> OCRPayload:
        vl_routes = {ReceiptRoute.HANDWRITTEN, ReceiptRoute.MIXED}
        if settings.paddle_vl_for_unknown_route:
            vl_routes.add(ReceiptRoute.UNKNOWN)
        if classification.route in vl_routes and settings.enable_vl:
            return self._run_paddle_vl(image, source_path)
        return self._run_ppocr(image)

    def _run_ppocr(self, image: Image.Image) -> OCRPayload:
        if not settings.enable_paddleocr:
            return OCRPayload(engine="paddle_ppocr")
        try:
            from paddleocr import PaddleOCR  # type: ignore
        except Exception as exc:
            raise RuntimeError(
                "PaddleOCR is not installed. Install PaddleOCR or set RECEIPT_ENABLE_PADDLEOCR=false."
            ) from exc

        if self._ppocr is None:
            self._ppocr = PaddleOCR(use_angle_cls=True, lang="en", show_log=False)

        result = self._ppocr.ocr(np.array(image.convert("RGB")), cls=True)
        tokens: list[OCRToken] = []
        for page in result or []:
            for item in page or []:
                box_points, recognition = item
                text, confidence = recognition
                xs = [float(point[0]) for point in box_points]
                ys = [float(point[1]) for point in box_points]
                tokens.append(
                    OCRToken(
                        text=str(text),
                        confidence=float(confidence),
                        bbox=BoundingBox(x1=min(xs), y1=min(ys), x2=max(xs), y2=max(ys)),
                    )
                )
        return OCRPayload(engine="paddle_ppocr", tokens=tokens)

    def _run_paddle_vl(self, image: Image.Image, source_path: Path | None) -> OCRPayload:
        try:
            from paddleocr import PaddleOCRVL  # type: ignore
        except Exception as exc:
            raise RuntimeError(
                "PaddleOCR-VL is not installed. Install paddleocr[doc-parser] and enable VL again."
            ) from exc

        if self._vl is None:
            kwargs: dict[str, Any] = {
                "pipeline_version": settings.paddle_vl_pipeline_version,
                "device": settings.paddle_vl_device,
                "use_layout_detection": settings.paddle_vl_use_layout_detection,
                "use_doc_orientation_classify": settings.paddle_vl_use_doc_orientation_classify,
                "use_doc_unwarping": settings.paddle_vl_use_doc_unwarping,
                "vl_rec_max_concurrency": settings.paddle_vl_max_concurrency,
            }
            if settings.paddle_vl_engine:
                kwargs["engine"] = settings.paddle_vl_engine
            if settings.paddle_vl_rec_backend:
                kwargs["vl_rec_backend"] = settings.paddle_vl_rec_backend
            if settings.paddle_vl_rec_server_url:
                kwargs["vl_rec_server_url"] = settings.paddle_vl_rec_server_url
            if settings.paddle_vl_rec_api_model_name:
                kwargs["vl_rec_api_model_name"] = settings.paddle_vl_rec_api_model_name
            if settings.paddle_vl_rec_api_key:
                kwargs["vl_rec_api_key"] = settings.paddle_vl_rec_api_key
            self._vl = PaddleOCRVL(**kwargs)

        input_path = source_path if source_path and source_path.exists() else _write_temp_image(image)
        try:
            results = list(self._vl.predict(str(input_path)))
            return _vl_results_to_payload(results)
        finally:
            if source_path is None and input_path.exists():
                input_path.unlink(missing_ok=True)


def _write_temp_image(image: Image.Image) -> Path:
    handle = tempfile.NamedTemporaryFile(prefix="receipt-vl-", suffix=".png", delete=False)
    path = Path(handle.name)
    handle.close()
    image.save(path)
    return path


def _vl_results_to_payload(results: list[Any]) -> OCRPayload:
    raw_parts: list[str] = []
    tokens: list[OCRToken] = []
    blocks: list[OCRBlock] = []
    tables: list[ExtractedTable] = []
    metadata: dict[str, Any] = {"result_count": len(results)}

    for result in results:
        data = _coerce_vl_result_to_dict(result)
        raw_parts.extend(_extract_text_fragments(data))
        tokens.extend(_extract_tokens_from_vl_dict(data))
        blocks.extend(_extract_blocks_from_vl_dict(data))
        tables.extend(_extract_tables_from_vl_dict(data))

    raw_text = "\n".join(part for part in raw_parts if part.strip())
    if raw_text and not tokens:
        tokens = [OCRToken(text=line.strip(), confidence=0.75) for line in raw_text.splitlines() if line.strip()]

    return OCRPayload(
        engine="paddleocr_vl_1.6",
        tokens=tokens,
        raw_text=raw_text,
        blocks=blocks,
        tables=tables,
        metadata=metadata,
    )


def _coerce_vl_result_to_dict(result: Any) -> dict[str, Any]:
    for attr in ("json", "res", "data"):
        value = getattr(result, attr, None)
        if isinstance(value, dict):
            return _json_safe(value)
        if callable(value):
            try:
                called = value()
            except TypeError:
                continue
            if isinstance(called, dict):
                return _json_safe(called)
            if isinstance(called, str):
                try:
                    return json.loads(called)
                except json.JSONDecodeError:
                    return {"text": called}
    if isinstance(result, dict):
        return _json_safe(result)
    return {"text": str(result)}


def _extract_text_fragments(data: Any) -> list[str]:
    fragments: list[str] = []
    if isinstance(data, dict):
        for key, value in data.items():
            lowered = str(key).lower()
            if lowered in {"text", "content", "markdown", "md", "rec_text", "html"} and isinstance(value, str):
                fragments.append(value)
            else:
                fragments.extend(_extract_text_fragments(value))
    elif isinstance(data, list):
        for item in data:
            fragments.extend(_extract_text_fragments(item))
    return fragments


def _extract_tokens_from_vl_dict(data: Any) -> list[OCRToken]:
    tokens: list[OCRToken] = []
    if isinstance(data, dict):
        text = _first_string(data, ("text", "content", "rec_text"))
        bbox = _first_bbox(data)
        confidence = _first_float(data, ("confidence", "score", "rec_score"), default=0.75)
        if text and bbox:
            tokens.append(OCRToken(text=text, bbox=bbox, confidence=confidence))
        for value in data.values():
            tokens.extend(_extract_tokens_from_vl_dict(value))
    elif isinstance(data, list):
        for item in data:
            tokens.extend(_extract_tokens_from_vl_dict(item))
    return tokens


def _extract_blocks_from_vl_dict(data: Any) -> list[OCRBlock]:
    blocks: list[OCRBlock] = []
    if isinstance(data, dict):
        label = str(data.get("label") or data.get("type") or data.get("cls") or "text")
        text = _first_string(data, ("text", "content", "rec_text"))
        bbox = _first_bbox(data)
        confidence = _first_float(data, ("confidence", "score"), default=0.75)
        if text and bbox:
            line = OCRLine(text=text, tokens=[OCRToken(text=text, bbox=bbox, confidence=confidence)], bbox=bbox, confidence=confidence)
            blocks.append(OCRBlock(label=label, lines=[line], bbox=bbox, confidence=confidence))
        for value in data.values():
            blocks.extend(_extract_blocks_from_vl_dict(value))
    elif isinstance(data, list):
        for item in data:
            blocks.extend(_extract_blocks_from_vl_dict(item))
    return blocks


def _extract_tables_from_vl_dict(data: Any) -> list[ExtractedTable]:
    tables: list[ExtractedTable] = []
    if isinstance(data, dict):
        rows = data.get("rows") or data.get("table") or data.get("cells")
        label = str(data.get("label") or data.get("type") or "table")
        if isinstance(rows, list) and rows and _looks_like_table(rows):
            tables.append(ExtractedTable(name=label, rows=_stringify_rows(rows), confidence=0.75))
        for value in data.values():
            tables.extend(_extract_tables_from_vl_dict(value))
    elif isinstance(data, list):
        for item in data:
            tables.extend(_extract_tables_from_vl_dict(item))
    return tables


def _first_string(data: dict[str, Any], keys: tuple[str, ...]) -> str | None:
    for key in keys:
        value = data.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
    return None


def _first_float(data: dict[str, Any], keys: tuple[str, ...], default: float) -> float:
    for key in keys:
        value = data.get(key)
        if isinstance(value, int | float):
            return float(value)
    return default


def _first_bbox(data: dict[str, Any]) -> BoundingBox | None:
    value = data.get("bbox") or data.get("box") or data.get("coordinate") or data.get("coordinates")
    if isinstance(value, list) and len(value) == 4:
        try:
            return BoundingBox(x1=float(value[0]), y1=float(value[1]), x2=float(value[2]), y2=float(value[3]))
        except (TypeError, ValueError):
            return None
    return None


def _looks_like_table(rows: list[Any]) -> bool:
    return all(isinstance(row, list | tuple | dict) for row in rows[:3])


def _stringify_rows(rows: list[Any]) -> list[list[str]]:
    out: list[list[str]] = []
    for row in rows:
        if isinstance(row, dict):
            out.append([str(value) for value in row.values()])
        elif isinstance(row, list | tuple):
            out.append([str(value) for value in row])
    return out


def _json_safe(value: Any) -> Any:
    try:
        json.dumps(value, default=str)
        return json.loads(json.dumps(value, default=str))
    except TypeError:
        return {"text": str(value)}
