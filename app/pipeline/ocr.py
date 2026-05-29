from __future__ import annotations

import json
import tempfile
from pathlib import Path
from typing import Any
import re

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
        if settings.ocr_mode == "single_vl":
            if not settings.enable_vl:
                raise RuntimeError("single_vl mode requires RECEIPT_ENABLE_VL=true.")
            return self._run_paddle_vl(image)

        vl_routes = {ReceiptRoute.HANDWRITTEN, ReceiptRoute.MIXED}
        if settings.paddle_vl_for_unknown_route:
            vl_routes.add(ReceiptRoute.UNKNOWN)
        if classification.route in vl_routes and settings.enable_vl:
            return self._run_paddle_vl(image)
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

    def _run_paddle_vl(self, image: Image.Image) -> OCRPayload:
        try:
            import paddle  # noqa: F401  # type: ignore
        except Exception as exc:
            raise RuntimeError(
                "PaddleOCR-VL requires paddlepaddle. Install it with: "
                'python -m pip install "paddlepaddle>=3.2.1"'
            ) from exc

        try:
            from paddleocr import PaddleOCRVL  # type: ignore
        except Exception as exc:
            raise RuntimeError(
                'PaddleOCR-VL is not installed. Install it with: python -m pip install -U "paddleocr[doc-parser]"'
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

        input_path = _write_temp_image(image)
        try:
            results = list(self._vl.predict(str(input_path)))
            return _vl_results_to_payload(results)
        finally:
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
    raw_data: list[dict[str, Any]] = []
    metadata: dict[str, Any] = {"result_count": len(results)}

    for result in results:
        data = _coerce_vl_result_to_dict(result)
        raw_data.append(data)
        raw_parts.extend(_extract_text_fragments(data))
        tokens.extend(_extract_tokens_from_vl_dict(data))
        blocks.extend(_extract_blocks_from_vl_dict(data))
        tables.extend(_extract_tables_from_vl_dict(data))

    raw_text = "\n".join(_dedupe_text_parts(raw_parts))
    if raw_text and not tokens:
        tokens = [OCRToken(text=line.strip(), confidence=0.75) for line in raw_text.splitlines() if line.strip()]

    metadata["raw_data"] = raw_data
    return OCRPayload(
        engine="paddleocr_vl_1.6",
        tokens=tokens,
        raw_text=raw_text,
        blocks=blocks,
        tables=tables,
        metadata=metadata,
    )


def _coerce_vl_result_to_dict(result: Any) -> dict[str, Any]:
    for method_name in ("json", "to_json", "to_dict"):
        value = getattr(result, method_name, None)
        if callable(value):
            try:
                called = value()
            except TypeError:
                continue
            coerced = _coerce_value_to_dict(called)
            if coerced:
                return coerced

    for attr in ("res", "data", "result", "results"):
        value = getattr(result, attr, None)
        coerced = _coerce_value_to_dict(value)
        if coerced:
            return coerced

    if isinstance(result, dict):
        return _json_safe(result)
    return {"text": str(result)}


def _coerce_value_to_dict(value: Any) -> dict[str, Any] | None:
    if isinstance(value, dict):
        return _json_safe(value)
    if isinstance(value, str) and value.strip():
        try:
            parsed = json.loads(value)
        except json.JSONDecodeError:
            return {"text": value}
        if isinstance(parsed, dict):
            return _json_safe(parsed)
        return {"text": value}
    return None


def _extract_text_fragments(data: Any) -> list[str]:
    fragments: list[str] = []
    if isinstance(data, dict):
        for key, value in data.items():
            lowered = str(key).lower()
            if isinstance(value, str) and _is_text_key(lowered):
                fragments.append(value)
            elif isinstance(value, list) and lowered in {"markdown_list", "text_list", "rec_texts"}:
                fragments.extend(str(item) for item in value if str(item).strip())
            elif isinstance(value, list) and lowered in {"parsing_res_list", "layout_parsing_result", "ocr_res"}:
                fragments.extend(_extract_text_fragments(value))
            else:
                fragments.extend(_extract_text_fragments(value))
    elif isinstance(data, list):
        for item in data:
            if isinstance(item, str) and "content:" in item:
                block = _parse_paddlex_block_string(item)
                if block and block.get("content"):
                    fragments.append(str(block["content"]))
            else:
                fragments.extend(_extract_text_fragments(item))
    elif isinstance(data, str) and "content:" in data:
        block = _parse_paddlex_block_string(data)
        if block and block.get("content"):
            fragments.append(str(block["content"]))
    return fragments


def _extract_tokens_from_vl_dict(data: Any) -> list[OCRToken]:
    tokens: list[OCRToken] = []
    if isinstance(data, dict):
        text = _first_string(
            data,
            (
                "text",
                "content",
                "rec_text",
                "markdown",
                "block_content",
            ),
        )
        bbox = _first_bbox(data)
        confidence = _first_float(data, ("confidence", "score", "rec_score"), default=0.75)
        if text and bbox:
            tokens.append(OCRToken(text=text, bbox=bbox, confidence=confidence))
        elif text and _looks_like_receipt_text(text):
            tokens.append(OCRToken(text=text, confidence=confidence))
        for value in data.values():
            tokens.extend(_extract_tokens_from_vl_dict(value))
    elif isinstance(data, list):
        for item in data:
            if isinstance(item, str) and "content:" in item:
                block = _parse_paddlex_block_string(item)
                if block and block.get("content"):
                    tokens.append(
                        OCRToken(
                            text=str(block["content"]),
                            bbox=block.get("bbox"),
                            confidence=0.75,
                        )
                    )
            else:
                tokens.extend(_extract_tokens_from_vl_dict(item))
    elif isinstance(data, str) and "content:" in data:
        block = _parse_paddlex_block_string(data)
        if block and block.get("content"):
            tokens.append(OCRToken(text=str(block["content"]), bbox=block.get("bbox"), confidence=0.75))
    return tokens


def _extract_blocks_from_vl_dict(data: Any) -> list[OCRBlock]:
    blocks: list[OCRBlock] = []
    if isinstance(data, dict):
        label = str(data.get("label") or data.get("type") or data.get("cls") or "text")
        text = _first_string(data, ("text", "content", "rec_text", "markdown", "block_content"))
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



def _parse_paddlex_block_string(value: str) -> dict[str, Any] | None:
    label_match = re.search(r"(?:^|\n)label:\s*([^\n]+)", value)
    bbox_match = re.search(r"(?:^|\n)bbox:\s*\[([^\]]+)\]", value)
    content_match = re.search(r"(?:^|\n)content:\s*(.*?)(?:\n#+|$)", value, flags=re.S)
    content = content_match.group(1).strip() if content_match else ""
    if not content or content.lower() in {"text", "paragraph_title", "table"}:
        return None

    bbox = None
    if bbox_match:
        try:
            coords = [float(part.strip()) for part in bbox_match.group(1).split(",")]
            if len(coords) == 4:
                bbox = BoundingBox(x1=coords[0], y1=coords[1], x2=coords[2], y2=coords[3])
        except ValueError:
            bbox = None

    return {
        "label": label_match.group(1).strip() if label_match else "text",
        "bbox": bbox,
        "content": content,
    }

def _is_text_key(key: str) -> bool:
    return key in {
        "text",
        "content",
        "markdown",
        "md",
        "rec_text",
        "html",
        "block_content",
        "description",
    }


def _looks_like_receipt_text(text: str) -> bool:
    lowered = text.lower()
    return any(marker in lowered for marker in ("total", "tax", "cash", "date", "rm", "gst", "receipt")) or len(text) > 12


def _dedupe_text_parts(parts: list[str]) -> list[str]:
    seen: set[str] = set()
    out: list[str] = []
    for part in parts:
        value = part.strip()
        key = value.lower()
        if value and key not in seen:
            seen.add(key)
            out.append(value)
    return out


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
