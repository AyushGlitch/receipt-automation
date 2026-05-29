from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from PIL import Image

from app.config.settings import settings
from app.models.schemas import ClassificationResult, ReceiptRoute

ROUTE_TO_ENGINE = {
    ReceiptRoute.PRINTED: "paddle_ppocr",
    ReceiptRoute.HANDWRITTEN: "paddleocr_vl_1.6",
    ReceiptRoute.MIXED: "paddleocr_vl_1.6",
    ReceiptRoute.UNKNOWN: "paddleocr_vl_1.6",
}


@dataclass(frozen=True)
class ClipPrompt:
    label: str
    route: ReceiptRoute
    prompts: tuple[str, ...]


CLIP_PROMPTS = (
    ClipPrompt(
        label="printed_receipt",
        route=ReceiptRoute.PRINTED,
        prompts=(
            "a photo of a printed store receipt",
            "a scanned printed purchase receipt",
            "a printed point of sale receipt with typed text",
        ),
    ),
    ClipPrompt(
        label="handwritten_receipt",
        route=ReceiptRoute.HANDWRITTEN,
        prompts=(
            "a photo of a handwritten receipt",
            "a receipt written by hand on paper",
            "a handwritten bill with prices and total amount",
        ),
    ),
    ClipPrompt(
        label="mixed_receipt",
        route=ReceiptRoute.MIXED,
        prompts=(
            "a receipt with both printed and handwritten text",
            "a printed receipt with handwritten notes",
        ),
    ),
    ClipPrompt(
        label="non_receipt",
        route=ReceiptRoute.UNKNOWN,
        prompts=(
            "a photo that is not a receipt",
            "a document that is not a purchase receipt",
            "a random image or document",
        ),
    ),
    ClipPrompt(
        label="low_quality",
        route=ReceiptRoute.UNKNOWN,
        prompts=(
            "a blurry low quality receipt photo",
            "a dark unreadable receipt image",
            "a damaged or cropped receipt image",
        ),
    ),
)


class ReceiptRouter:
    """Chooses the OCR route according to the configured OCR architecture."""

    def __init__(self) -> None:
        self._clip = CLIPReceiptRouter()

    def route(self, image: Image.Image) -> ClassificationResult:
        if settings.ocr_mode == "single_vl":
            return ClassificationResult(
                route=ReceiptRoute.UNKNOWN,
                document_type="receipt",
                confidence=1.0,
                preprocessing_profile="vl_light",
                ocr_engine="paddleocr_vl_1.6",
            )
        if settings.enable_clip_router:
            return self._clip.route(image)
        return heuristic_route(image)


class CLIPReceiptRouter:
    """Zero-shot CLIP router for multi-model OCR mode."""

    def __init__(self) -> None:
        self._model: Any = None
        self._processor: Any = None
        self._torch: Any = None
        self._load_attempted = False

    def route(self, image: Image.Image) -> ClassificationResult:
        if not self._load_attempted:
            self._load()
        if self._model is None or self._processor is None or self._torch is None:
            return heuristic_route(image)
        try:
            return self._route_with_clip(image)
        except Exception:
            return heuristic_route(image)

    def _load(self) -> None:
        self._load_attempted = True
        try:
            import torch  # type: ignore
            from transformers import CLIPModel, CLIPProcessor  # type: ignore

            self._torch = torch
            self._processor = CLIPProcessor.from_pretrained(settings.clip_model_name)
            self._model = CLIPModel.from_pretrained(settings.clip_model_name)
            self._model.eval()
        except Exception:
            self._model = None
            self._processor = None
            self._torch = None

    def _route_with_clip(self, image: Image.Image) -> ClassificationResult:
        assert self._model is not None
        assert self._processor is not None
        assert self._torch is not None

        prompts = [prompt for group in CLIP_PROMPTS for prompt in group.prompts]
        inputs = self._processor(text=prompts, images=image.convert("RGB"), return_tensors="pt", padding=True)
        with self._torch.no_grad():
            outputs = self._model(**inputs)
            probs = outputs.logits_per_image.softmax(dim=1)[0]

        grouped_scores: list[tuple[ClipPrompt, float]] = []
        cursor = 0
        for group in CLIP_PROMPTS:
            group_probs = probs[cursor : cursor + len(group.prompts)]
            grouped_scores.append((group, float(group_probs.max().item())))
            cursor += len(group.prompts)

        best, confidence = max(grouped_scores, key=lambda item: item[1])
        if confidence < settings.clip_min_confidence:
            best = ClipPrompt("unknown", ReceiptRoute.UNKNOWN, ("unknown",))
            confidence = max(confidence, 0.0)

        return ClassificationResult(
            route=best.route,
            document_type=best.label,
            confidence=confidence,
            preprocessing_profile=preprocessing_profile_for_route(best.route),
            ocr_engine=ocr_engine_for_route(best.route),
        )


def heuristic_route(image: Image.Image) -> ClassificationResult:
    width, height = image.size
    aspect = height / max(width, 1)
    if aspect > 1.8:
        route = ReceiptRoute.PRINTED
        document_type = "printed_receipt"
        confidence = 0.58
    else:
        route = ReceiptRoute.UNKNOWN
        document_type = "unknown"
        confidence = 0.40
    return ClassificationResult(
        route=route,
        document_type=document_type,
        confidence=confidence,
        preprocessing_profile=preprocessing_profile_for_route(route),
        ocr_engine=ocr_engine_for_route(route),
    )


def preprocessing_profile_for_route(route: ReceiptRoute) -> str:
    if settings.ocr_mode == "single_vl":
        return "vl_light"
    if route == ReceiptRoute.PRINTED:
        return "printed_clean"
    if route in {ReceiptRoute.HANDWRITTEN, ReceiptRoute.MIXED}:
        return "handwritten_light"
    return "default"


def ocr_engine_for_route(route: ReceiptRoute) -> str:
    if settings.ocr_mode == "single_vl":
        return "paddleocr_vl_1.6"
    if route == ReceiptRoute.UNKNOWN and not settings.enable_vl:
        return "paddle_ppocr"
    return ROUTE_TO_ENGINE.get(route, "paddle_ppocr")
