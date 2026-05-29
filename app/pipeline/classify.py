from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

from app.config.settings import settings
from app.models.schemas import ClassificationResult, ReceiptRoute

LABEL_TO_ROUTE = {
    "printed_receipt": ReceiptRoute.PRINTED,
    "printed": ReceiptRoute.PRINTED,
    "handwritten_receipt": ReceiptRoute.HANDWRITTEN,
    "handwritten": ReceiptRoute.HANDWRITTEN,
    "mixed_receipt": ReceiptRoute.MIXED,
    "mixed": ReceiptRoute.MIXED,
    "invoice": ReceiptRoute.PRINTED,
    "non_receipt": ReceiptRoute.UNKNOWN,
    "low_quality": ReceiptRoute.UNKNOWN,
    "unknown": ReceiptRoute.UNKNOWN,
}

ROUTE_TO_ENGINE = {
    ReceiptRoute.PRINTED: "paddle_ppocr",
    ReceiptRoute.HANDWRITTEN: "paddleocr_vl_1.6",
    ReceiptRoute.MIXED: "paddleocr_vl_1.6",
    ReceiptRoute.UNKNOWN: "paddleocr_vl_1.6",
}


class ReceiptClassifier:
    """MobileViT/Core ML receipt router with a lightweight heuristic fallback."""

    def __init__(self, model_path: Path | None = None) -> None:
        self.model_path = model_path or settings.classifier_model_path
        self._model = None
        self._load_attempted = False
        self._input_name: str | None = None
        self._output_name: str | None = None

    def _load(self) -> None:
        self._load_attempted = True
        if not settings.enable_classifier or not self.model_path.exists():
            return
        try:
            import coremltools as ct  # type: ignore

            self._model = ct.models.MLModel(str(self.model_path))
            spec = self._model.get_spec()
            self._input_name = spec.description.input[0].name if spec.description.input else None
            self._output_name = spec.description.output[0].name if spec.description.output else None
        except Exception:
            self._model = None
            self._input_name = None
            self._output_name = None

    def classify(self, image: Image.Image) -> ClassificationResult:
        if not self._load_attempted:
            self._load()
        if self._model is None:
            return self._heuristic_classification(image)
        try:
            return self._classify_with_coreml(image)
        except Exception:
            return self._heuristic_classification(image)

    def _classify_with_coreml(self, image: Image.Image) -> ClassificationResult:
        assert self._model is not None
        input_name = self._input_name or settings.classifier_input_name
        output = self._model.predict({input_name: self._prepare_input(image)})
        label, confidence = _extract_label_and_confidence(output, self._output_name)
        document_type = _normalize_label(label)
        route = LABEL_TO_ROUTE.get(document_type, ReceiptRoute.UNKNOWN)
        return ClassificationResult(
            route=route,
            document_type=document_type,
            confidence=confidence,
            preprocessing_profile=_preprocessing_profile_for_route(route),
            ocr_engine=_ocr_engine_for_route(route),
        )

    def _prepare_input(self, image: Image.Image) -> Image.Image | np.ndarray:
        resized = image.convert("RGB").resize(
            (settings.classifier_image_size, settings.classifier_image_size),
            Image.Resampling.BICUBIC,
        )
        if settings.classifier_input_type == "image":
            return resized
        arr = np.asarray(resized).astype("float32") / 255.0
        mean = np.array(settings.classifier_mean, dtype="float32")
        std = np.array(settings.classifier_std, dtype="float32")
        arr = (arr - mean) / std
        return np.expand_dims(arr.transpose(2, 0, 1), axis=0)

    def _heuristic_classification(self, image: Image.Image) -> ClassificationResult:
        width, height = image.size
        aspect = height / max(width, 1)
        if aspect > 1.8:
            route = ReceiptRoute.PRINTED
            document_type = "printed_receipt"
            confidence = 0.62
        else:
            route = ReceiptRoute.UNKNOWN
            document_type = "unknown"
            confidence = 0.45
        return ClassificationResult(
            route=route,
            document_type=document_type,
            confidence=confidence,
            preprocessing_profile=_preprocessing_profile_for_route(route),
            ocr_engine=_ocr_engine_for_route(route),
        )


def _extract_label_and_confidence(output: dict[str, Any], preferred_output_name: str | None) -> tuple[str, float]:
    class_label = output.get("classLabel") or output.get("class_label") or output.get("label")
    probabilities = _find_probability_dict(output, preferred_output_name)
    if probabilities:
        label = str(class_label or max(probabilities, key=probabilities.get))
        return label, float(probabilities.get(label, max(probabilities.values())))

    if class_label is not None:
        return str(class_label), 1.0

    vector = _find_probability_vector(output, preferred_output_name)
    if vector is not None:
        index = int(np.argmax(vector))
        labels = settings.classifier_labels
        label = labels[index] if index < len(labels) else str(index)
        return label, float(vector[index])

    return "unknown", 0.0


def _find_probability_dict(output: dict[str, Any], preferred_output_name: str | None) -> dict[str, float] | None:
    keys = [preferred_output_name, "classLabel_probs", "probabilities", "probs"]
    for key in keys:
        if key and isinstance(output.get(key), dict):
            return {str(k): float(v) for k, v in output[key].items()}
    for value in output.values():
        if isinstance(value, dict):
            return {str(k): float(v) for k, v in value.items()}
    return None


def _find_probability_vector(output: dict[str, Any], preferred_output_name: str | None) -> np.ndarray | None:
    keys = [preferred_output_name, "linear_0", "Identity", "probs"]
    for key in keys:
        value = output.get(key) if key else None
        if isinstance(value, np.ndarray):
            return value.reshape(-1)
        if isinstance(value, list):
            return np.asarray(value, dtype="float32").reshape(-1)
    for value in output.values():
        if isinstance(value, np.ndarray):
            return value.reshape(-1)
    return None


def _normalize_label(label: str) -> str:
    return label.strip().lower().replace(" ", "_").replace("-", "_")


def _preprocessing_profile_for_route(route: ReceiptRoute) -> str:
    if route == ReceiptRoute.PRINTED:
        return "printed_clean"
    if route in {ReceiptRoute.HANDWRITTEN, ReceiptRoute.MIXED}:
        return "handwritten_light"
    return "default"


def _ocr_engine_for_route(route: ReceiptRoute) -> str:
    if route == ReceiptRoute.UNKNOWN and not settings.enable_vl:
        return "paddle_ppocr"
    return ROUTE_TO_ENGINE.get(route, "paddle_ppocr")
