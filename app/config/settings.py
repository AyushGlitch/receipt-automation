from __future__ import annotations

from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_prefix="RECEIPT_")

    project_root: Path = Field(default_factory=lambda: Path.cwd())
    data_dir: Path = Path("data")
    output_dir: Path = Path("outputs")
    log_dir: Path = Path("logs")

    classifier_model_path: Path = Path("models/mobilevit_xxs_receipt.mlpackage")
    classifier_input_name: str = "image"
    classifier_input_type: str = "image"
    classifier_image_size: int = 256
    classifier_labels: list[str] = Field(
        default_factory=lambda: [
            "printed_receipt",
            "handwritten_receipt",
            "mixed_receipt",
            "invoice",
            "non_receipt",
            "low_quality",
        ]
    )
    classifier_mean: tuple[float, float, float] = (0.485, 0.456, 0.406)
    classifier_std: tuple[float, float, float] = (0.229, 0.224, 0.225)
    qwen_model_path: Path = Path("models/unsloth-qwen3.5-2b-ud-q4_k_xl.gguf")

    max_image_long_edge: int = 2200
    llm_context_size: int = 4096
    llm_gpu_layers: int = -1
    llm_temperature: float = 0.1

    enable_classifier: bool = True
    enable_paddleocr: bool = True
    enable_llm: bool = True
    enable_vl: bool = False

    paddle_vl_pipeline_version: str = "v1.6"
    paddle_vl_device: str = "cpu"
    paddle_vl_engine: str | None = None
    paddle_vl_use_layout_detection: bool = True
    paddle_vl_use_doc_orientation_classify: bool = False
    paddle_vl_use_doc_unwarping: bool = False
    paddle_vl_max_concurrency: int = 1
    paddle_vl_for_unknown_route: bool = True
    paddle_vl_rec_backend: str | None = None
    paddle_vl_rec_server_url: str | None = None
    paddle_vl_rec_api_model_name: str | None = None
    paddle_vl_rec_api_key: str | None = None

    review_confidence_threshold: float = 0.78
    total_tolerance_ratio: float = 0.02


settings = Settings()
