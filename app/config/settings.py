from __future__ import annotations

from pathlib import Path
from typing import Literal

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_prefix="RECEIPT_")

    project_root: Path = Field(default_factory=lambda: Path.cwd())
    data_dir: Path = Path("data")
    output_dir: Path = Path("outputs")
    log_dir: Path = Path("logs")

    ocr_mode: Literal["single_vl", "multi_model"] = "single_vl"
    enable_clip_router: bool = True
    clip_model_name: str = "openai/clip-vit-base-patch32"
    clip_min_confidence: float = 0.34
    qwen_model_path: Path = Path("models/qwen3.5-2b-gguf/Qwen3.5-2B-UD-Q4_K_XL.gguf")

    max_image_long_edge: int = 1600
    export_incremental: bool = True
    debug_artifacts: bool = True
    debug_dir: Path = Path("outputs/debug_artifacts")
    llm_context_size: int = 4096
    llm_gpu_layers: int = -1
    llm_temperature: float = 0.1

    enable_classifier: bool = False
    enable_paddleocr: bool = True
    enable_llm: bool = True
    enable_vl: bool = True

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
