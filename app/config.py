from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Any

import yaml
from dotenv import load_dotenv
from pydantic import BaseModel, Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class RuntimeSettings(BaseSettings):
    llm_model_path: str = Field(
        default="models/Qwen2.5-7B-Instruct-GGUF/Qwen2.5-7B-Instruct-Q4_K_M.gguf",
        alias="LLM_MODEL_PATH",
    )
    llm_n_ctx: int = Field(default=4096, alias="LLM_N_CTX")
    llm_n_threads: int = Field(default=8, alias="LLM_N_THREADS")
    llm_n_gpu_layers: int = Field(default=0, alias="LLM_N_GPU_LAYERS")
    llm_temperature: float = Field(default=0.0, alias="LLM_TEMPERATURE")
    llm_max_tokens: int = Field(default=1024, alias="LLM_MAX_TOKENS")
    llm_chat_format: str | None = Field(default=None, alias="LLM_CHAT_FORMAT")
    qdrant_url: str = Field(default="http://localhost:6333", alias="QDRANT_URL")
    qdrant_api_key: str | None = Field(default=None, alias="QDRANT_API_KEY")
    qdrant_path: str = Field(default=".qdrant", alias="QDRANT_PATH")

    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    @field_validator("qdrant_api_key", mode="before")
    @classmethod
    def blank_string_as_none(cls, value: str | None) -> str | None:
        if isinstance(value, str) and not value.strip():
            return None
        return value


class AppConfig(BaseModel):
    raw: dict[str, Any]
    runtime: RuntimeSettings

    @property
    def columns(self) -> list[str]:
        return list(self.raw["schema"]["columns"])

    @property
    def extraction_groups(self) -> dict[str, list[str]]:
        return dict(self.raw["extraction_groups"])

    def get(self, *keys: str, default: Any = None) -> Any:
        value: Any = self.raw
        for key in keys:
            if not isinstance(value, dict) or key not in value:
                return default
            value = value[key]
        return value


@lru_cache(maxsize=1)
def load_config(config_path: str = "config/config.yaml") -> AppConfig:
    load_dotenv()
    path = Path(config_path)
    with path.open("r", encoding="utf-8") as handle:
        raw = yaml.safe_load(handle)
    return AppConfig(raw=raw, runtime=RuntimeSettings())
