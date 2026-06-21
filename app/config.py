from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Any

import yaml
from dotenv import load_dotenv
from pydantic import BaseModel, Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class RuntimeSettings(BaseSettings):
    groq_api_key: str | None = Field(default=None, alias="GROQ_API_KEY")
    groq_base_url: str = Field(default="https://api.groq.com/openai/v1", alias="GROQ_BASE_URL")
    groq_model: str = Field(default="openai/gpt-oss-20b", alias="GROQ_MODEL")
    qdrant_url: str = Field(default="http://localhost:6333", alias="QDRANT_URL")
    qdrant_api_key: str | None = Field(default=None, alias="QDRANT_API_KEY")
    qdrant_path: str = Field(default=".qdrant", alias="QDRANT_PATH")

    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    @field_validator("groq_api_key", "qdrant_api_key", mode="before")
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
