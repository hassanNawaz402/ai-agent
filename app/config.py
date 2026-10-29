from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import BaseModel, Field, model_validator


PROJECT_ROOT = Path(__file__).resolve().parent.parent
CONFIG_PATH = PROJECT_ROOT / "llm_configs.yml"


class ProfileConfig(BaseModel):
    provider: str
    backend: str | None = None
    model: str
    embedding_model: str


class EnvConfig(BaseModel):
    active: Literal["dev", "prod"]
    profiles: dict[str, ProfileConfig]

    @model_validator(mode="after")
    def _active_profile_exists(self) -> "EnvConfig":
        if self.active not in self.profiles:
            raise ValueError(f"active env '{self.active}' not found in profiles")
        return self


class GenerationConfig(BaseModel):
    temperature: float = 0.2
    max_output_tokens: int = 1024
    timeout_seconds: int = 60
    retries: int = 2


class RagConfig(BaseModel):
    chunk_size_tokens: int = 512
    chunk_overlap_tokens: int = 64
    top_k: int = 5
    min_similarity: float = 0.25


class EmailConfig(BaseModel):
    max_summary_words: int = 100
    chunk_size_words: int = 4000


class ChatConfig(BaseModel):
    session_turns: int = 6


class ServerConfig(BaseModel):
    host: str = "0.0.0.0"
    port: int = 8000


class DatabaseConfig(BaseModel):
    host: str = "localhost"
    port: int = 3306
    name: str
    user: str
    password_env: str = "MYSQL_PASSWORD"

    @property
    def password(self) -> str:
        return os.environ.get(self.password_env, "")


class AppConfig(BaseModel):
    env: EnvConfig
    providers: dict[str, dict[str, Any]] = Field(default_factory=dict)
    generation: GenerationConfig = Field(default_factory=GenerationConfig)
    extract_topics: list[str] = Field(default_factory=list)
    rag: RagConfig = Field(default_factory=RagConfig)
    email: EmailConfig = Field(default_factory=EmailConfig)
    chat: ChatConfig = Field(default_factory=ChatConfig)
    server: ServerConfig = Field(default_factory=ServerConfig)
    database: DatabaseConfig

    @property
    def active_profile(self) -> ProfileConfig:
        return self.env.profiles[self.env.active]

    @property
    def active_env(self) -> str:
        return self.env.active

    @property
    def conda_env_name(self) -> str:
        return f"{self.active_env}_advantage_removals"

    def provider_settings(self, name: str) -> dict[str, Any]:
        return self.providers.get(name, {})

   
    def check_required_env_vars(self) -> list[str]:
        """Return required environment variables that are missing."""
        missing: list[str] = []
        profile = self.active_profile
        provider_cfg = self.provider_settings(profile.provider)

        for key in ("api_key_env", "endpoint_env", "region_env"):
            var = provider_cfg.get(key)
            if var and not os.environ.get(var):
                missing.append(var)

        # The MySQL password may legitimately be empty on a local dev setup.
        # Check whether the variable exists, rather than whether its value is truthy.
        if self.database.password_env not in os.environ:
            missing.append(self.database.password_env)

        return missing


@lru_cache(maxsize=1)
def load_config(path: Path | None = None) -> AppConfig:
    config_path = Path(path) if path else CONFIG_PATH
    with config_path.open("r") as f:
        raw = yaml.safe_load(f)
    return AppConfig.model_validate(raw)
