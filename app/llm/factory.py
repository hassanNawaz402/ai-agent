from __future__ import annotations

import logging
from functools import lru_cache

from app.config import load_config
from app.llm.client import LLMClient


log = logging.getLogger(__name__)


@lru_cache(maxsize=1)
def get_llm_client() -> LLMClient:
    cfg = load_config()
    profile = cfg.active_profile
    provider = profile.provider
    backend = profile.backend or ""
    model = profile.model

    log.info("initializing LLM client provider=%s backend=%s model=%s",
             provider, backend, model)

    if provider == "local_qwen":
        if backend == "mock":
            from app.llm.backends.mock import MockQwenClient
            return MockQwenClient(model_name=model)
        if backend == "mlx":
            from app.llm.backends.mlx_qwen import MLXQwenClient
            return MLXQwenClient(model_name=model)
        if backend == "ollama":
            from app.llm.backends.ollama_qwen import OllamaQwenClient

            provider_cfg = cfg.provider_settings("ollama")

            return OllamaQwenClient(
                model_name=model,
                base_url=provider_cfg.get(
                    "base_url", "http://127.0.0.1:11434"
                ),
                timeout_seconds=float(
                    cfg.generation.timeout_seconds
                ),
            )
        if backend in ("transformers", "vllm"):
            raise NotImplementedError(
                f"backend '{backend}' not implemented. "
                "Use backend: mlx or ollama on Mac, "
                "or provider: openai for hosted inference."
            )
        raise ValueError(
            f"unknown backend '{backend}' for provider local_qwen. "
            "Supported: mock, mlx, ollama."
        )

    if provider == "openai":
        from app.llm.backends.openai_client import OpenAIClient
        provider_cfg = cfg.provider_settings("openai")
        return OpenAIClient(
            model_name=model,
            api_key_env=provider_cfg.get("api_key_env", "OPENAI_API_KEY"),
            base_url=provider_cfg.get("base_url"),
            timeout_seconds=float(cfg.generation.timeout_seconds),
        )

    raise NotImplementedError(
        f"provider '{provider}' not implemented. "
        "Supported: local_qwen (mock|mlx|ollama), openai."
    )


def startup_connectivity_check() -> None:
    """For hosted providers, verify we can reach them before claiming ready."""
    cfg = load_config()
    if cfg.active_profile.provider == "openai":
        client = get_llm_client()
        ping = getattr(client, "ping", None)
        if ping and not ping():
            raise RuntimeError("OpenAI connectivity check failed")
