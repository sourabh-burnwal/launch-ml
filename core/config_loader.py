"""
LaunchML Configuration Loader — validates and deserializes ``deploy_config.yaml``.

Architecture:
    - Uses **Pydantic v2** models for strict, typed validation.
    - Every field has sensible defaults so users only need to override
      what they care about.
    - Validation errors are surfaced as human-readable messages via Rich
      before the pipeline ever starts.
    - The loader returns a frozen (immutable) ``DeployConfig`` object that
      is threaded through every node in the LangGraph.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Dict, Literal, Optional

import yaml
from pydantic import BaseModel, Field, field_validator, model_validator

from utils.logger import get_logger, error as cli_error

log = get_logger(__name__)


# ─── Sub-models ───────────────────────────────────────────────────────────────

class ModelConfig(BaseModel):
    """Pointer to the user's local model artefacts."""
    path: str = Field(..., description="Path to model directory")
    framework_hint: Optional[str] = Field(
        None,
        description="Optional hint: pytorch | tensorflow | onnx | transformers",
    )
    entrypoint: Optional[str] = Field(
        None,
        description="Custom predict.py entrypoint filename",
    )

    @field_validator("path")
    @classmethod
    def _path_must_exist(cls, v: str) -> str:
        p = Path(v).expanduser()
        if not p.exists():
            raise ValueError(f"Model path does not exist: {p}")
        return str(p)


class SchemaDefinition(BaseModel):
    """JSON-Schema-style definition for API request/response."""
    type: str = "object"
    properties: Dict[str, Any] = Field(default_factory=dict)


class APIConfig(BaseModel):
    """API contract for the served model."""
    request_schema: SchemaDefinition = Field(default_factory=SchemaDefinition)
    response_schema: SchemaDefinition = Field(default_factory=SchemaDefinition)


class DeploymentConfig(BaseModel):
    """Infrastructure / deployment knobs.

    Set ``cloud: local`` to deploy locally via Docker containers instead
    of provisioning cloud infrastructure.

    Set ``backend`` to force a specific serving backend (``fastapi``,
    ``triton``, ``seldon``, ``vertex_ai``).  Leave as ``auto`` (the
    default) to let the AI strategy agent pick the best one.
    """
    backend: Optional[str] = Field(
        None,
        description=(
            "Force a specific backend: fastapi | triton | seldon | vertex_ai. "
            "Leave unset or 'auto' to let the AI choose."
        ),
    )
    cloud: Literal["gcp", "aws", "local"] = "gcp"
    region: str = "us-central1"
    latency_target_ms: int = Field(200, ge=1)
    expected_rps: int = Field(50, ge=1)
    load_balancer: Literal["public", "private"] = "public"
    gpu_required: Literal["auto", "true", "false"] = "auto"

    @property
    def is_local(self) -> bool:
        """True when the user wants local Docker deployment."""
        return self.cloud == "local"

    @property
    def has_backend_override(self) -> bool:
        """True when the user explicitly chose a backend."""
        return self.backend is not None

    @field_validator("backend", mode="before")
    @classmethod
    def _normalise_backend(cls, v: Any) -> Optional[str]:
        """Treat ``'auto'``, empty string, and ``None`` as 'let AI decide'."""
        if v is None:
            return None
        v_str = str(v).strip().lower()
        if v_str in ("auto", ""):
            return None
        return v_str

    @field_validator("gpu_required", mode="before")
    @classmethod
    def _coerce_gpu(cls, v: Any) -> str:
        """Accept Python bools from YAML (True/False) and normalise."""
        if isinstance(v, bool):
            return str(v).lower()
        return str(v).lower()


class ObservabilityConfig(BaseModel):
    enable_metrics: bool = True
    enable_logging: bool = True


class LLMConfig(BaseModel):
    """Which LLM provider / model to use for reasoning."""
    provider: Literal["openai", "anthropic", "gemini"] = "openai"
    model: str = "gpt-4o"

    @model_validator(mode="after")
    def _check_api_key(self) -> "LLMConfig":
        """Warn (not crash) if the expected env var is missing."""
        env_map = {
            "openai": "OPENAI_API_KEY",
            "anthropic": "ANTHROPIC_API_KEY",
            "gemini": "GOOGLE_API_KEY",
        }
        key_var = env_map.get(self.provider, "")
        if key_var and not os.environ.get(key_var):
            log.warning(
                "missing_api_key",
                provider=self.provider,
                env_var=key_var,
                msg=f"Environment variable {key_var} is not set. LLM calls will fail.",
            )
        return self


# ─── Top-level config ─────────────────────────────────────────────────────────

class DeployConfig(BaseModel):
    """Root configuration object — the single source of truth for a run."""
    model: ModelConfig
    api: APIConfig = Field(default_factory=APIConfig)
    deployment: DeploymentConfig = Field(default_factory=DeploymentConfig)
    observability: ObservabilityConfig = Field(default_factory=ObservabilityConfig)
    llm: LLMConfig = Field(default_factory=LLMConfig)

    model_config = {"frozen": True}  # immutable once validated


# ─── Loader function ──────────────────────────────────────────────────────────

def load_config(path: str | Path) -> DeployConfig:
    """Load, parse, and validate a YAML config file.

    Raises:
        SystemExit: on any validation error (after printing a helpful message).
    """
    config_path = Path(path).expanduser().resolve()
    if not config_path.is_file():
        cli_error(f"Config file not found: {config_path}")
        raise SystemExit(1)

    log.info("loading_config", path=str(config_path))
    with open(config_path) as fh:
        raw = yaml.safe_load(fh)

    if not isinstance(raw, dict):
        cli_error("Config file is empty or not a valid YAML mapping.")
        raise SystemExit(1)

    try:
        cfg = DeployConfig(**raw)
    except Exception as exc:
        cli_error(f"Config validation failed:\n{exc}")
        raise SystemExit(1) from exc

    log.info("config_loaded", cloud=cfg.deployment.cloud, provider=cfg.llm.provider)
    return cfg
