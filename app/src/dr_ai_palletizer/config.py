"""Application settings loaded from environment variables."""

from __future__ import annotations

from pydantic import field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Configuration for the app server.

    All fields are populated from environment variables (case-insensitive).
    A ``.env`` file in the working directory is loaded automatically.
    """

    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    sim_server_url: str = "http://sim-server:8100"
    inference_server_url: str = "http://inference-server:8200/v1"
    inference_model: str = "nvidia/Cosmos-Reason2-2B"
    lora_model: str = ""
    request_timeout: float = 30.0
    app_host: str = "0.0.0.0"
    app_port: int = 8000
    max_thinking_tokens: int = 512
    max_completion_tokens: int = 2048
    lora_adapter_path: str = ""
    step_log_dir: str = ""
    # Demo mode: when true, the server bypasses sim/inference clients and
    # runs an in-process synthetic event generator. Used for CI smoke tests
    # only -- the cloud build runs real Conductor inference.
    demo_mode: bool = False
    cors_origins: str = "*"

    # NVIDIA hosted endpoint (build.nvidia.com -> integrate.api.nvidia.com).
    # When `nvidia_api_key` is set the Conductor team uses these endpoints
    # instead of a self-hosted vLLM server, so the cloud build needs no GPU.
    nvidia_api_key: str = ""
    nvidia_base_url: str = "https://integrate.api.nvidia.com/v1"

    # Per-agent model selection for the Conductor team.
    # These are NVIDIA-catalog model IDs; override per-deployment via env.
    perception_model: str = "meta/llama-3.2-11b-vision-instruct"
    planner_model: str = "nvidia/llama-3.1-nemotron-70b-instruct"
    verifier_model: str = "meta/llama-3.1-8b-instruct"
    conductor_model: str = "nvidia/llama-3.1-nemotron-70b-instruct"

    @property
    def use_nvidia_endpoint(self) -> bool:
        """True when the Conductor team should call NVIDIA's hosted endpoints."""
        return bool(self.nvidia_api_key.strip())

    @field_validator("lora_adapter_path", mode="before")
    @classmethod
    def _strip_lora_path(cls, v: str) -> str:
        return v.strip() if isinstance(v, str) else v

    @property
    def active_model(self) -> str:
        """Return the LoRA adapter model name if configured, else the base model."""
        return self.lora_model or self.inference_model
