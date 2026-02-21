"""
LaunchML Backend Base Class — abstract interface every deployment backend must implement.

Architecture:
    - ``DeploymentBackend`` is the ABC that enforces the plugin contract.
    - Each method represents a stage in the deployment pipeline.
    - The ``capabilities`` dict advertises what the backend supports so the
      LLM strategy agent can make an informed decision.
    - Backends self-register via the ``@register_backend`` decorator
      (see ``registry.py``).

Extensibility:
    To add a new backend, create a file ``launchml/backends/my_backend.py``,
    subclass ``DeploymentBackend``, and decorate the class with
    ``@register_backend``.  That's it — the agent will automatically
    discover it.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Optional

from launchml.core.config_loader import DeployConfig
from launchml.core.model_analyzer import ModelMetadata


@dataclass
class BackendCapabilities:
    """Machine-readable description of what a backend can do.

    The strategy agent reads these to decide which backend fits best.
    """
    supports_gpu: bool = False
    supports_batching: bool = False
    supports_autoscaling: bool = True
    supports_multi_model: bool = False
    supports_streaming: bool = False
    managed_service: bool = False
    supports_local_deployment: bool = False
    max_model_size_gb: float = 10.0
    supported_frameworks: list[str] = field(
        default_factory=lambda: ["pytorch", "tensorflow", "onnx", "transformers"]
    )
    typical_latency_ms: int = 50
    complexity: str = "low"  # low | medium | high
    cost_tier: str = "low"   # low | medium | high


class DeploymentBackend(ABC):
    """Abstract base for all deployment backends.

    Lifecycle (called in order by the agent graph):
        1. ``validate``          — can this backend handle this model + config?
        2. ``generate_inference_code`` — emit serving code (FastAPI app, Triton config, …)
        3. ``generate_terraform``      — emit IaC under ``<output_dir>/terraform/``
        4. ``deploy``                  — orchestrate the actual deployment
        5. ``setup_observability``     — wire metrics / logging
    """

    # Subclasses MUST set these
    name: str = "base"
    description: str = ""
    capabilities: BackendCapabilities = BackendCapabilities()

    def __init__(
        self,
        config: DeployConfig,
        model_metadata: ModelMetadata,
        output_dir: str | Path = "output",
    ) -> None:
        self.config = config
        self.model_metadata = model_metadata
        self.output_dir = Path(output_dir).resolve()
        self.terraform_dir = self.output_dir / "terraform"
        self.serving_dir = self.output_dir / "serving_code"

    # ------------------------------------------------------------------
    # Abstract interface
    # ------------------------------------------------------------------

    @abstractmethod
    def validate(self) -> bool:
        """Return True if this backend can serve the given model + config."""
        ...

    @abstractmethod
    def generate_inference_code(self) -> Dict[str, str]:
        """Generate serving code.

        Returns:
            Mapping of ``filename -> contents`` that will be written under
            ``<output_dir>/serving_code/``.
        """
        ...

    @abstractmethod
    def generate_terraform(self) -> Dict[str, str]:
        """Generate Terraform HCL.

        Returns:
            Mapping of ``filename -> contents`` that will be written under
            ``<output_dir>/terraform/``.
        """
        ...

    @abstractmethod
    def deploy(self) -> Dict[str, Any]:
        """Execute the deployment (or simulate it for POC).

        Returns:
            Dict with at least ``endpoint_url`` and ``status``.
        """
        ...

    @abstractmethod
    def setup_observability(self) -> Dict[str, str]:
        """Configure observability (metrics, logging).

        Returns:
            Dict with monitoring instructions / URLs.
        """
        ...

    # ------------------------------------------------------------------
    # Local deployment (optional — override in backends that support it)
    # ------------------------------------------------------------------

    def generate_local_compose(self) -> Dict[str, str]:
        """Generate ``docker-compose.yaml`` and any extra files for local dev.

        Override in backends where ``capabilities.supports_local_deployment``
        is True.  The default raises so callers never silently skip.

        Returns:
            Mapping of ``filename -> contents`` written under
            ``<output_dir>/serving_code/``.
        """
        raise NotImplementedError(
            f"Backend '{self.name}' does not support local deployment."
        )

    def deploy_local(self) -> Dict[str, Any]:
        """Build images, start containers, and expose the inference server locally.

        Override in backends where ``capabilities.supports_local_deployment``
        is True.  The default raises.

        Returns:
            Dict with at least ``endpoint_url`` and ``status``.
        """
        raise NotImplementedError(
            f"Backend '{self.name}' does not support local deployment."
        )

    # ------------------------------------------------------------------
    # Helpers available to all backends
    # ------------------------------------------------------------------

    def ensure_dirs(self) -> None:
        """Create output directories if they don't exist."""
        self.terraform_dir.mkdir(parents=True, exist_ok=True)
        self.serving_dir.mkdir(parents=True, exist_ok=True)

    def write_files(self, base_dir: Path, files: Dict[str, str]) -> list[str]:
        """Write a dict of ``{filename: content}`` under *base_dir*.

        Returns:
            List of absolute paths written.
        """
        written: list[str] = []
        for fname, content in files.items():
            dest = base_dir / fname
            dest.parent.mkdir(parents=True, exist_ok=True)
            dest.write_text(content)
            written.append(str(dest.resolve()))
        return written

    def capabilities_dict(self) -> Dict[str, Any]:
        """Serialise capabilities for the LLM prompt."""
        cap = self.capabilities
        return {
            "name": self.name,
            "description": self.description,
            "supports_gpu": cap.supports_gpu,
            "supports_batching": cap.supports_batching,
            "supports_autoscaling": cap.supports_autoscaling,
            "supports_multi_model": cap.supports_multi_model,
            "supports_streaming": cap.supports_streaming,
            "managed_service": cap.managed_service,
            "supports_local_deployment": cap.supports_local_deployment,
            "max_model_size_gb": cap.max_model_size_gb,
            "supported_frameworks": cap.supported_frameworks,
            "typical_latency_ms": cap.typical_latency_ms,
            "complexity": cap.complexity,
            "cost_tier": cap.cost_tier,
        }
