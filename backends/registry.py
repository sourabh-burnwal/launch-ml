"""
LaunchML Backend Registry — auto-discovery and registration of deployment backends.

Architecture:
    - Backends register themselves using the ``@register_backend`` class
      decorator.
    - At import time the registry is populated; the agent graph calls
      ``get_all_backends()`` to enumerate capabilities.
    - ``get_backend(name, config, metadata)`` instantiates a backend by name.

Extensibility:
    Adding a backend is as simple as:

        # backends/my_new_backend.py
        from backends.base import DeploymentBackend, BackendCapabilities
        from backends.registry import register_backend

        @register_backend
        class MyNewBackend(DeploymentBackend):
            name = "my_new"
            ...

    Then add an import in ``backends/__init__.py`` so it is picked up.
"""

from __future__ import annotations

import importlib
import pkgutil
from typing import Any, Dict, Type

from backends.base import DeploymentBackend
from core.config_loader import DeployConfig
from core.model_analyzer import ModelMetadata
from utils.logger import get_logger

log = get_logger(__name__)

# ─── Internal registry ────────────────────────────────────────────────────────

_REGISTRY: Dict[str, Type[DeploymentBackend]] = {}


def register_backend(cls: Type[DeploymentBackend]) -> Type[DeploymentBackend]:
    """Class decorator that registers a backend in the global registry."""
    name = getattr(cls, "name", cls.__name__.lower())
    if name in _REGISTRY:
        log.warning("backend_already_registered", name=name)
    _REGISTRY[name] = cls
    log.debug("backend_registered", name=name)
    return cls


# ─── Public API ───────────────────────────────────────────────────────────────

def discover_backends() -> None:
    """Import all modules in the ``backends`` package so decorators fire.

    This is called once at startup.
    """
    import backends as _pkg

    for _importer, modname, _ispkg in pkgutil.iter_modules(_pkg.__path__):
        if modname in ("base", "registry", "__init__"):
            continue
        importlib.import_module(f"backends.{modname}")
    log.info("backends_discovered", count=len(_REGISTRY), names=list(_REGISTRY.keys()))


def list_backends() -> Dict[str, Dict[str, Any]]:
    """Return ``{name: capabilities_dict}`` for every registered backend."""
    result: Dict[str, Dict[str, Any]] = {}
    for name, cls in _REGISTRY.items():
        # Instantiate with dummy args just to read capabilities
        try:
            caps = cls.capabilities  # class-level attribute
            result[name] = {
                "name": name,
                "description": getattr(cls, "description", ""),
                "supports_gpu": caps.supports_gpu,
                "supports_batching": caps.supports_batching,
                "supports_autoscaling": caps.supports_autoscaling,
                "managed_service": caps.managed_service,
                "supported_frameworks": caps.supported_frameworks,
                "typical_latency_ms": caps.typical_latency_ms,
                "complexity": caps.complexity,
                "cost_tier": caps.cost_tier,
            }
        except Exception as exc:
            log.warning("backend_caps_error", name=name, error=str(exc))
    return result


def get_backend(
    name: str,
    config: DeployConfig,
    model_metadata: ModelMetadata,
) -> DeploymentBackend:
    """Instantiate and return a backend by *name*.

    Raises:
        KeyError: if the name is not registered.
    """
    if name not in _REGISTRY:
        raise KeyError(
            f"Backend '{name}' not found. Available: {list(_REGISTRY.keys())}"
        )
    cls = _REGISTRY[name]
    return cls(config=config, model_metadata=model_metadata)


def registered_names() -> list[str]:
    """Return sorted list of registered backend names."""
    return sorted(_REGISTRY.keys())
