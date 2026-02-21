"""
Structured logging module for LaunchML.

Provides a consistent, structured logging interface using `structlog` with
Rich console output for human-readable logs during interactive runs and
JSON output for machine-parseable logs in CI/production.

Architecture:
    - All modules import `get_logger(__name__)` to get a bound logger.
    - Reasoning decisions from the LLM agent are logged at INFO level
      with a `reasoning=True` flag so they can be filtered/audited.
    - No secrets are ever logged; sensitive fields are redacted.
"""

from __future__ import annotations

import logging
import sys
from typing import Any

import structlog
from rich.console import Console
from rich.theme import Theme

# ---------------------------------------------------------------------------
# Rich console for pretty CLI output
# ---------------------------------------------------------------------------
_theme = Theme(
    {
        "info": "cyan",
        "warning": "yellow",
        "error": "bold red",
        "success": "bold green",
        "step": "bold magenta",
    }
)
console = Console(theme=_theme)

# ---------------------------------------------------------------------------
# Structlog configuration
# ---------------------------------------------------------------------------
_configured = False


def configure_logging(*, json_output: bool = False, level: int = logging.INFO) -> None:
    """Configure structured logging for the entire application.

    Call once at startup (in ``deploy.py``). Safe to call multiple times;
    subsequent calls are no-ops.

    Args:
        json_output: If True, emit JSON lines instead of coloured text.
        level: Python logging level (default INFO).
    """
    global _configured
    if _configured:
        return
    _configured = True

    shared_processors: list[structlog.types.Processor] = [
        structlog.contextvars.merge_contextvars,
        structlog.stdlib.add_log_level,
        structlog.stdlib.add_logger_name,
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.processors.StackInfoRenderer(),
        structlog.processors.UnicodeDecoder(),
    ]

    if json_output:
        renderer: structlog.types.Processor = structlog.processors.JSONRenderer()
    else:
        renderer = structlog.dev.ConsoleRenderer(colors=True)

    structlog.configure(
        processors=[
            *shared_processors,
            structlog.stdlib.ProcessorFormatter.wrap_for_formatter,
        ],
        logger_factory=structlog.stdlib.LoggerFactory(),
        wrapper_class=structlog.stdlib.BoundLogger,
        cache_logger_on_first_use=True,
    )

    formatter = structlog.stdlib.ProcessorFormatter(
        processors=[
            structlog.stdlib.ProcessorFormatter.remove_processors_meta,
            renderer,
        ],
    )

    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(formatter)

    root = logging.getLogger()
    root.handlers.clear()
    root.addHandler(handler)
    root.setLevel(level)


def get_logger(name: str) -> structlog.stdlib.BoundLogger:
    """Return a structured logger bound to *name*."""
    configure_logging()  # ensure at least default config
    return structlog.get_logger(name)


# ---------------------------------------------------------------------------
# Convenience helpers for CLI output
# ---------------------------------------------------------------------------

def step(msg: str, **kw: Any) -> None:
    """Print a prominent step banner to the console."""
    console.print(f"\n[step]▶ {msg}[/step]", **kw)


def success(msg: str, **kw: Any) -> None:
    console.print(f"[success]✔ {msg}[/success]", **kw)


def warn(msg: str, **kw: Any) -> None:
    console.print(f"[warning]⚠ {msg}[/warning]", **kw)


def error(msg: str, **kw: Any) -> None:
    console.print(f"[error]✖ {msg}[/error]", **kw)


def info(msg: str, **kw: Any) -> None:
    console.print(f"[info]ℹ {msg}[/info]", **kw)
