"""
LaunchML LLM Client — unified interface to OpenAI / Anthropic / Gemini.

Architecture:
    - Factory function ``create_llm()`` returns a LangChain ``BaseChatModel``
      configured according to the user's YAML config.
    - API keys are **always** read from environment variables; nothing is
      written to disk.
    - A lightweight ``invoke_llm()`` helper handles structured output
      parsing so the agent graph nodes can call it with a prompt and get
      back a Python dict.
"""

from __future__ import annotations

import json
import os
from typing import Any, Dict, Optional

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import HumanMessage, SystemMessage

from core.config_loader import LLMConfig
from utils.logger import get_logger

log = get_logger(__name__)


# ─── Factory ──────────────────────────────────────────────────────────────────

def create_llm(config: LLMConfig) -> BaseChatModel:
    """Instantiate the correct LangChain chat model for *config*.

    Raises:
        ValueError: if the provider is unknown or the API key is missing.
    """
    provider = config.provider.lower()
    model_name = config.model

    if provider == "openai":
        api_key = os.environ.get("OPENAI_API_KEY", "")
        if not api_key:
            raise ValueError("OPENAI_API_KEY environment variable is required.")
        from langchain_openai import ChatOpenAI
        return ChatOpenAI(model=model_name, api_key=api_key, temperature=0.0)

    elif provider == "anthropic":
        api_key = os.environ.get("ANTHROPIC_API_KEY", "")
        if not api_key:
            raise ValueError("ANTHROPIC_API_KEY environment variable is required.")
        from langchain_anthropic import ChatAnthropic
        return ChatAnthropic(model=model_name, api_key=api_key, temperature=0.0)

    elif provider == "gemini":
        api_key = os.environ.get("GOOGLE_API_KEY", "")
        if not api_key:
            raise ValueError("GOOGLE_API_KEY environment variable is required.")
        from langchain_google_genai import ChatGoogleGenerativeAI
        return ChatGoogleGenerativeAI(model=model_name, google_api_key=api_key, temperature=0.0)

    else:
        raise ValueError(f"Unsupported LLM provider: {provider}")


# ─── Helper: structured invoke ────────────────────────────────────────────────

def invoke_llm(
    llm: BaseChatModel,
    *,
    system_prompt: str,
    user_prompt: str,
    expect_json: bool = True,
) -> Dict[str, Any] | str:
    """Send a prompt to the LLM and return the response.

    Args:
        llm: A LangChain chat model.
        system_prompt: System instruction.
        user_prompt: User query.
        expect_json: If True, attempt to parse the response as JSON.

    Returns:
        Parsed dict if ``expect_json`` else raw string.
    """
    messages = [
        SystemMessage(content=system_prompt),
        HumanMessage(content=user_prompt),
    ]

    log.info("llm_invoke", model=getattr(llm, "model_name", str(llm)), expect_json=expect_json)

    response = llm.invoke(messages)
    content: str = response.content  # type: ignore[attr-defined]

    if not expect_json:
        return content

    # Attempt JSON extraction (the LLM sometimes wraps in ```json blocks)
    return _extract_json(content)


def _extract_json(text: str) -> Dict[str, Any]:
    """Best-effort JSON extraction from LLM output."""
    # Strip markdown fences
    cleaned = text.strip()
    if cleaned.startswith("```"):
        lines = cleaned.split("\n")
        # Drop first and last fence lines
        lines = [l for l in lines if not l.strip().startswith("```")]
        cleaned = "\n".join(lines).strip()

    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        # Try to find JSON object in the text
        start = cleaned.find("{")
        end = cleaned.rfind("}") + 1
        if start != -1 and end > start:
            try:
                return json.loads(cleaned[start:end])
            except json.JSONDecodeError:
                pass

    log.warning("json_parse_failed", text=text[:200])
    return {"raw_response": text}


# ─── Fallback for when no API key is available (POC / demo mode) ──────────────

class MockLLM:
    """A mock LLM for demo/testing when no API key is set.

    Returns deterministic JSON responses so the pipeline can be
    exercised end-to-end without incurring API costs.
    """

    def __init__(self, model: str = "mock") -> None:
        self.model_name = model

    def invoke(self, messages: Any) -> Any:  # noqa: ANN401
        """Return a mock response object."""
        # Inspect the user message to decide what to return
        user_msg = ""
        for m in messages:
            if hasattr(m, "content"):
                user_msg += m.content

        log.info("mock_llm_invoke", prompt_length=len(user_msg))

        response_json = {
            "selected_backend": "fastapi",
            "reasoning": (
                "For a small-to-medium model with moderate throughput requirements, "
                "FastAPI provides the best balance of simplicity, flexibility, and "
                "deployment speed. GPU is not strictly required for this model size, "
                "and FastAPI allows easy horizontal scaling behind a load balancer."
            ),
            "instance_type": "n1-standard-4",
            "gpu_type": "none",
            "scaling_strategy": "horizontal_autoscaling",
            "replicas_min": 1,
            "replicas_max": 5,
        }

        class _Resp:
            content = json.dumps(response_json)

        return _Resp()


def create_llm_or_mock(config: LLMConfig) -> BaseChatModel | MockLLM:
    """Try to create a real LLM; fall back to mock for demo mode."""
    try:
        return create_llm(config)
    except ValueError as exc:
        log.warning("llm_fallback_mock", reason=str(exc))
        return MockLLM(model=config.model)
