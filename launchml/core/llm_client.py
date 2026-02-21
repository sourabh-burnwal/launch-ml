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

from launchml.core.config_loader import LLMConfig
from launchml.utils.logger import get_logger

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


def extract_code(text: str) -> str:
    """Extract Python code from an LLM response that may include markdown fences."""
    cleaned = text.strip()

    # Strip ```python ... ``` or ``` ... ``` fences
    if "```" in cleaned:
        lines = cleaned.split("\n")
        code_lines: list[str] = []
        inside_fence = False
        for line in lines:
            stripped = line.strip()
            if stripped.startswith("```"):
                inside_fence = not inside_fence
                continue
            if inside_fence:
                code_lines.append(line)
        if code_lines:
            return "\n".join(code_lines).strip()

    return cleaned


# ─── Fallback for when no API key is available (POC / demo mode) ──────────────

class MockLLM:
    """A mock LLM for demo/testing when no API key is set.

    Returns deterministic JSON responses so the pipeline can be
    exercised end-to-end without incurring API costs.

    Also handles **code generation** prompts (e.g. Triton model.py) by
    returning a reasonable template derived from keywords in the prompt.
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

        # ── Code generation: Triton model.py ──────────────────────────
        if "TritonPythonModel" in user_msg:
            code = self._mock_triton_model_py(user_msg)

            class _CodeResp:
                content = code

            return _CodeResp()

        # ── Strategy decision ─────────────────────────────────────────
        is_local = "LOCAL (Docker" in user_msg or "Cloud: local" in user_msg

        if is_local:
            response_json = {
                "selected_backend": "fastapi",
                "reasoning": (
                    "For local development, FastAPI is the ideal choice. It provides "
                    "a lightweight, fast-starting container with minimal resource "
                    "requirements. The model can be mounted as a volume for rapid "
                    "iteration without rebuilding the image. Prometheus metrics are "
                    "available at /metrics for local observability."
                ),
                "instance_type": "local",
                "gpu_type": "none",
                "scaling_strategy": "single_container",
                "replicas_min": 1,
                "replicas_max": 1,
            }
        else:
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

    # ── Mock code generation ──────────────────────────────────────────

    @staticmethod
    def _mock_triton_model_py(prompt: str) -> str:
        """Generate a mock TritonPythonModel from prompt keywords.

        This is used when no real LLM API key is available.  It inspects
        the prompt for framework/import clues and produces a plausible
        (but possibly imperfect) implementation.
        """
        # Detect framework from the prompt (which contains predict.py code)
        has_torch = "import torch" in prompt or "from torch" in prompt
        has_transformers = "transformers" in prompt
        has_tensorflow = "tensorflow" in prompt or "import tf" in prompt

        if has_transformers:
            return MockLLM._mock_transformers_triton()
        elif has_torch:
            return MockLLM._mock_pytorch_triton()
        elif has_tensorflow:
            return MockLLM._mock_tensorflow_triton()
        else:
            return MockLLM._mock_generic_triton()

    @staticmethod
    def _mock_transformers_triton() -> str:
        return '''"""
TritonPythonModel — auto-generated by LaunchML (mock LLM).

Implements a HuggingFace Transformers model for Triton Python Backend.
"""

import json
import os

import numpy as np
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForSequenceClassification

import triton_python_backend_utils as pb_utils


class TritonPythonModel:

    def initialize(self, args):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model_name = os.environ.get(
            "MODEL_NAME", "distilbert-base-uncased-finetuned-sst-2-english"
        )
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_name)
        self.model.to(self.device)
        self.model.eval()

    def execute(self, requests):
        responses = []
        for request in requests:
            try:
                inp = pb_utils.get_input_tensor_by_name(request, "text")
                raw = inp.as_numpy().flatten()[0]
                text = raw.decode("utf-8") if isinstance(raw, bytes) else str(raw)

                inputs = self.tokenizer(
                    text, return_tensors="pt", truncation=True, padding=True
                ).to(self.device)

                with torch.no_grad():
                    outputs = self.model(**inputs)
                    probs = F.softmax(outputs.logits, dim=1)

                confidence, predicted_class = torch.max(probs, dim=1)
                label_map = self.model.config.id2label

                result = {
                    "text": text,
                    "label": label_map[predicted_class.item()],
                    "confidence": round(confidence.item(), 4),
                }
                result_json = json.dumps(result)
                out_tensor = pb_utils.Tensor(
                    "result", np.array([result_json], dtype=np.object_)
                )
                responses.append(pb_utils.InferenceResponse(output_tensors=[out_tensor]))
            except Exception as e:
                responses.append(
                    pb_utils.InferenceResponse(
                        output_tensors=[], error=pb_utils.TritonError(str(e))
                    )
                )
        return responses

    def finalize(self):
        pass
'''

    @staticmethod
    def _mock_pytorch_triton() -> str:
        return '''"""
TritonPythonModel — auto-generated by LaunchML (mock LLM).

Implements a PyTorch model for Triton Python Backend.
"""

import json

import numpy as np
import torch

import triton_python_backend_utils as pb_utils


class TritonPythonModel:

    def initialize(self, args):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        # TODO: Load your model here
        self.model = None

    def execute(self, requests):
        responses = []
        for request in requests:
            try:
                body = {}
                for inp in request.inputs():
                    name = inp.name()
                    data = inp.as_numpy()
                    if data.dtype == np.object_ or data.dtype.kind in ("U", "S"):
                        val = data.flatten()[0]
                        body[name] = val.decode("utf-8") if isinstance(val, bytes) else str(val)
                    else:
                        body[name] = data.flatten().tolist()

                # TODO: Run inference
                result = {"prediction": "sample", "input": body}
                result_json = json.dumps(result, default=str)
                out_tensor = pb_utils.Tensor(
                    "result", np.array([result_json], dtype=np.object_)
                )
                responses.append(pb_utils.InferenceResponse(output_tensors=[out_tensor]))
            except Exception as e:
                responses.append(
                    pb_utils.InferenceResponse(
                        output_tensors=[], error=pb_utils.TritonError(str(e))
                    )
                )
        return responses

    def finalize(self):
        pass
'''

    @staticmethod
    def _mock_tensorflow_triton() -> str:
        return '''"""
TritonPythonModel — auto-generated by LaunchML (mock LLM).

Implements a TensorFlow model for Triton Python Backend.
"""

import json

import numpy as np
import tensorflow as tf

import triton_python_backend_utils as pb_utils


class TritonPythonModel:

    def initialize(self, args):
        # TODO: Load your TensorFlow/Keras model here
        self.model = None

    def execute(self, requests):
        responses = []
        for request in requests:
            try:
                body = {}
                for inp in request.inputs():
                    name = inp.name()
                    data = inp.as_numpy()
                    if data.dtype == np.object_ or data.dtype.kind in ("U", "S"):
                        val = data.flatten()[0]
                        body[name] = val.decode("utf-8") if isinstance(val, bytes) else str(val)
                    else:
                        body[name] = data.flatten().tolist()

                # TODO: Run inference
                result = {"prediction": "sample", "input": body}
                result_json = json.dumps(result, default=str)
                out_tensor = pb_utils.Tensor(
                    "result", np.array([result_json], dtype=np.object_)
                )
                responses.append(pb_utils.InferenceResponse(output_tensors=[out_tensor]))
            except Exception as e:
                responses.append(
                    pb_utils.InferenceResponse(
                        output_tensors=[], error=pb_utils.TritonError(str(e))
                    )
                )
        return responses

    def finalize(self):
        pass
'''

    @staticmethod
    def _mock_generic_triton() -> str:
        return '''"""
TritonPythonModel — auto-generated by LaunchML (mock LLM).

Generic Python Backend implementation for Triton.
"""

import json

import numpy as np

import triton_python_backend_utils as pb_utils


class TritonPythonModel:

    def initialize(self, args):
        # TODO: Load your model here
        pass

    def execute(self, requests):
        responses = []
        for request in requests:
            try:
                body = {}
                for inp in request.inputs():
                    name = inp.name()
                    data = inp.as_numpy()
                    if data.dtype == np.object_ or data.dtype.kind in ("U", "S"):
                        val = data.flatten()[0]
                        body[name] = val.decode("utf-8") if isinstance(val, bytes) else str(val)
                    else:
                        body[name] = data.flatten().tolist()

                # TODO: Run inference
                result = {"prediction": "sample", "input": body}
                result_json = json.dumps(result, default=str)
                out_tensor = pb_utils.Tensor(
                    "result", np.array([result_json], dtype=np.object_)
                )
                responses.append(pb_utils.InferenceResponse(output_tensors=[out_tensor]))
            except Exception as e:
                responses.append(
                    pb_utils.InferenceResponse(
                        output_tensors=[], error=pb_utils.TritonError(str(e))
                    )
                )
        return responses

    def finalize(self):
        pass
'''


def create_llm_or_mock(config: LLMConfig) -> BaseChatModel | MockLLM:
    """Try to create a real LLM; fall back to mock for demo mode."""
    try:
        return create_llm(config)
    except ValueError as exc:
        log.warning("llm_fallback_mock", reason=str(exc))
        return MockLLM(model=config.model)
