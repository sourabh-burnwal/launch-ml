"""
NVIDIA Triton Inference Server Backend — **Python Backend only**.

Best for:
    - High-throughput GPU inference
    - Multi-model serving
    - Dynamic batching
    - Any Python-based model (PyTorch, TensorFlow, HuggingFace, etc.)
    - Production workloads requiring maximum GPU utilisation
    - **Local deployment** via Docker Compose with a custom Triton image

Architecture:
    The tool reads the user's ``predict.py``, sends it to the configured
    LLM, and the LLM **generates a standalone** ``model.py`` that
    implements Triton's ``TritonPythonModel`` class.  The generated
    ``model.py`` embeds all model-loading and inference logic — the
    user's original ``predict.py`` is **not** copied into the model
    repository.

    Generated model-repository layout::

        model_repository/
          <model_name>/
            config.pbtxt          ← declares Python backend, I/O tensors
            1/
              model.py            ← TritonPythonModel (LLM-generated)

    A custom ``Dockerfile`` is built on top of the official NGC Triton
    image, installing the user's Python dependencies and embedding the
    model repository.

Ports:
    - 8000  HTTP inference  (KServe V2 REST)
    - 8001  gRPC inference  (protobuf)
    - 8002  Prometheus metrics
"""

from __future__ import annotations

import json
import shutil
import subprocess
from pathlib import Path
from typing import Any, Dict, List, Tuple

from launchml.backends.base import BackendCapabilities, DeploymentBackend
from launchml.backends.registry import register_backend
from launchml.utils.logger import get_logger

log = get_logger(__name__)

# ── Triton NGC image tag ──────────────────────────────────────────────────────
_TRITON_IMAGE_TAG = "24.01-py3"
_TRITON_BASE_IMAGE = f"nvcr.io/nvidia/tritonserver:{_TRITON_IMAGE_TAG}"

# ── JSON-schema → Triton type mapping ────────────────────────────────────────
_TRITON_TYPE_MAP: Dict[str, Tuple[str, str]] = {
    # json_schema_type → (config.pbtxt TYPE, V2 HTTP datatype)
    "string":  ("TYPE_STRING",  "BYTES"),
    "number":  ("TYPE_FP32",    "FP32"),
    "integer": ("TYPE_INT32",   "INT32"),
    "boolean": ("TYPE_BOOL",    "BOOL"),
    "array":   ("TYPE_FP32",    "FP32"),
    "object":  ("TYPE_STRING",  "BYTES"),
}

# Default example values per type (for curl / client snippets)
_EXAMPLE_VALUES: Dict[str, Any] = {
    "string":  "example_value",
    "number":  1.0,
    "integer": 1,
    "boolean": True,
    "array":   [1.0, 2.0, 3.0],
    "object":  "{}",
}


@register_backend
class TritonBackend(DeploymentBackend):
    name = "triton"
    description = (
        "NVIDIA Triton Inference Server — high-performance GPU inference "
        "with Python backend. Wraps any predict() function."
    )
    capabilities = BackendCapabilities(
        supports_gpu=True,
        supports_batching=True,
        supports_autoscaling=True,
        supports_multi_model=True,
        supports_streaming=True,
        managed_service=False,
        supports_local_deployment=True,
        max_model_size_gb=50.0,
        supported_frameworks=["pytorch", "tensorflow", "onnx", "transformers"],
        typical_latency_ms=10,
        complexity="high",
        cost_tier="medium",
    )

    # We derive the Triton model name from the config (safe default).
    _model_name = "model"

    # ------------------------------------------------------------------
    # Interface implementation
    # ------------------------------------------------------------------

    def validate(self) -> bool:
        fw = self.model_metadata.detected_framework
        if fw == "unknown" and not self.model_metadata.has_custom_entrypoint:
            log.warning("triton_unknown_framework")
            return False
        return True

    def generate_inference_code(self) -> Dict[str, str]:
        self.ensure_dirs()

        mn = self._model_name

        # ── Use LLM to generate model.py from the user's predict.py ───
        model_py_code = self._generate_model_py_via_llm()

        files: Dict[str, str] = {
            f"model_repository/{mn}/config.pbtxt": self._render_config_pbtxt(),
            f"model_repository/{mn}/1/model.py": model_py_code,
            "Dockerfile": self._render_dockerfile(),
            "requirements.txt": self._render_requirements(),
            "client.py": self._render_client(),
            "curl_example.sh": self._render_curl_example(),
        }

        written = self.write_files(self.serving_dir, files)

        # Copy any local model weight files into the version directory
        # (but NOT predict.py — model.py is self-contained).
        if self.model_metadata.model_files:
            version_dir = self.serving_dir / "model_repository" / mn / "1"
            version_dir.mkdir(parents=True, exist_ok=True)
            model_dir = Path(self.model_metadata.model_path)
            for model_file in self.model_metadata.model_files:
                src = model_dir / model_file
                if src.is_file() and src.name != self._entrypoint_filename():
                    dst = version_dir / model_file
                    dst.parent.mkdir(parents=True, exist_ok=True)
                    shutil.copy2(src, dst)
                    written.append(str(dst.resolve()))

        log.info("triton_code_generated", files=written)
        return files

    def generate_terraform(self) -> Dict[str, str]:
        self.ensure_dirs()
        files = {
            "main.tf": self._render_main_tf(),
            "variables.tf": self._render_variables_tf(),
            "outputs.tf": self._render_outputs_tf(),
        }
        written = self.write_files(self.terraform_dir, files)
        log.info("triton_terraform_generated", files=written)
        return files

    def deploy(self) -> Dict[str, Any]:
        log.info("triton_deploy_start")
        host = f"ml-triton-{self.config.deployment.region}.example.com:8000"
        return {
            "status": "deployed",
            "endpoint_url": f"https://{host}/v2/models/{self._model_name}/infer",
            "grpc_url": f"ml-triton-{self.config.deployment.region}.example.com:8001",
            "metrics_url": f"https://ml-triton-{self.config.deployment.region}.example.com:8002/metrics",
            "backend": self.name,
            "sample_curl": self._build_sample_curl(host),
        }

    def setup_observability(self) -> Dict[str, str]:
        is_local = self.config.deployment.is_local
        instructions: Dict[str, str] = {}
        if self.config.observability.enable_metrics:
            if is_local:
                instructions["metrics"] = (
                    "Triton exposes Prometheus metrics at http://localhost:8002/metrics\n"
                    "Key metrics: nv_inference_request_success, nv_inference_request_failure, "
                    "nv_inference_queue_duration_us, nv_gpu_utilization\n"
                    "You can scrape this with a local Prometheus instance."
                )
            else:
                instructions["metrics"] = (
                    "Triton exposes Prometheus metrics on port 8002.\n"
                    "Scrape target: <host>:8002/metrics\n"
                    "Key metrics: nv_inference_request_success, nv_inference_request_failure, "
                    "nv_inference_queue_duration_us, nv_gpu_utilization"
                )
        if self.config.observability.enable_logging:
            if is_local:
                instructions["logging"] = (
                    "Triton logs stream to stdout (visible via docker compose logs).\n"
                    f"Run: docker compose -f {self.serving_dir}/docker-compose.yaml logs -f\n"
                    "Verbose logging enabled (--log-verbose=1)."
                )
            else:
                instructions["logging"] = (
                    "Triton logs to stdout in structured format.\n"
                    "Set --log-verbose=1 for detailed request logs.\n"
                    "Integrate with Cloud Logging or Fluentd."
                )
        return instructions

    # ------------------------------------------------------------------
    # Local deployment
    # ------------------------------------------------------------------

    def generate_local_compose(self) -> Dict[str, str]:
        """Generate docker-compose.yaml for local Triton serving."""
        self.ensure_dirs()
        files = {
            "docker-compose.yaml": self._render_local_compose(),
        }
        written = self.write_files(self.serving_dir, files)
        log.info("triton_local_compose_generated", files=written)
        return files

    def deploy_local(self) -> Dict[str, Any]:
        """Build the custom Triton image and start the container locally."""
        log.info("triton_local_deploy_start")
        compose_dir = self.serving_dir.resolve()
        compose_file = compose_dir / "docker-compose.yaml"

        if not compose_file.is_file():
            log.error("triton_compose_missing", path=str(compose_file))
            return {"status": "failed", "error": "docker-compose.yaml not found"}

        docker_cmd = self._resolve_docker_compose_cmd()
        if docker_cmd is None:
            log.warning("docker_compose_not_found", msg="Simulating local deployment")
            return self._simulate_local_deploy()

        try:
            # Tear down any previous run
            subprocess.run(
                [*docker_cmd, "down", "--remove-orphans"],
                cwd=compose_dir, capture_output=True, text=True, timeout=60,
            )

            # Build the custom image (NGC base + user deps + model repo)
            log.info("triton_local_build")
            build = subprocess.run(
                [*docker_cmd, "build"],
                cwd=compose_dir, capture_output=True, text=True, timeout=600,
            )
            if build.returncode != 0:
                log.error("triton_local_build_failed", stderr=build.stderr[:500])
                return {"status": "failed", "error": build.stderr[:500]}

            # Start detached
            log.info("triton_local_up")
            up = subprocess.run(
                [*docker_cmd, "up", "-d"],
                cwd=compose_dir, capture_output=True, text=True, timeout=120,
            )
            if up.returncode != 0:
                log.error("triton_local_up_failed", stderr=up.stderr[:500])
                return {"status": "failed", "error": up.stderr[:500]}

            mn = self._model_name
            return {
                "status": "running_locally",
                "endpoint_url": f"http://localhost:8000/v2/models/{mn}/infer",
                "grpc_url": "localhost:8001",
                "monitoring_url": "http://localhost:8002/metrics",
                "backend": self.name,
                "stop_command": f"cd {compose_dir} && {' '.join(docker_cmd)} down",
                "sample_curl": self._build_sample_curl(),
            }
        except subprocess.TimeoutExpired:
            log.error("triton_local_timeout")
            return {"status": "failed", "error": "Docker command timed out"}
        except Exception as exc:
            log.error("triton_local_error", error=str(exc))
            return self._simulate_local_deploy()

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _entrypoint_stem(self) -> str:
        """Return the stem (no extension) of the user's entrypoint file."""
        if self.model_metadata.entrypoint_path:
            return Path(self.model_metadata.entrypoint_path).stem
        return "predict"

    def _entrypoint_filename(self) -> str:
        """Return the filename of the user's entrypoint."""
        if self.model_metadata.entrypoint_path:
            return Path(self.model_metadata.entrypoint_path).name
        return "predict.py"

    def _read_entrypoint_source(self) -> str:
        """Read the user's entrypoint source code."""
        if self.model_metadata.entrypoint_path:
            ep = Path(self.model_metadata.entrypoint_path)
            if ep.is_file():
                return ep.read_text()
        return ""

    def _input_specs(self) -> List[Dict[str, Any]]:
        """Derive Triton input tensor specs from ``api.request_schema``."""
        props = self.config.api.request_schema.properties
        if not props:
            return [{"name": "INPUT", "pbtxt_type": "TYPE_STRING",
                     "http_type": "BYTES", "json_type": "string"}]
        specs = []
        for name, prop in props.items():
            jtype = prop.get("type", "string") if isinstance(prop, dict) else "string"
            pbtxt, http = _TRITON_TYPE_MAP.get(jtype, ("TYPE_STRING", "BYTES"))
            specs.append({"name": name, "pbtxt_type": pbtxt,
                          "http_type": http, "json_type": jtype})
        return specs

    def _output_specs(self) -> List[Dict[str, Any]]:
        """Derive Triton output tensor specs from ``api.response_schema``.

        For Python backend we always emit a single ``result`` tensor of
        ``TYPE_STRING`` that carries the full JSON response.  This keeps
        the wrapper generic — the user's predict() can return anything
        serialisable and we don't have to match heterogeneous output
        shapes.
        """
        return [{"name": "result", "pbtxt_type": "TYPE_STRING",
                 "http_type": "BYTES", "json_type": "string"}]

    # ------------------------------------------------------------------
    # Code generation — model repository
    # ------------------------------------------------------------------

    def _render_config_pbtxt(self) -> str:
        """Render ``config.pbtxt`` for the **Python backend**.

        Key differences from native backends:
        - ``backend: "python"`` instead of ``platform: …``
        - ``max_batch_size: 0`` — batching is handled in model.py
        - Single ``result`` output tensor (JSON string)
        """
        inputs = self._input_specs()
        gpu = self.model_metadata.gpu_recommendation == "true"
        kind = "KIND_GPU" if gpu else "KIND_CPU"
        mn = self._model_name

        def _tensor_block(specs: List[Dict[str, Any]], direction: str) -> str:
            entries = []
            for s in specs:
                entries.append(
                    f'  {{\n    name: "{s["name"]}"\n'
                    f'    data_type: {s["pbtxt_type"]}\n'
                    f'    dims: [1]\n  }}'
                )
            return f"{direction} [\n" + ",\n".join(entries) + "\n]"

        input_block = _tensor_block(inputs, "input")
        output_block = _tensor_block(self._output_specs(), "output")

        return f'''# Triton Python Backend Configuration
# Auto-generated by LaunchML
#
# Input tensors derived from api.request_schema in deploy_config.yaml.
# Output is a single JSON string containing the full prediction result.

name: "{mn}"
backend: "python"

max_batch_size: 0

{input_block}

{output_block}

instance_group [
  {{
    count: 1
    kind: {kind}
  }}
]
'''

    def _generate_model_py_via_llm(self) -> str:
        """Use the configured LLM to generate ``model.py``.

        The LLM reads the user's ``predict.py`` source code, understands
        the model loading / inference logic, and produces a **standalone**
        ``TritonPythonModel`` implementation.  The user's ``predict.py``
        is NOT copied — all logic is embedded in the generated code.
        """
        from launchml.core.config_loader import LLMConfig
        from launchml.core.llm_client import create_llm_or_mock, invoke_llm, extract_code

        entrypoint_source = self._read_entrypoint_source()
        if not entrypoint_source:
            log.warning("triton_no_entrypoint_source", msg="No predict.py found, generating skeleton")

        # Build structured prompt for the LLM
        input_specs = self._input_specs()
        input_desc = "\n".join(
            f'  - name: "{s["name"]}", data_type: {s["pbtxt_type"]} '
            f'(HTTP datatype: {s["http_type"]})'
            for s in input_specs
        )
        output_desc = '  - name: "result", data_type: TYPE_STRING (BYTES), dims: [1] — contains JSON-serialized prediction result'

        framework = self.model_metadata.detected_framework
        deps = ", ".join(self.model_metadata.entrypoint_dependencies) if self.model_metadata.entrypoint_dependencies else "none detected"

        system_prompt = f"""\
You are an expert ML engineer specializing in NVIDIA Triton Inference Server.

Your task: read the user's predict.py and implement a complete, standalone
``model.py`` for Triton's **Python Backend**.

## Triton Python Backend API

The file MUST define a class ``TritonPythonModel`` with these methods:

1. ``initialize(self, args)``
   - Load the model, tokenizer, and any resources needed for inference.
   - ``args`` is a dict with model metadata (you can ignore it).

2. ``execute(self, requests)``
   - ``requests`` is a list of ``pb_utils.InferenceRequest`` objects.
   - For each request, extract input tensors:
       ``inp = pb_utils.get_input_tensor_by_name(request, "<name>")``
       ``data = inp.as_numpy()``
   - For STRING/BYTES tensors: ``data.flatten()[0]`` gives bytes → decode with ``.decode("utf-8")``.
   - Run inference using the loaded model.
   - Serialize the result dict as JSON, wrap in a BYTES tensor:
       ``pb_utils.Tensor("result", np.array([json_str], dtype=np.object_))``
   - Return a list of ``pb_utils.InferenceResponse(output_tensors=[...])``
   - On error: ``pb_utils.InferenceResponse(output_tensors=[], error=pb_utils.TritonError(str(e)))``

3. ``finalize(self)`` — cleanup (can be empty ``pass``).

## Required imports
```python
import triton_python_backend_utils as pb_utils
import numpy as np
import json
```

## Critical rules
- The model.py must be **SELF-CONTAINED**. Do NOT import or reference the user's predict.py.
- Re-implement ALL model loading, preprocessing, and inference logic directly.
- Respond with ONLY the Python code. No markdown fences, no explanations."""

        user_prompt = f"""\
## User's predict.py (understand this, then re-implement as TritonPythonModel)

```python
{entrypoint_source}
```

## Detected framework: {framework}
## Dependencies: {deps}

## Triton config.pbtxt input tensors:
{input_desc}

## Triton config.pbtxt output tensor:
{output_desc}

Generate the complete model.py."""

        # Invoke the LLM
        llm_config = LLMConfig(**self.config.model_dump().get("llm", {}))
        llm = create_llm_or_mock(llm_config)

        log.info("triton_llm_generate_model_py", framework=framework)
        raw_response = invoke_llm(
            llm,
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            expect_json=False,
        )

        # Extract clean Python code from the response
        if isinstance(raw_response, dict):
            # MockLLM might return the code directly as a string via
            # _Resp.content; invoke_llm with expect_json=False returns str,
            # but if it somehow parsed as JSON, grab raw_response
            code = raw_response.get("raw_response", str(raw_response))
        else:
            code = str(raw_response)

        code = extract_code(code)

        # Validate: must contain TritonPythonModel class
        if "TritonPythonModel" not in code:
            log.warning("triton_llm_bad_response", msg="LLM response missing TritonPythonModel, using fallback")
            code = self._fallback_model_py()

        log.info("triton_model_py_generated", lines=code.count("\n") + 1)
        return code

    def _fallback_model_py(self) -> str:
        """Last-resort skeleton if the LLM produces unusable output."""
        inputs = self._input_specs()
        extract_lines = []
        for spec in inputs:
            name = spec["name"]
            if spec["http_type"] == "BYTES":
                extract_lines.append(
                    f'        inp = pb_utils.get_input_tensor_by_name(request, "{name}")\n'
                    f'        raw = inp.as_numpy().flatten()[0]\n'
                    f'        {name} = raw.decode("utf-8") if isinstance(raw, bytes) else str(raw)'
                )
            else:
                extract_lines.append(
                    f'        inp = pb_utils.get_input_tensor_by_name(request, "{name}")\n'
                    f'        {name} = inp.as_numpy().flatten().tolist()'
                )
        extract_code_block = "\n".join(extract_lines)

        return f'''"""
TritonPythonModel — auto-generated by LaunchML (fallback template).

TODO: The LLM was unable to generate model.py from your predict.py.
      Please fill in the model loading and inference logic below.
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
                # ── Extract inputs ────────────────────────────
{extract_code_block}

                # TODO: Run inference
                result = {{"prediction": "not_implemented"}}

                result_json = json.dumps(result, default=str)
                out_tensor = pb_utils.Tensor(
                    "result", np.array([result_json], dtype=np.object_)
                )
                responses.append(
                    pb_utils.InferenceResponse(output_tensors=[out_tensor])
                )
            except Exception as e:
                responses.append(
                    pb_utils.InferenceResponse(
                        output_tensors=[],
                        error=pb_utils.TritonError(str(e)),
                    )
                )
        return responses

    def finalize(self):
        pass
'''

    # ------------------------------------------------------------------
    # Code generation — Docker
    # ------------------------------------------------------------------

    def _render_dockerfile(self) -> str:
        """Render a Dockerfile based on the official NGC Triton image.

        The image includes:
        - NVIDIA Triton Inference Server with Python backend support
        - User's Python dependencies (from requirements.txt)
        - The complete model_repository (config.pbtxt + LLM-generated model.py)
        """
        gpu = self.model_metadata.gpu_recommendation == "true"

        # Triton NGC image already has Python, CUDA, etc.
        # We just need to install the user's extra pip packages.
        return f"""# Auto-generated Dockerfile — LaunchML (Triton Python Backend)
FROM {_TRITON_BASE_IMAGE}

# Install user Python dependencies
COPY requirements.txt /tmp/requirements.txt
RUN pip install --no-cache-dir -r /tmp/requirements.txt && rm /tmp/requirements.txt

# Copy model repository (config.pbtxt + model.py)
COPY model_repository /models

EXPOSE 8000 8001 8002

CMD ["tritonserver", \\
     "--model-repository=/models", \\
     "--strict-model-config=false", \\
     "--log-verbose=1", \\
     "--metrics-port=8002"]
"""

    def _render_requirements(self) -> str:
        """Render requirements.txt with the user's entrypoint dependencies.

        The Triton NGC image already ships numpy, so we skip it.
        We add whatever the model analyzer detected from the user's
        imports (torch, transformers, requests, Pillow, etc.).
        """
        # Packages already in the Triton NGC image
        skip = {"numpy", "triton", "tritonclient"}
        base: list[str] = []

        ep_deps = self.model_metadata.entrypoint_dependencies or []
        seen: set[str] = set()
        for dep in ep_deps:
            dep_lower = dep.lower()
            if dep_lower not in skip and dep_lower not in seen:
                base.append(dep)
                seen.add(dep_lower)

        return "\n".join(base) + "\n" if base else "# No extra dependencies\n"

    def _render_local_compose(self) -> str:
        """Render docker-compose.yaml that builds from our Dockerfile."""
        gpu = self.model_metadata.gpu_recommendation == "true"
        gpu_block = ""
        if gpu:
            gpu_block = (
                "    deploy:\n"
                "      resources:\n"
                "        reservations:\n"
                "          devices:\n"
                "            - driver: nvidia\n"
                "              count: 1\n"
                "              capabilities: [gpu]\n"
            )

        return f"""# Local Triton Deployment — auto-generated by LaunchML
# Usage:
#   docker compose up --build        (foreground)
#   docker compose up --build -d     (detached)
#   docker compose down              (stop)

services:
  triton:
    build:
      context: .
      dockerfile: Dockerfile
    container_name: launchml-triton
    ports:
      - "8000:8000"   # HTTP inference  (KServe V2 REST)
      - "8001:8001"   # gRPC inference
      - "8002:8002"   # Prometheus metrics
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:8000/v2/health/ready"]
      interval: 10s
      timeout: 5s
      retries: 10
      start_period: 60s
    restart: unless-stopped
{gpu_block}"""

    # ------------------------------------------------------------------
    # Code generation — docker-compose for cloud (non-local) deploys
    # ------------------------------------------------------------------

    def _render_docker_compose(self) -> str:
        """Cloud-oriented compose (for reference / staging)."""
        gpu = self.model_metadata.gpu_recommendation == "true"
        gpu_block = ""
        if gpu:
            gpu_block = (
                "    deploy:\n"
                "      resources:\n"
                "        reservations:\n"
                "          devices:\n"
                "            - driver: nvidia\n"
                "              count: 1\n"
                "              capabilities: [gpu]\n"
            )

        return f"""# Triton Inference Server — Docker Compose (cloud/staging)
# Auto-generated by LaunchML

services:
  triton:
    build:
      context: .
      dockerfile: Dockerfile
    ports:
      - "8000:8000"
      - "8001:8001"
      - "8002:8002"
{gpu_block}"""

    # ------------------------------------------------------------------
    # curl / client examples
    # ------------------------------------------------------------------

    def _build_sample_curl(self, host: str = "localhost:8000") -> str:
        """Build a curl command for the Triton V2 HTTP API."""
        inputs = self._input_specs()
        mn = self._model_name
        input_entries: list[str] = []
        for spec in inputs:
            name = spec["name"]
            http_type = spec["http_type"]
            jtype = spec["json_type"]

            name_lower = name.lower()
            if "url" in name_lower or "image" in name_lower:
                example = "https://example.com/image.jpg"
            elif "text" in name_lower:
                example = "Hello, world!"
            else:
                example = _EXAMPLE_VALUES.get(jtype, "example")

            if http_type == "BYTES":
                data_str = json.dumps([example])
                shape_str = "[1]"
            elif jtype == "array":
                data_str = "[[1.0, 2.0, 3.0]]"
                shape_str = "[1, 3]"
            else:
                data_str = json.dumps([example])
                shape_str = "[1]"

            entry = (
                f'    {{"name": "{name}", "shape": {shape_str}, '
                f'"datatype": "{http_type}", "data": {data_str}}}'
            )
            input_entries.append(entry)

        inputs_json = ",\n".join(input_entries)

        return (
            f'curl -X POST http://{host}/v2/models/{mn}/infer \\\n'
            f'  -H "Content-Type: application/json" \\\n'
            f"  -d '{{\n"
            f'  \"inputs\": [\n{inputs_json}\n  ]\n'
            f"}}'"
        )

    def _render_curl_example(self) -> str:
        """Render a shell script with sample curl commands."""
        curl_cmd = self._build_sample_curl()
        mn = self._model_name
        return f'''#!/usr/bin/env bash
# ============================================================================
# Sample curl commands for Triton V2 HTTP API — auto-generated by LaunchML
# ============================================================================
#
# Triton exposes three ports:
#   8000 — HTTP inference  (KServe V2 REST protocol, used by curl below)
#   8001 — gRPC inference  (protobuf, use tritonclient for this)
#   8002 — Prometheus metrics
#
# NOTE: Triton uses "BYTES" as the datatype for strings in the V2 HTTP
# protocol, even though config.pbtxt uses TYPE_STRING.
# ============================================================================

# ── Health check ─────────────────────────────────────────────────────────────
curl -s http://localhost:8000/v2/health/ready
echo

# ── Model metadata ───────────────────────────────────────────────────────────
curl -s http://localhost:8000/v2/models/{mn} | python3 -m json.tool

# ── Inference request ────────────────────────────────────────────────────────
{curl_cmd}
'''

    def _render_client(self) -> str:
        """Render a Python client using tritonclient HTTP."""
        inputs = self._input_specs()
        mn = self._model_name

        input_setup: list[str] = []
        input_names: list[str] = []
        for i, spec in enumerate(inputs):
            name = spec["name"]
            http_type = spec["http_type"]
            input_names.append(name)

            if http_type == "BYTES":
                input_setup.append(
                    f'    inp_{i} = httpclient.InferInput("{name}", [1], "{http_type}")\n'
                    f'    inp_{i}.set_data_from_numpy(np.array([input_data["{name}"].encode("utf-8")], dtype=np.object_))'
                )
            else:
                np_dtype = {"FP32": "np.float32", "INT32": "np.int32",
                            "BOOL": "np.bool_"}.get(http_type, "np.float32")
                input_setup.append(
                    f'    inp_{i} = httpclient.InferInput("{name}", [1], "{http_type}")\n'
                    f'    inp_{i}.set_data_from_numpy(np.array([input_data["{name}"]], dtype={np_dtype}))'
                )

        inputs_var = ", ".join(f"inp_{i}" for i in range(len(inputs)))

        example_input: dict[str, Any] = {}
        for spec in inputs:
            name_lower = spec["name"].lower()
            jtype = spec["json_type"]
            if "url" in name_lower or "image" in name_lower:
                example_input[spec["name"]] = "https://example.com/image.jpg"
            elif "text" in name_lower:
                example_input[spec["name"]] = "Hello, world!"
            else:
                example_input[spec["name"]] = _EXAMPLE_VALUES.get(jtype, "example")
        example_json = json.dumps(example_input, indent=8)

        return f'''"""
Sample Triton client — auto-generated by LaunchML.

Uses the tritonclient HTTP library to communicate with Triton.
Input/output names and types are derived from deploy_config.yaml.

Install: pip install tritonclient[http] numpy
"""
import json
import numpy as np

try:
    import tritonclient.http as httpclient
except ImportError:
    print("Install: pip install tritonclient[http]")
    raise


def predict(input_data: dict, url: str = "localhost:8000") -> dict:
    """Send an inference request to Triton.

    Args:
        input_data: Dict mapping input names to values.
                    Expected keys: {input_names}
        url:        Triton HTTP endpoint (host:port).

    Returns:
        Parsed JSON prediction result from the model.
    """
    client = httpclient.InferenceServerClient(url=url)

    # ── Inputs ────────────────────────────────────────────────────────────
{chr(10).join(input_setup)}

    # ── Output ────────────────────────────────────────────────────────────
    outputs = [httpclient.InferRequestedOutput("result")]

    result = client.infer("{mn}", inputs=[{inputs_var}], outputs=outputs)

    # The "result" tensor is a JSON string — parse it
    raw = result.as_numpy("result")[0]
    decoded = raw.decode("utf-8") if isinstance(raw, bytes) else str(raw)
    return json.loads(decoded)


if __name__ == "__main__":
    sample = {example_json}
    print("Prediction:", predict(sample))
'''

    # ------------------------------------------------------------------
    # Terraform
    # ------------------------------------------------------------------

    def _render_main_tf(self) -> str:
        region = self.config.deployment.region
        return f'''# Auto-generated Terraform — LaunchML (Triton backend)
terraform {{
  required_version = ">= 1.5.0"
  required_providers {{
    google = {{
      source  = "hashicorp/google"
      version = "~> 5.0"
    }}
  }}
}}

provider "google" {{
  project = var.project_id
  region  = var.region
}}

# ─── GKE Cluster for Triton ──────────────────────────────────────────────
resource "google_container_cluster" "triton_cluster" {{
  name     = "ml-triton-cluster"
  location = var.region

  initial_node_count = 1
  deletion_protection = false

  node_config {{
    machine_type = var.machine_type
    oauth_scopes = ["https://www.googleapis.com/auth/cloud-platform"]

    guest_accelerator {{
      type  = var.gpu_type
      count = var.gpu_count
    }}
  }}
}}

# ─── GPU Node Pool ────────────────────────────────────────────────────────
resource "google_container_node_pool" "triton_pool" {{
  name       = "ml-triton-gpu-pool"
  location   = var.region
  cluster    = google_container_cluster.triton_cluster.name

  initial_node_count = var.min_nodes

  node_config {{
    machine_type = var.machine_type
    guest_accelerator {{
      type  = var.gpu_type
      count = var.gpu_count
    }}
  }}

  autoscaling {{
    min_node_count = var.min_nodes
    max_node_count = var.max_nodes
  }}
}}

# ─── Load Balancer ────────────────────────────────────────────────────────
resource "google_compute_global_address" "triton_ip" {{
  name = "ml-triton-ip"
}}
'''

    def _render_variables_tf(self) -> str:
        return f'''variable "project_id" {{
  type = string
}}

variable "region" {{
  type    = string
  default = "{self.config.deployment.region}"
}}

variable "machine_type" {{
  type    = string
  default = "n1-standard-8"
}}

variable "gpu_type" {{
  type    = string
  default = "nvidia-tesla-t4"
}}

variable "gpu_count" {{
  type    = number
  default = 1
}}

variable "min_nodes" {{
  type    = number
  default = 1
}}

variable "max_nodes" {{
  type    = number
  default = 3
}}
'''

    def _render_outputs_tf(self) -> str:
        return '''output "endpoint_url" {
  description = "Triton HTTP endpoint"
  value       = "http://${google_compute_global_address.triton_ip.address}:8000"
}

output "grpc_url" {
  description = "Triton gRPC endpoint"
  value       = "${google_compute_global_address.triton_ip.address}:8001"
}

output "monitoring_url" {
  description = "Triton metrics endpoint"
  value       = "http://${google_compute_global_address.triton_ip.address}:8002/metrics"
}
'''

    # ------------------------------------------------------------------
    # Local deployment helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _resolve_docker_compose_cmd() -> list[str] | None:
        """Return the docker compose CLI command, or None if unavailable."""
        if shutil.which("docker"):
            result = subprocess.run(
                ["docker", "compose", "version"],
                capture_output=True, text=True, timeout=10,
            )
            if result.returncode == 0:
                return ["docker", "compose"]
        if shutil.which("docker-compose"):
            return ["docker-compose"]
        return None

    def _simulate_local_deploy(self) -> Dict[str, Any]:
        """Return a simulated result when Docker is not available."""
        mn = self._model_name
        return {
            "status": "simulated_local",
            "endpoint_url": f"http://localhost:8000/v2/models/{mn}/infer",
            "grpc_url": "localhost:8001",
            "monitoring_url": "http://localhost:8002/metrics",
            "backend": "triton",
            "note": "Docker not available — simulated. Install Docker to deploy locally.",
            "sample_curl": self._build_sample_curl(),
        }
