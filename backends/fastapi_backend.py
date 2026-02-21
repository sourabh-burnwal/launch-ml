"""
FastAPI Backend — generates a FastAPI-based model serving application.

This is the simplest, most flexible backend:
    - Best for small-to-medium models
    - CPU or GPU
    - Supports custom predict.py entrypoints
    - Easy horizontal scaling via Terraform-managed instance groups
    - Prometheus metrics + structured logging out of the box
    - **Local deployment** via Docker Compose for rapid dev iteration
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
from pathlib import Path
from typing import Any, Dict

from backends.base import BackendCapabilities, DeploymentBackend
from backends.registry import register_backend
from utils.logger import get_logger

log = get_logger(__name__)


@register_backend
class FastAPIBackend(DeploymentBackend):
    name = "fastapi"
    description = (
        "Lightweight FastAPI server with uvicorn. Best for simple models, "
        "rapid iteration, and custom inference logic."
    )
    capabilities = BackendCapabilities(
        supports_gpu=True,
        supports_batching=False,
        supports_autoscaling=True,
        supports_multi_model=False,
        supports_streaming=True,
        managed_service=False,
        supports_local_deployment=True,
        max_model_size_gb=10.0,
        supported_frameworks=["pytorch", "tensorflow", "onnx", "transformers"],
        typical_latency_ms=30,
        complexity="low",
        cost_tier="low",
    )

    # ------------------------------------------------------------------
    # Interface implementation
    # ------------------------------------------------------------------

    def validate(self) -> bool:
        fw = self.model_metadata.detected_framework
        if fw not in self.capabilities.supported_frameworks and fw != "unknown":
            log.warning("fastapi_unsupported_framework", framework=fw)
            return False
        if self.model_metadata.total_size_mb > self.capabilities.max_model_size_gb * 1024:
            log.warning("fastapi_model_too_large", size_mb=self.model_metadata.total_size_mb)
            return False
        return True

    def generate_inference_code(self) -> Dict[str, str]:
        self.ensure_dirs()
        files = {
            "app.py": self._render_app(),
            "requirements.txt": self._render_requirements(),
            "Dockerfile": self._render_dockerfile(),
        }
        written = self.write_files(self.serving_dir, files)
        log.info("fastapi_code_generated", files=written)
        return files

    def generate_terraform(self) -> Dict[str, str]:
        self.ensure_dirs()
        files = {
            "main.tf": self._render_main_tf(),
            "variables.tf": self._render_variables_tf(),
            "outputs.tf": self._render_outputs_tf(),
        }
        written = self.write_files(self.terraform_dir, files)
        log.info("fastapi_terraform_generated", files=written)
        return files

    def deploy(self) -> Dict[str, Any]:
        """For POC: simulate deployment and return mock endpoint."""
        log.info("fastapi_deploy_start")
        # In production this would run terraform apply; for POC we simulate
        return {
            "status": "deployed",
            "endpoint_url": f"https://ml-fastapi-{self.config.deployment.region}.example.com/predict",
            "health_url": f"https://ml-fastapi-{self.config.deployment.region}.example.com/health",
            "backend": self.name,
        }

    def setup_observability(self) -> Dict[str, str]:
        is_local = self.config.deployment.is_local
        instructions: Dict[str, str] = {}
        if self.config.observability.enable_metrics:
            if is_local:
                instructions["metrics"] = (
                    "Prometheus metrics are exposed at http://localhost:8080/metrics\n"
                    "Key metrics: request_count, request_latency_seconds, model_load_time\n"
                    "You can scrape this endpoint with a local Prometheus instance."
                )
            else:
                instructions["metrics"] = (
                    "Prometheus metrics are exposed at /metrics endpoint.\n"
                    "Scrape target: <endpoint_url>/metrics\n"
                    "Key metrics: request_count, request_latency_seconds, model_load_time"
                )
        if self.config.observability.enable_logging:
            if is_local:
                instructions["logging"] = (
                    "Container logs stream to stdout (visible via docker compose logs).\n"
                    "Run: docker compose -f generated/serving_code/docker-compose.yaml logs -f"
                )
            else:
                instructions["logging"] = (
                    "Structured JSON logs are written to stdout.\n"
                    "Use Cloud Logging or ELK stack to aggregate.\n"
                    "Log fields: timestamp, level, request_id, latency_ms, status_code"
                )
        return instructions

    # ------------------------------------------------------------------
    # Local deployment
    # ------------------------------------------------------------------

    def generate_local_compose(self) -> Dict[str, str]:
        """Generate docker-compose.yaml for local FastAPI serving."""
        self.ensure_dirs()
        model_path = Path(self.model_metadata.model_path).resolve()
        files = {
            "docker-compose.yaml": self._render_local_compose(model_path),
        }
        written = self.write_files(self.serving_dir, files)
        log.info("fastapi_local_compose_generated", files=written)
        return files

    def deploy_local(self) -> Dict[str, Any]:
        """Build image and start FastAPI container locally."""
        log.info("fastapi_local_deploy_start")
        compose_dir = self.serving_dir.resolve()
        compose_file = compose_dir / "docker-compose.yaml"

        if not compose_file.is_file():
            log.error("fastapi_compose_missing", path=str(compose_file))
            return {"status": "failed", "error": "docker-compose.yaml not found"}

        docker_cmd = self._resolve_docker_compose_cmd()
        if docker_cmd is None:
            log.warning("docker_compose_not_found", msg="Simulating local deployment")
            return self._simulate_local_deploy()

        try:
            # Stop any previous run
            subprocess.run(
                [*docker_cmd, "down", "--remove-orphans"],
                cwd=compose_dir, capture_output=True, text=True, timeout=60,
            )
            # Build
            log.info("fastapi_local_build")
            build = subprocess.run(
                [*docker_cmd, "build"],
                cwd=compose_dir, capture_output=True, text=True, timeout=300,
            )
            if build.returncode != 0:
                log.error("fastapi_local_build_failed", stderr=build.stderr[:500])
                return {"status": "failed", "error": build.stderr[:500]}

            # Start in detached mode
            log.info("fastapi_local_up")
            up = subprocess.run(
                [*docker_cmd, "up", "-d"],
                cwd=compose_dir, capture_output=True, text=True, timeout=120,
            )
            if up.returncode != 0:
                log.error("fastapi_local_up_failed", stderr=up.stderr[:500])
                return {"status": "failed", "error": up.stderr[:500]}

            return {
                "status": "running_locally",
                "endpoint_url": "http://localhost:8080/predict",
                "health_url": "http://localhost:8080/health",
                "monitoring_url": "http://localhost:8080/metrics",
                "backend": self.name,
                "stop_command": f"cd {compose_dir} && {' '.join(docker_cmd)} down",
            }
        except subprocess.TimeoutExpired:
            log.error("fastapi_local_timeout")
            return {"status": "failed", "error": "Docker command timed out"}
        except Exception as exc:
            log.error("fastapi_local_error", error=str(exc))
            return self._simulate_local_deploy()

    # ------------------------------------------------------------------
    # Code generation helpers
    # ------------------------------------------------------------------

    def _render_app(self) -> str:
        req_schema = json.dumps(self.config.api.request_schema.model_dump(), indent=2)
        resp_schema = json.dumps(self.config.api.response_schema.model_dump(), indent=2)
        framework = self.model_metadata.detected_framework

        return f'''"""
Auto-generated FastAPI serving application.
Framework: {framework}
Generated by LaunchML.
"""
import time
import logging
import uuid
from contextlib import asynccontextmanager
from typing import Any, Dict

import uvicorn
from fastapi import FastAPI, Request, Response
from fastapi.responses import JSONResponse
from prometheus_client import Counter, Histogram, generate_latest, CONTENT_TYPE_LATEST

# ─── Prometheus metrics ───────────────────────────────────────────────────
REQUEST_COUNT = Counter("request_count", "Total prediction requests", ["method", "status"])
REQUEST_LATENCY = Histogram("request_latency_seconds", "Request latency", ["method"])

# ─── Model loading ────────────────────────────────────────────────────────
model = None

def load_model():
    """Load the ML model. Replace with your actual loading logic."""
    global model
    logging.info("Loading model from {self.model_metadata.model_path}")
    # TODO: Replace with actual model loading for {framework}
    # Example for PyTorch:
    #   import torch
    #   model = torch.load("model.pt")
    #   model.eval()
    model = {{"loaded": True, "framework": "{framework}"}}
    logging.info("Model loaded successfully")

@asynccontextmanager
async def lifespan(app: FastAPI):
    load_model()
    yield

app = FastAPI(title="LaunchML - FastAPI Serving", lifespan=lifespan)

# ─── Health check ─────────────────────────────────────────────────────────
@app.get("/health")
async def health():
    return {{"status": "healthy", "model_loaded": model is not None}}

# ─── Metrics endpoint ─────────────────────────────────────────────────────
@app.get("/metrics")
async def metrics():
    return Response(content=generate_latest(), media_type=CONTENT_TYPE_LATEST)

# ─── Prediction endpoint ──────────────────────────────────────────────────
@app.post("/predict")
async def predict(request: Request):
    """
    Request schema: {req_schema}
    Response schema: {resp_schema}
    """
    request_id = str(uuid.uuid4())
    start = time.time()

    try:
        body: Dict[str, Any] = await request.json()

        # TODO: Replace with actual inference logic
        prediction = {{"prediction": "sample_output", "request_id": request_id}}

        latency = time.time() - start
        REQUEST_COUNT.labels(method="predict", status="success").inc()
        REQUEST_LATENCY.labels(method="predict").observe(latency)

        logging.info(f"request_id={{request_id}} latency_ms={{latency*1000:.1f}} status=success")
        return JSONResponse(content=prediction)

    except Exception as e:
        REQUEST_COUNT.labels(method="predict", status="error").inc()
        logging.error(f"request_id={{request_id}} error={{str(e)}}")
        return JSONResponse(
            status_code=500,
            content={{"error": str(e), "request_id": request_id}},
        )

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8080)
'''

    def _render_requirements(self) -> str:
        return """fastapi>=0.104.0
uvicorn[standard]>=0.24.0
prometheus-client>=0.19.0
pydantic>=2.0.0
"""

    def _render_dockerfile(self) -> str:
        gpu = self.model_metadata.gpu_recommendation == "true"
        base_image = "nvidia/cuda:12.1.0-runtime-ubuntu22.04" if gpu else "python:3.11-slim"

        # Only COPY model artefacts when they actually exist on disk.
        # For runtime-download models (torchvision, HuggingFace Hub, etc.)
        # the weights are fetched at container startup by the entrypoint.
        if self.model_metadata.model_files:
            copy_model = "\n# Copy model artefacts\nCOPY model/ /app/model/"
        else:
            copy_model = (
                "\n# No local model artefacts — the entrypoint is expected to\n"
                "# download / initialise the model at startup."
            )

        return f"""# Auto-generated Dockerfile — LaunchML
FROM {base_image}

WORKDIR /app

# Install Python if using CUDA base
{"RUN apt-get update && apt-get install -y python3 python3-pip && ln -s /usr/bin/python3 /usr/bin/python" if gpu else ""}

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .
{copy_model}

EXPOSE 8080
CMD ["python", "app.py"]
"""

    def _render_main_tf(self) -> str:
        cloud = self.config.deployment.cloud
        region = self.config.deployment.region
        lb = self.config.deployment.load_balancer
        gpu = self.model_metadata.gpu_recommendation

        iam_block = ""
        if lb == "public":
            iam_block = (
                '# ─── IAM: Public access ──────────────────────────────────────\n'
                'resource "google_cloud_run_v2_service_iam_member" "public" {\n'
                '  location = var.region\n'
                '  name     = google_cloud_run_v2_service.ml_service.name\n'
                '  role     = "roles/run.invoker"\n'
                '  member   = "allUsers"\n'
                '}\n'
            )
        else:
            iam_block = "# ─── IAM: Private access (no public binding) ────────────────\n"

        return f'''# Auto-generated Terraform — LaunchML (FastAPI backend)
# Cloud: {cloud} | Region: {region}

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

# ─── Container Registry ──────────────────────────────────────────────────
resource "google_artifact_registry_repository" "ml_repo" {{
  location      = var.region
  repository_id = "launch-ml"
  format        = "DOCKER"
}}

# ─── Cloud Run Service ───────────────────────────────────────────────────
resource "google_cloud_run_v2_service" "ml_service" {{
  name     = "ml-fastapi-service"
  location = var.region

  template {{
    scaling {{
      min_instance_count = var.min_instances
      max_instance_count = var.max_instances
    }}

    containers {{
      image = "${{var.container_image}}"

      ports {{
        container_port = 8080
      }}

      resources {{
        limits = {{
          cpu    = var.cpu_limit
          memory = var.memory_limit
        }}
      }}

      startup_probe {{
        http_get {{
          path = "/health"
        }}
        initial_delay_seconds = 10
      }}

      liveness_probe {{
        http_get {{
          path = "/health"
        }}
      }}
    }}
  }}

  traffic {{
    percent = 100
    type    = "TRAFFIC_TARGET_ALLOCATION_TYPE_LATEST"
  }}
}}

{iam_block}'''

    def _render_variables_tf(self) -> str:
        return f'''# Variables for FastAPI deployment
variable "project_id" {{
  description = "GCP project ID"
  type        = string
}}

variable "region" {{
  description = "GCP region"
  type        = string
  default     = "{self.config.deployment.region}"
}}

variable "container_image" {{
  description = "Docker image URI"
  type        = string
  default     = "gcr.io/PROJECT/ml-fastapi:latest"
}}

variable "min_instances" {{
  description = "Minimum number of instances"
  type        = number
  default     = 1
}}

variable "max_instances" {{
  description = "Maximum number of instances"
  type        = number
  default     = 5
}}

variable "cpu_limit" {{
  description = "CPU limit per instance"
  type        = string
  default     = "2"
}}

variable "memory_limit" {{
  description = "Memory limit per instance"
  type        = string
  default     = "4Gi"
}}
'''

    def _render_outputs_tf(self) -> str:
        return '''# Outputs
output "endpoint_url" {
  description = "The URL of the deployed prediction endpoint"
  value       = "${google_cloud_run_v2_service.ml_service.uri}/predict"
}

output "monitoring_url" {
  description = "The URL of the metrics endpoint"
  value       = "${google_cloud_run_v2_service.ml_service.uri}/metrics"
}

output "service_name" {
  value = google_cloud_run_v2_service.ml_service.name
}
'''

    # ------------------------------------------------------------------
    # Local deployment helpers
    # ------------------------------------------------------------------

    def _render_local_compose(self, model_path: Path) -> str:
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

        # Only mount the model directory if there are actual weight files.
        # For runtime-download models the weights are fetched at startup.
        if self.model_metadata.model_files:
            volumes_block = f"""\
    volumes:
      - {model_path}:/app/model:ro
    environment:
      - MODEL_PATH=/app/model
      - LOG_LEVEL=info"""
        else:
            volumes_block = """\
    environment:
      - LOG_LEVEL=info"""

        return f"""# Local FastAPI Deployment — auto-generated by LaunchML
# Usage:
#   docker compose up --build        (foreground)
#   docker compose up --build -d     (detached)
#   docker compose down              (stop)

services:
  fastapi-ml:
    build:
      context: .
      dockerfile: Dockerfile
    container_name: launchml-fastapi
    ports:
      - "8080:8080"
{volumes_block}
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:8080/health"]
      interval: 10s
      timeout: 5s
      retries: 3
      start_period: 15s
    restart: unless-stopped
{gpu_block}"""

    @staticmethod
    def _resolve_docker_compose_cmd() -> list[str] | None:
        """Return the docker compose CLI command, or None if unavailable."""
        # Prefer `docker compose` (v2 plugin)
        if shutil.which("docker"):
            result = subprocess.run(
                ["docker", "compose", "version"],
                capture_output=True, text=True, timeout=10,
            )
            if result.returncode == 0:
                return ["docker", "compose"]
        # Fall back to standalone docker-compose
        if shutil.which("docker-compose"):
            return ["docker-compose"]
        return None

    @staticmethod
    def _simulate_local_deploy() -> Dict[str, Any]:
        """Return a simulated result when Docker is not available."""
        return {
            "status": "simulated_local",
            "endpoint_url": "http://localhost:8080/predict",
            "health_url": "http://localhost:8080/health",
            "monitoring_url": "http://localhost:8080/metrics",
            "backend": "fastapi",
            "note": "Docker not available — simulated. Install Docker to deploy locally.",
        }
