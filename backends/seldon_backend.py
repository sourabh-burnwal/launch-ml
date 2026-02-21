"""
Seldon Core Backend — Kubernetes-native ML serving with advanced traffic management.

Best for:
    - Kubernetes-native environments
    - A/B testing and canary deployments
    - Multi-model serving with custom routing
    - Organizations already running Seldon on K8s
    - **Local deployment** via Docker Compose with seldon-core-microservice
"""

from __future__ import annotations

import json
import shutil
import subprocess
from pathlib import Path
from typing import Any, Dict

from backends.base import BackendCapabilities, DeploymentBackend
from backends.registry import register_backend
from utils.logger import get_logger

log = get_logger(__name__)


@register_backend
class SeldonBackend(DeploymentBackend):
    name = "seldon"
    description = (
        "Seldon Core — Kubernetes-native ML serving with A/B testing, "
        "canary rollouts, and advanced traffic management."
    )
    capabilities = BackendCapabilities(
        supports_gpu=True,
        supports_batching=True,
        supports_autoscaling=True,
        supports_multi_model=True,
        supports_streaming=False,
        managed_service=False,
        supports_local_deployment=True,
        max_model_size_gb=20.0,
        supported_frameworks=["pytorch", "tensorflow", "onnx", "transformers"],
        typical_latency_ms=20,
        complexity="medium",
        cost_tier="medium",
    )

    def validate(self) -> bool:
        fw = self.model_metadata.detected_framework
        if fw not in self.capabilities.supported_frameworks and fw != "unknown":
            return False
        return True

    def generate_inference_code(self) -> Dict[str, str]:
        self.ensure_dirs()
        files = {
            "Model.py": self._render_model_class(),
            "requirements.txt": self._render_requirements(),
            "seldon_deployment.yaml": self._render_seldon_yaml(),
            "Dockerfile": self._render_dockerfile(),
        }

        # Copy user entrypoint so Model.py can import it at runtime.
        if self.model_metadata.has_custom_entrypoint and self.model_metadata.entrypoint_path:
            ep_src = Path(self.model_metadata.entrypoint_path)
            if ep_src.is_file():
                ep_dst = self.serving_dir / ep_src.name
                import shutil as _shutil
                _shutil.copy2(ep_src, ep_dst)
                log.info("entrypoint_copied", src=str(ep_src), dst=str(ep_dst))

        written = self.write_files(self.serving_dir, files)
        log.info("seldon_code_generated", files=written)
        return files

    def generate_terraform(self) -> Dict[str, str]:
        self.ensure_dirs()
        files = {
            "main.tf": self._render_main_tf(),
            "variables.tf": self._render_variables_tf(),
            "outputs.tf": self._render_outputs_tf(),
        }
        written = self.write_files(self.terraform_dir, files)
        log.info("seldon_terraform_generated", files=written)
        return files

    def deploy(self) -> Dict[str, Any]:
        log.info("seldon_deploy_start")
        return {
            "status": "deployed",
            "endpoint_url": f"https://ml-seldon-{self.config.deployment.region}.example.com/seldon/default/ml-model/api/v1.0/predictions",
            "health_url": f"https://ml-seldon-{self.config.deployment.region}.example.com/seldon/default/ml-model/api/v1.0/health/status",
            "backend": self.name,
        }

    def setup_observability(self) -> Dict[str, str]:
        is_local = self.config.deployment.is_local
        instructions: Dict[str, str] = {}
        if self.config.observability.enable_metrics:
            if is_local:
                instructions["metrics"] = (
                    "Seldon microservice exposes Prometheus metrics at http://localhost:6000/prometheus\n"
                    "Key metrics: seldon_api_executor_server_requests_seconds, "
                    "seldon_api_executor_server_requests_total\n"
                    "You can scrape this endpoint with a local Prometheus instance."
                )
            else:
                instructions["metrics"] = (
                    "Seldon Core exposes Prometheus metrics automatically.\n"
                    "Install Seldon Analytics dashboard for Grafana:\n"
                    "  helm install seldon-analytics seldon-charts/seldon-analytics\n"
                    "Key metrics: seldon_api_executor_server_requests_seconds, "
                    "seldon_api_executor_server_requests_total"
                )
        if self.config.observability.enable_logging:
            if is_local:
                instructions["logging"] = (
                    "Container logs stream to stdout (visible via docker compose logs).\n"
                    "Run: docker compose -f generated/serving_code/docker-compose.yaml logs -f"
                )
            else:
                instructions["logging"] = (
                    "Enable request logging in SeldonDeployment spec:\n"
                    "  predictors[0].logger.mode: all\n"
                    "Logs are sent to a configured Kafka/ELK endpoint."
                )
        return instructions

    # ------------------------------------------------------------------
    # Local deployment
    # ------------------------------------------------------------------

    def generate_local_compose(self) -> Dict[str, str]:
        """Generate docker-compose.yaml for local Seldon serving."""
        self.ensure_dirs()
        model_path = Path(self.model_metadata.model_path).resolve()
        files = {
            "docker-compose.yaml": self._render_local_compose(model_path),
        }
        written = self.write_files(self.serving_dir, files)
        log.info("seldon_local_compose_generated", files=written)
        return files

    def deploy_local(self) -> Dict[str, Any]:
        """Build image and start Seldon microservice container locally."""
        log.info("seldon_local_deploy_start")
        compose_dir = self.serving_dir.resolve()
        compose_file = compose_dir / "docker-compose.yaml"

        if not compose_file.is_file():
            log.error("seldon_compose_missing", path=str(compose_file))
            return {"status": "failed", "error": "docker-compose.yaml not found"}

        docker_cmd = self._resolve_docker_compose_cmd()
        if docker_cmd is None:
            log.warning("docker_compose_not_found", msg="Simulating local deployment")
            return self._simulate_local_deploy()

        try:
            subprocess.run(
                [*docker_cmd, "down", "--remove-orphans"],
                cwd=compose_dir, capture_output=True, text=True, timeout=60,
            )
            log.info("seldon_local_build")
            build = subprocess.run(
                [*docker_cmd, "build"],
                cwd=compose_dir, capture_output=True, text=True, timeout=300,
            )
            if build.returncode != 0:
                log.error("seldon_local_build_failed", stderr=build.stderr[:500])
                return {"status": "failed", "error": build.stderr[:500]}

            log.info("seldon_local_up")
            up = subprocess.run(
                [*docker_cmd, "up", "-d"],
                cwd=compose_dir, capture_output=True, text=True, timeout=120,
            )
            if up.returncode != 0:
                log.error("seldon_local_up_failed", stderr=up.stderr[:500])
                return {"status": "failed", "error": up.stderr[:500]}

            return {
                "status": "running_locally",
                "endpoint_url": "http://localhost:5000/predict",
                "health_url": "http://localhost:5000/health/status",
                "monitoring_url": "http://localhost:6000/prometheus",
                "backend": self.name,
                "stop_command": f"cd {compose_dir} && {' '.join(docker_cmd)} down",
            }
        except subprocess.TimeoutExpired:
            log.error("seldon_local_timeout")
            return {"status": "failed", "error": "Docker command timed out"}
        except Exception as exc:
            log.error("seldon_local_error", error=str(exc))
            return self._simulate_local_deploy()

    # ------------------------------------------------------------------
    # Code generation
    # ------------------------------------------------------------------

    def _render_model_class(self) -> str:
        if self.model_metadata.has_custom_entrypoint and self.model_metadata.entrypoint_path:
            return self._render_model_class_with_entrypoint()
        return self._render_model_class_generic()

    def _render_model_class_with_entrypoint(self) -> str:
        fw = self.model_metadata.detected_framework
        ep_name = Path(self.model_metadata.entrypoint_path).stem

        return f'''"""
Seldon Core Model Class — auto-generated by LaunchML.
Wraps user entrypoint: {ep_name}.py
Framework: {fw}

This class implements the Seldon Python wrapper interface and delegates
to the user's predict() function from their entrypoint.
"""
import importlib.util
import inspect
import logging
import os
import sys
from typing import Any, Dict, List, Union

import numpy as np

logger = logging.getLogger(__name__)


class Model:
    """Seldon model wrapper — delegates to user's {ep_name}.predict()."""

    def __init__(self) -> None:
        self._user_module = None
        self._predict_fn = None
        self.ready = False

    def load(self) -> None:
        """Load user entrypoint module and extract predict function."""
        logger.info("Loading user entrypoint: {ep_name}.py")
        module_path = os.path.join(
            os.path.dirname(os.path.abspath(__file__)),
            "{ep_name}.py",
        )
        spec = importlib.util.spec_from_file_location("user_{ep_name}", module_path)
        if spec is None or spec.loader is None:
            raise RuntimeError(f"Cannot load entrypoint: {{module_path}}")

        module = importlib.util.module_from_spec(spec)
        saved_argv = sys.argv[:]
        sys.argv = ["{ep_name}.py", "http://placeholder"]
        try:
            spec.loader.exec_module(module)
        except SystemExit:
            logger.warning("User entrypoint called sys.exit() during import — ignored.")
        finally:
            sys.argv = saved_argv

        self._user_module = module
        self._predict_fn = getattr(module, "predict", None)
        if self._predict_fn is None:
            raise RuntimeError("{ep_name}.py does not define a predict() function")

        # If the module has a load_model function, call it
        load_fn = getattr(module, "load_model", None)
        if callable(load_fn):
            load_fn()

        self.ready = True
        logger.info("User entrypoint loaded successfully")

    def predict(
        self,
        X: np.ndarray,
        names: List[str] | None = None,
        meta: Dict[str, Any] | None = None,
    ) -> Union[np.ndarray, Dict[str, Any]]:
        """Delegate to user's predict function."""
        sig = inspect.signature(self._predict_fn)
        params = list(sig.parameters.keys())

        if len(params) == 1:
            result = self._predict_fn(X)
        elif len(params) == 0:
            result = self._predict_fn()
        else:
            result = self._predict_fn(X, names=names, meta=meta)

        return result

    def health_status(self) -> Dict[str, Any]:
        return {{"status": "healthy", "ready": self.ready}}
'''

    def _render_model_class_generic(self) -> str:
        fw = self.model_metadata.detected_framework
        return f'''"""
Seldon Core Model Class — auto-generated by LaunchML.
Framework: {fw}

This class implements the Seldon Python wrapper interface.
See: https://docs.seldon.io/projects/seldon-core/en/latest/python/python_wrapping_docker.html
"""
import logging
from typing import Any, Dict, List, Union
import numpy as np

logger = logging.getLogger(__name__)


class Model:
    """Seldon model wrapper for {fw} model."""

    def __init__(self) -> None:
        self.model = None
        self.ready = False

    def load(self) -> None:
        """Load model artefacts. Called once at startup."""
        logger.info("Loading {fw} model...")
        # TODO: Replace with actual model loading
        # Example for PyTorch:
        #   import torch
        #   self.model = torch.load("/mnt/models/model.pt")
        #   self.model.eval()
        self.model = {{"framework": "{fw}", "loaded": True}}
        self.ready = True
        logger.info("Model loaded successfully")

    def predict(
        self,
        X: np.ndarray,
        names: List[str] | None = None,
        meta: Dict[str, Any] | None = None,
    ) -> Union[np.ndarray, Dict[str, Any]]:
        """Run inference.

        Args:
            X: Input array (batch_size, features).
            names: Feature names.
            meta: Request metadata.

        Returns:
            Prediction array or dict.
        """
        logger.info(f"Prediction request: shape={{X.shape}}")

        # TODO: Replace with actual inference
        result = np.zeros((X.shape[0], 1))

        return result

    def health_status(self) -> Dict[str, Any]:
        return {{"status": "healthy", "ready": self.ready}}
'''

    def _render_requirements(self) -> str:
        base = [
            "numpy>=1.24.0",
            "seldon-core>=1.17.0",
        ]
        ep_deps = self.model_metadata.entrypoint_dependencies or []
        existing = {p.split(">=")[0].split("==")[0].lower() for p in base}
        for dep in ep_deps:
            if dep.lower() not in existing:
                base.append(dep)
                existing.add(dep.lower())
        return "\n".join(base) + "\n"

    def _render_seldon_yaml(self) -> str:
        gpu = self.model_metadata.gpu_recommendation == "true"
        return f'''# Seldon Deployment manifest — auto-generated by LaunchML
apiVersion: machinelearning.seldon.io/v1
kind: SeldonDeployment
metadata:
  name: ml-model
  namespace: default
spec:
  predictors:
    - name: default
      replicas: {self.config.deployment.expected_rps // 50 + 1}
      graph:
        name: model
        implementation: SKLEARN_SERVER  # or CUSTOM for Python wrapper
        modelUri: "gs://your-bucket/model"
        envSecretRefName: seldon-init-container-secret
        children: []
      componentSpecs:
        - spec:
            containers:
              - name: model
                image: "your-registry/ml-model:latest"
                resources:
                  requests:
                    cpu: "1"
                    memory: "2Gi"
                    {"nvidia.com/gpu: '1'" if gpu else ""}
                  limits:
                    cpu: "2"
                    memory: "4Gi"
                    {"nvidia.com/gpu: '1'" if gpu else ""}
      traffic: 100
      {"logger:" if self.config.observability.enable_logging else ""}
      {"  mode: all" if self.config.observability.enable_logging else ""}
'''

    def _render_dockerfile(self) -> str:
        return """# Seldon Core Python Wrapper — auto-generated
FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# Seldon expects the model class at this path
ENV MODEL_NAME=Model
ENV SERVICE_TYPE=MODEL

EXPOSE 5000
CMD exec seldon-core-microservice $MODEL_NAME --service-type $SERVICE_TYPE
"""

    def _render_main_tf(self) -> str:
        region = self.config.deployment.region
        return f'''# Auto-generated Terraform — LaunchML (Seldon backend)
terraform {{
  required_version = ">= 1.5.0"
  required_providers {{
    google = {{
      source  = "hashicorp/google"
      version = "~> 5.0"
    }}
    kubernetes = {{
      source  = "hashicorp/kubernetes"
      version = "~> 2.25"
    }}
    helm = {{
      source  = "hashicorp/helm"
      version = "~> 2.12"
    }}
  }}
}}

provider "google" {{
  project = var.project_id
  region  = var.region
}}

# ─── GKE Cluster ─────────────────────────────────────────────────────────
resource "google_container_cluster" "seldon_cluster" {{
  name     = "ml-seldon-cluster"
  location = var.region

  initial_node_count = var.node_count
  deletion_protection = false

  node_config {{
    machine_type = var.machine_type
    oauth_scopes = ["https://www.googleapis.com/auth/cloud-platform"]
  }}
}}

# ─── Seldon Core Helm Release ────────────────────────────────────────────
resource "helm_release" "seldon_core" {{
  name       = "seldon-core"
  repository = "https://storage.googleapis.com/seldon-charts"
  chart      = "seldon-core-operator"
  namespace  = "seldon-system"
  create_namespace = true

  set {{
    name  = "usageMetrics.enabled"
    value = "true"
  }}

  set {{
    name  = "istio.enabled"
    value = "false"
  }}

  depends_on = [google_container_cluster.seldon_cluster]
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
  default = "n1-standard-4"
}}

variable "node_count" {{
  type    = number
  default = 3
}}
'''

    def _render_outputs_tf(self) -> str:
        return '''output "endpoint_url" {
  description = "Seldon prediction endpoint"
  value       = "http://${google_container_cluster.seldon_cluster.endpoint}/seldon/default/ml-model/api/v1.0/predictions"
}

output "monitoring_url" {
  description = "Prometheus metrics URL"
  value       = "http://${google_container_cluster.seldon_cluster.endpoint}:9090"
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

        # Only mount model dir when weight files exist on disk.
        if self.model_metadata.model_files:
            volumes_block = f"""\
    volumes:
      - {model_path}:/mnt/models:ro"""
        else:
            volumes_block = """\
    # No local model weights — entrypoint downloads at startup"""

        return f"""# Local Seldon Deployment — auto-generated by LaunchML
# Usage:
#   docker compose up --build        (foreground)
#   docker compose up --build -d     (detached)
#   docker compose down              (stop)

services:
  seldon-ml:
    build:
      context: .
      dockerfile: Dockerfile
    container_name: launchml-seldon
    ports:
      - "5000:5000"    # REST prediction API
      - "5001:5001"    # gRPC
      - "6000:6000"    # Prometheus metrics
{volumes_block}
    environment:
      - MODEL_NAME=Model
      - SERVICE_TYPE=MODEL
      - PERSISTENCE=0
      - LOG_LEVEL=INFO
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:5000/health/status"]
      interval: 10s
      timeout: 5s
      retries: 3
      start_period: 20s
    restart: unless-stopped
{gpu_block}"""

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

    @staticmethod
    def _simulate_local_deploy() -> Dict[str, Any]:
        """Return a simulated result when Docker is not available."""
        return {
            "status": "simulated_local",
            "endpoint_url": "http://localhost:5000/predict",
            "health_url": "http://localhost:5000/health/status",
            "monitoring_url": "http://localhost:6000/prometheus",
            "backend": "seldon",
            "note": "Docker not available — simulated. Install Docker to deploy locally.",
        }
