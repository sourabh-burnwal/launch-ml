# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [0.1.0] - 2026-02-22

### Added

- Initial release of **LaunchML** — AI-driven ML model deployment from directory to endpoint in one command
- Installable Python package (`pip install .`) with `launch-ml` CLI entry point
- CLI accepts `--model-dir`, `--config`, `--output-dir`, `--dry-run`, `--backend`, `--verbose`, and `--json-logs`
- LangGraph-based 6-node deployment pipeline (analyze → strategize → generate → deploy → observe → summarize)
- Model analyzer with framework auto-detection (PyTorch, TensorFlow, ONNX, HuggingFace Transformers)
- LLM-driven deployment strategy agent with support for OpenAI, Anthropic, and Google Gemini
- Four deployment backends with `@register_backend` plugin auto-discovery:
  - **FastAPI** — lightweight serving with Docker Compose (local) or Cloud Run (GCP)
  - **NVIDIA Triton** — high-performance GPU inference with GKE
  - **Seldon Core** — Kubernetes-native serving with Helm
  - **Vertex AI** — fully managed GCP prediction service
- Dynamic Terraform generation per backend with simulation mode for demo/CI
- User-configurable output directory (`--output-dir`) for all generated artifacts
- Prometheus metrics and structured logging per backend
- Deployment summary generation (`DEPLOYMENT_SUMMARY.md`)
- Pydantic v2 configuration validation (`deploy_config.yaml`)
- Structured logging via structlog + Rich
- Rich CLI output with banner, progress steps, and result summary
- Example deployments: FastAPI sentiment analysis, FastAPI image prediction, Triton sentiment analysis
- Project logo and README with quickstart guide
