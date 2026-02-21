# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [0.1.0] - 2026-02-21

### Added

- Initial POC release of **LaunchML**
- LangGraph-based 6-node deployment pipeline
- Model analyzer with framework auto-detection (PyTorch, TensorFlow, ONNX, HuggingFace Transformers)
- LLM-driven deployment strategy agent (OpenAI, Anthropic, Gemini)
- Four deployment backends:
  - **FastAPI** — lightweight serving with Cloud Run
  - **NVIDIA Triton** — high-performance GPU inference with GKE
  - **Seldon Core** — Kubernetes-native serving with Helm
  - **Vertex AI** — fully managed GCP prediction service
- Plugin architecture with `@register_backend` auto-discovery
- Dynamic Terraform generation per backend
- Terraform runner with simulation mode for demo/CI
- Prometheus metrics and structured logging per backend
- Mock LLM fallback for running without API keys
- Rich CLI output with `--dry-run`, `--json-logs`, `--verbose` options
- Deployment summary generation (`DEPLOYMENT_SUMMARY.md`)
- Pydantic v2 config validation
- Structured logging via structlog + Rich
