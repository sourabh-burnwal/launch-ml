<p align="center">
  <img src="assets/launchml-logo.png" alt="LaunchML" width="500" />
</p>

<p align="center">
  <strong>AI-driven ML model deployment — from directory to endpoint in one command.</strong>
</p>

<p align="center">
  <a href="#quickstart">Quickstart</a> •
  <a href="#how-it-works">How It Works</a> •
  <a href="#configuration">Configuration</a> •
  <a href="#backends">Backends</a> •
  <a href="#extending">Extending</a> •
  <a href="#contributing">Contributing</a>
</p>

<p align="center">
  <img src="https://img.shields.io/badge/python-3.11+-blue?logo=python&logoColor=white" alt="Python 3.11+" />
  <img src="https://img.shields.io/badge/license-MIT-green" alt="License" />
  <img src="https://img.shields.io/badge/LangGraph-orchestrated-purple" alt="LangGraph" />
  <img src="https://img.shields.io/badge/IaC-Terraform-623CE4?logo=terraform" alt="Terraform" />
</p>

---

LaunchML analyzes a local ML model directory, uses an LLM to reason about the best deployment strategy, generates infrastructure-as-code, and deploys it — outputting a working prediction endpoint.

```bash
launch-ml --model-dir ./my_model --config deploy_config.yaml --output-dir ./output
```

**No Dockerfile writing. No Terraform authoring. No manual infra decisions.**

---

## Features

- **AI-driven deployment strategy** — An LLM agent analyzes your model and reasons about GPU needs, latency targets, throughput, and cost to pick the optimal backend.
- **Multiple inference backends** — FastAPI, NVIDIA Triton, Seldon Core, Google Vertex AI — with a plugin system to add more.
- **Automatic Terraform generation** — Infrastructure-as-code is generated dynamically based on the chosen strategy.
- **Framework auto-detection** — Detects PyTorch, TensorFlow, ONNX, and HuggingFace Transformers models automatically.
- **Multi-provider LLM support** — Works with OpenAI, Anthropic, or Google Gemini via API key.
- **Observability built-in** — Prometheus metrics and structured logging configured per backend.
- **Local deployment** — Deploy locally via Docker Compose. No cloud account required.
- **Plugin architecture** — Add a new backend in a single file with zero boilerplate.
- **Installable CLI** — `pip install .` and run from anywhere.

---

## Quickstart

### Prerequisites

- Python 3.11+
- (Optional) [Docker](https://www.docker.com/) for local deployment
- (Optional) [Terraform CLI](https://developer.hashicorp.com/terraform/install) for cloud deployments
- (Optional) An API key for one of: OpenAI, Anthropic, or Google Gemini

### Installation

```bash
git clone https://github.com/your-org/launch-ml.git
cd launch-ml
pip install .
```

> For development (editable mode):
> ```bash
> pip install -e .
> ```

### Set up your LLM

LaunchML uses an LLM to analyze your model and generate deployment code. Export one of the following API keys:

```bash
# Pick one:
export OPENAI_API_KEY=sk-...
export ANTHROPIC_API_KEY=sk-ant-...
export GOOGLE_API_KEY=AI...

# Or copy the example env file:
cp .env.example .env
# Then edit .env with your key
```

### Run the pipeline

```bash
launch-ml \
  --model-dir ./model_dir \
  --config deploy_config.yaml \
  --output-dir ./output
```

#### Run an example

```bash
# Deploy a FastAPI sentiment analysis service locally
launch-ml \
  --model-dir ./examples/fastapi/sentiment_analysis \
  --config ./examples/fastapi/sentiment_analysis/deploy_config.yaml \
  --output-dir ./output
```

You'll see:

```
╔══════════════════════════════════════════════════════════════╗
║                          LaunchML                            ║
║             AI-Driven Model Deployment Pipeline              ║
╚══════════════════════════════════════════════════════════════╝

▶ Node 1 — Analyzing model directory
ℹ Framework: transformers
ℹ Model size: 438.2 MB (medium)
ℹ GPU recommendation: true

▶ Node 2 — Strategy Agent reasoning about deployment backend
ℹ Selected backend: triton
ℹ Reasoning: Given the model size, GPU requirement, and 50 RPS target...

▶ Node 3 — Generating inference code and Terraform
ℹ Generated 3 serving files
ℹ Generated 3 Terraform files

▶ Node 5 — Deploying infrastructure
✔ Deployment complete! Endpoint: https://ml-triton-us-central1.example.com:8000/v2/models/model/infer

▶ Node 6 — Configuring observability and generating summary
✔ Deployment summary written to ./output/DEPLOYMENT_SUMMARY.md

══════════════════════════════════════════════════════════════
  ✅  Pipeline Complete
══════════════════════════════════════════════════════════════
```

### CLI Options

| Flag | Description |
|------|-------------|
| `--model-dir PATH` | Path to your model directory (required) |
| `--config PATH` | Path to your `deploy_config.yaml` (required) |
| `--output-dir PATH` | Directory for generated serving code, Terraform, and summaries (required) |
| `--backend / -b` | Force a specific backend: `fastapi`, `triton`, `seldon`, `vertex_ai` |
| `--dry-run` | Run analysis + strategy only, skip deployment |
| `--json-logs` | Emit structured JSON logs instead of colored text |
| `--verbose / -v` | Enable DEBUG-level logging |

---

## How It Works

LaunchML uses a **LangGraph** state graph to orchestrate six pipeline nodes:

```
┌─────────────────────┐
│  1. Model Analyzer  │  Scans model dir → framework, size, GPU heuristic
└──────────┬──────────┘
           ▼
┌─────────────────────┐
│  2. Strategy Agent  │  LLM reasons about best backend (logged + auditable)
└──────────┬──────────┘
           ▼
┌─────────────────────┐
│  3. Backend Plugin  │  Generates inference code + Terraform via plugin
└──────────┬──────────┘
           ▼
┌─────────────────────┐
│  4. Deploy Executor │  Runs terraform init → plan → apply (or Docker locally)
└──────────┬──────────┘
           ▼
┌─────────────────────┐
│  5. Observability   │  Configures metrics/logging, writes summary
└─────────────────────┘
```

Each node is a **pure function** that reads from and writes to a typed `PipelineState` dictionary. LLM calls only happen in Node 2, making the rest of the pipeline fully deterministic and reproducible.

---

## Configuration

Create a `deploy_config.yaml` (a template is included in the repo):

```yaml
model:
  # model path is provided via --model-dir CLI argument
  framework_hint: pytorch          # optional: pytorch | tensorflow | onnx | transformers
  entrypoint: predict.py           # optional: custom inference script

api:
  request_schema:
    type: object
    properties:
      text:
        type: string
  response_schema:
    type: object
    properties:
      prediction:
        type: string

deployment:
  cloud: gcp                       # gcp | aws | local
  region: us-central1
  latency_target_ms: 200
  expected_rps: 50
  load_balancer: public            # public | private
  gpu_required: auto               # auto | true | false

observability:
  enable_metrics: true
  enable_logging: true

llm:
  provider: openai                 # openai | anthropic | gemini
  model: gpt-4o
```

All fields have sensible defaults — you only need to set `llm.provider` to get started. Model path and output directory are provided as CLI arguments.

---

## Backends

The strategy agent chooses from these backends based on your model + requirements:

| Backend | Best For | GPU | Managed | Complexity |
|---------|----------|-----|---------|------------|
| **FastAPI** | Simple models, rapid iteration, custom logic | ✅ | ❌ | Low |
| **NVIDIA Triton** | High-throughput GPU inference, dynamic batching | ✅ | ❌ | High |
| **Seldon Core** | K8s-native, A/B testing, canary rollouts | ✅ | ❌ | Medium |
| **Vertex AI** | Zero-ops, fully managed GCP serving | ✅ | ✅ | Low |

The agent considers GPU needs, latency target, throughput, cost, and operational complexity when making its decision. The full reasoning is logged and included in the deployment summary.

---

## Project Structure

```
launch-ml/
├── pyproject.toml                # Package metadata & entry point
├── deploy_config.yaml            # Config template
│
├── launchml/                     # Installable package
│   ├── __init__.py
│   ├── cli.py                    # Click CLI entrypoint
│   │
│   ├── core/
│   │   ├── config_loader.py      # Pydantic v2 config validation
│   │   ├── model_analyzer.py     # Framework detection + model scanning
│   │   ├── llm_client.py         # Multi-provider LLM abstraction
│   │   └── agent_graph.py        # LangGraph pipeline (6 nodes)
│   │
│   ├── backends/
│   │   ├── base.py               # ABC + BackendCapabilities
│   │   ├── registry.py           # Auto-discovery plugin registry
│   │   ├── fastapi_backend.py
│   │   ├── triton_backend.py
│   │   ├── seldon_backend.py
│   │   └── vertex_ai_backend.py
│   │
│   ├── templates/                # Jinja2 templates
│   └── utils/
│       ├── logger.py             # Structured logging (structlog + Rich)
│       └── terraform_runner.py   # Safe Terraform subprocess wrapper
│
├── examples/                     # Example configs + predict.py files
│   └── fastapi/
│       ├── sentiment_analysis/
│       └── image_prediction/
│
└── model_dir/                    # Demo model for testing
```

---

## Extending

### Adding a New Backend

Create a single file in `launchml/backends/`:

```python
# launchml/backends/my_backend.py
from launchml.backends.base import DeploymentBackend, BackendCapabilities
from launchml.backends.registry import register_backend

@register_backend
class MyBackend(DeploymentBackend):
    name = "my_backend"
    description = "My custom inference backend."
    capabilities = BackendCapabilities(
        supports_gpu=True,
        supports_batching=True,
        managed_service=False,
        typical_latency_ms=15,
        complexity="medium",
        cost_tier="low",
    )

    def validate(self) -> bool:
        return True

    def generate_inference_code(self) -> dict[str, str]:
        self.ensure_dirs()
        files = {"server.py": "# your serving code"}
        self.write_files(self.serving_dir, files)
        return files

    def generate_terraform(self) -> dict[str, str]:
        self.ensure_dirs()
        files = {"main.tf": "# your terraform"}
        self.write_files(self.terraform_dir, files)
        return files

    def deploy(self) -> dict:
        return {"status": "deployed", "endpoint_url": "http://..."}

    def setup_observability(self) -> dict[str, str]:
        return {"metrics": "Prometheus at /metrics"}
```

That's it. The `@register_backend` decorator auto-registers it, and the strategy agent will see its capabilities on the next run.

### Adding a New LLM Provider

Edit `launchml/core/llm_client.py` and add a new branch in `create_llm()`. Follow the existing pattern for OpenAI/Anthropic/Gemini.

---

## Security

- **No secrets on disk** — API keys are read exclusively from environment variables.
- **No hardcoded credentials** — Terraform picks up cloud credentials from standard env vars (`GOOGLE_APPLICATION_CREDENTIALS`, `AWS_PROFILE`, etc.).
- **Public/private load balancer** — Controlled via config; private mode skips public IAM bindings.
- **`.env` is gitignored** — Even if you use a `.env` file locally, it won't be committed.

---

## Development

```bash
# Clone and install in editable mode
git clone https://github.com/your-org/launch-ml.git
cd launch-ml
pip install -e .

# Run with demo model (no API key needed)
launch-ml --model-dir ./model_dir --config deploy_config.yaml --output-dir ./output

# Run with verbose logging
launch-ml --model-dir ./model_dir --config deploy_config.yaml --output-dir ./output -v

# Run with JSON logs (for piping to jq, etc.)
launch-ml --model-dir ./model_dir --config deploy_config.yaml --output-dir ./output --json-logs

# Force a specific backend
launch-ml --model-dir ./model_dir --config deploy_config.yaml --output-dir ./output -b fastapi
```

---

## Roadmap

- [ ] AWS backend (SageMaker, ECS)
- [ ] Azure backend (AzureML)
- [ ] Model optimization suggestions (quantization, ONNX conversion)
- [ ] Interactive TUI mode
- [ ] GitHub Actions CI/CD integration
- [ ] Cost estimation before deployment
- [ ] Batch inference support

---

## License

This project is licensed under the [MIT License](LICENSE).

---

## Contributing

We welcome contributions! Please see [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

This project follows the [Contributor Covenant Code of Conduct](CODE_OF_CONDUCT.md).
