# Contributing to LaunchML

First off — thank you for considering a contribution! Every bug report, feature request, documentation improvement, and code patch helps make LaunchML better for everyone.

## Table of Contents

- [Code of Conduct](#code-of-conduct)
- [How Can I Contribute?](#how-can-i-contribute)
- [Development Setup](#development-setup)
- [Adding a New Backend](#adding-a-new-backend)
- [Pull Request Process](#pull-request-process)
- [Style Guide](#style-guide)
- [Reporting Bugs](#reporting-bugs)
- [Suggesting Features](#suggesting-features)

---

## Code of Conduct

This project follows the [Contributor Covenant Code of Conduct](CODE_OF_CONDUCT.md). By participating, you are expected to uphold this code. Please report unacceptable behavior via GitHub Issues.

---

## How Can I Contribute?

| Type | Description |
|------|-------------|
| **Bug reports** | Found something broken? [Open an issue](../../issues/new?template=bug_report.md) |
| **Feature requests** | Have an idea? [Start a discussion](../../issues/new?template=feature_request.md) |
| **New backend** | Add support for a new inference server or cloud provider |
| **Documentation** | Fix typos, improve examples, add guides |
| **Code quality** | Add tests, improve types, refactor internals |

---

## Development Setup

```bash
# 1. Fork and clone
git clone https://github.com/<your-username>/launch-ml.git
cd launch-ml

# 2. Create a virtual environment
python -m venv .venv
source .venv/bin/activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. (Optional) Set up an LLM API key
cp .env.example .env
# Edit .env with your key

# 5. Verify the pipeline runs
python deploy.py --config deploy_config.yaml
```

---

## Adding a New Backend

This is the most impactful contribution you can make. The plugin architecture is designed to make this easy.

### Step-by-step

1. **Create a file** in `backends/`:

```python
# backends/kserve_backend.py
from backends.base import DeploymentBackend, BackendCapabilities
from backends.registry import register_backend

@register_backend
class KServeBackend(DeploymentBackend):
    name = "kserve"
    description = "KServe (formerly KFServing) — Kubernetes ML serving."
    capabilities = BackendCapabilities(
        supports_gpu=True,
        supports_batching=True,
        supports_autoscaling=True,
        managed_service=False,
        supported_frameworks=["pytorch", "tensorflow", "onnx", "transformers"],
        typical_latency_ms=20,
        complexity="medium",
        cost_tier="medium",
    )

    def validate(self) -> bool: ...
    def generate_inference_code(self) -> dict[str, str]: ...
    def generate_terraform(self) -> dict[str, str]: ...
    def deploy(self) -> dict: ...
    def setup_observability(self) -> dict[str, str]: ...
```

2. **That's it.** The `@register_backend` decorator handles auto-discovery. The LLM strategy agent will automatically consider your backend's capabilities on the next run.

3. **Test it** by setting `framework_hint` and other config values that would favor your backend, then run the pipeline.

### Backend checklist

- [ ] Implements all 5 abstract methods from `DeploymentBackend`
- [ ] Sets `name`, `description`, and `capabilities` class attributes
- [ ] `generate_inference_code()` writes files under `generated/serving_code/`
- [ ] `generate_terraform()` writes files under `generated/terraform/`
- [ ] `setup_observability()` returns monitoring instructions
- [ ] Uses `@register_backend` decorator
- [ ] No hardcoded credentials

---

## Pull Request Process

1. **Fork** the repo and create a feature branch from `main`:
   ```bash
   git checkout -b feature/my-backend
   ```

2. **Make your changes** following the [style guide](#style-guide).

3. **Test** that the pipeline runs end-to-end:
   ```bash
   python deploy.py --config deploy_config.yaml
   ```

4. **Commit** with a clear message:
   ```bash
   git commit -m "feat: add KServe backend with GKE Terraform"
   ```

5. **Push** and open a Pull Request against `main`.

6. Fill out the PR template. A maintainer will review your PR — usually within a few days.

### Commit message conventions

We loosely follow [Conventional Commits](https://www.conventionalcommits.org/):

| Prefix | Usage |
|--------|-------|
| `feat:` | New feature |
| `fix:` | Bug fix |
| `docs:` | Documentation only |
| `refactor:` | Code change that neither fixes a bug nor adds a feature |
| `chore:` | Maintenance (deps, CI, etc.) |

---

## Style Guide

- **Python 3.11+** — use modern type hints (`dict[str, Any]`, `list[str]`, `X | Y`)
- **Fully typed** — all functions should have type annotations
- **Docstrings** — use Google-style docstrings for public functions and classes
- **No hardcoded secrets** — read credentials from environment variables
- **Structured logging** — use `from utils.logger import get_logger` instead of `print()`
- **Fail gracefully** — catch exceptions, log them, and provide helpful error messages

---

## Reporting Bugs

When filing a bug report, please include:

1. **Python version** (`python --version`)
2. **OS** (macOS, Linux, Windows)
3. **Steps to reproduce**
4. **Expected vs. actual behavior**
5. **Full error output** (with `--verbose` flag)

```bash
python deploy.py --config deploy_config.yaml --verbose 2>&1 | tee debug.log
```

---

## Suggesting Features

Feature requests are welcome! When suggesting a feature, please describe:

1. **The problem** you're trying to solve
2. **Your proposed solution**
3. **Alternatives** you've considered

---

## License

By contributing, you agree that your contributions will be licensed under the [MIT License](LICENSE).
