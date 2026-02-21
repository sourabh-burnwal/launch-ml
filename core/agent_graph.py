"""
Agent Graph — LangGraph-based orchestration pipeline for LaunchML.

This is the brain of the system. It wires together nodes into a
directed acyclic graph with a conditional branch for local vs cloud
deployment:

    ┌─────────────────┐
    │ Model Analyzer  │  ← Node 1: scan model dir, emit metadata
    └───────┬─────────┘
            ▼
    ┌─────────────────┐
    │ Strategy Agent  │  ← Node 2: LLM reasons about best backend
    └───────┬─────────┘
            ▼
    ┌─────────────────┐
    │ Backend Plugin  │  ← Node 3: generate inference code + IaC / compose
    └───────┬─────────┘
            ▼
       ┌────┴────┐
       │ local?  │
       └────┬────┘
      ┌─────┴──────┐
      ▼            ▼
 ┌──────────┐ ┌──────────────┐
 │ Local    │ │ Terraform /  │
 │ Docker   │ │ Cloud Deploy │
 └────┬─────┘ └──────┬───────┘
      └───────┬──────┘
              ▼
    ┌─────────────────┐
    │ Observability   │  ← Node 6: setup monitoring, emit summary
    └─────────────────┘

State is carried in a ``PipelineState`` TypedDict that grows as each
node appends its outputs.

Architecture decisions:
    - We use LangGraph's ``StateGraph`` for explicit, auditable control flow.
    - Each node is a pure function ``(state) -> partial state update``.
    - LLM calls happen only in Node 2 (Strategy Agent); everything else
      is deterministic so the pipeline is reproducible.
    - When ``cloud: local`` is set, the graph skips Terraform and instead
      generates docker-compose files, builds images, and starts containers
      locally.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, TypedDict

from langgraph.graph import END, StateGraph

from backends.base import DeploymentBackend
from backends.registry import discover_backends, get_backend, list_backends
from core.config_loader import DeployConfig
from core.llm_client import create_llm_or_mock, invoke_llm
from core.model_analyzer import ModelAnalyzer, ModelMetadata
from utils.logger import get_logger, step, success, info, warn, error as cli_error
from utils.terraform_runner import TerraformRunner

log = get_logger(__name__)


# ─── Pipeline State ───────────────────────────────────────────────────────────

class PipelineState(TypedDict, total=False):
    """Typed state dictionary flowing through the LangGraph."""
    # Inputs
    config: Dict[str, Any]

    # Node 1 outputs
    model_metadata: Dict[str, Any]

    # Node 2 outputs
    strategy_decision: Dict[str, Any]
    strategy_reasoning: str

    # Node 3 outputs
    generated_serving_files: Dict[str, str]
    generated_terraform_files: Dict[str, str]

    # Node 5 outputs
    deployment_result: Dict[str, Any]

    # Node 6 outputs
    observability_instructions: Dict[str, str]
    deployment_summary: str

    # Error tracking
    errors: List[str]


# ─── Node implementations ─────────────────────────────────────────────────────

def node_model_analyzer(state: PipelineState) -> PipelineState:
    """Node 1: Analyze the model directory and produce structured metadata."""
    step("Node 1 — Analyzing model directory")

    config_dict = state["config"]
    model_path = config_dict["model"]["path"]
    framework_hint = config_dict["model"].get("framework_hint")
    entrypoint = config_dict["model"].get("entrypoint")

    analyzer = ModelAnalyzer(
        model_path=model_path,
        framework_hint=framework_hint,
        entrypoint=entrypoint,
    )
    metadata = analyzer.analyze()

    info(f"Framework: {metadata.detected_framework}")
    info(f"Model size: {metadata.total_size_mb:.1f} MB ({metadata.size_category})")
    info(f"GPU recommendation: {metadata.gpu_recommendation}")

    return {"model_metadata": metadata.to_dict()}


def node_strategy_agent(state: PipelineState) -> PipelineState:
    """Node 2: Use LLM to reason about the best deployment backend."""
    step("Node 2 — Strategy Agent reasoning about deployment backend")

    config_dict = state["config"]
    metadata = state["model_metadata"]
    is_local = config_dict.get("deployment", {}).get("cloud") == "local"

    # Ensure backends are discovered
    discover_backends()
    available = list_backends(local_only=is_local)

    if not available:
        warn("No backends available for the selected deployment mode, falling back to all backends")
        available = list_backends()

    # Build deployment mode context for the LLM
    deploy_mode = "LOCAL (Docker Compose on developer machine)" if is_local else "CLOUD"
    local_constraint = ""
    if is_local:
        local_constraint = (
            "\n\nIMPORTANT: The user has requested LOCAL DEPLOYMENT mode.\n"
            "The model will be containerized and run on the developer's local machine "
            "using Docker. Select the backend that is easiest to run locally while "
            "still meeting the model's requirements. Prefer simplicity for local dev.\n"
            "Instance type should be 'local' and scaling_strategy should be 'single_container'."
        )

    # Build the LLM prompt
    system_prompt = f"""You are an expert ML infrastructure engineer. Your task is to analyze 
a machine learning model and its deployment requirements, then select the optimal 
deployment backend.

Deployment mode: {deploy_mode}

You MUST respond with a valid JSON object (no markdown, no explanation outside the JSON).
The JSON must have these exact keys:
{{
  "selected_backend": "<backend_name>",
  "reasoning": "<detailed reasoning>",
  "instance_type": "<compute instance type>",
  "gpu_type": "<gpu type or 'none'>",
  "scaling_strategy": "<scaling approach>",
  "replicas_min": <int>,
  "replicas_max": <int>
}}

The selected_backend MUST be one of the available backend names provided.{local_constraint}"""

    user_prompt = f"""## Model Metadata
{json.dumps(metadata, indent=2)}

## User Deployment Config
- Deployment mode: {deploy_mode}
- Cloud: {config_dict.get('deployment', {}).get('cloud', 'gcp')}
- Region: {config_dict.get('deployment', {}).get('region', 'us-central1')}
- Latency target: {config_dict.get('deployment', {}).get('latency_target_ms', 200)}ms
- Expected RPS: {config_dict.get('deployment', {}).get('expected_rps', 50)}
- GPU required: {config_dict.get('deployment', {}).get('gpu_required', 'auto')}
- Load balancer: {config_dict.get('deployment', {}).get('load_balancer', 'public')}

## Available Backends
{json.dumps(available, indent=2)}

Select the best backend and explain your reasoning considering:
1. GPU needs vs model size
2. Throughput requirements ({config_dict.get('deployment', {}).get('expected_rps', 50)} RPS)
3. Latency target ({config_dict.get('deployment', {}).get('latency_target_ms', 200)}ms)
4. Cost efficiency
5. Operational complexity (managed vs self-managed)
6. Framework compatibility with {metadata.get('detected_framework', 'unknown')}"""

    # Create LLM and invoke
    from core.config_loader import LLMConfig
    llm_config = LLMConfig(**config_dict.get("llm", {}))
    llm = create_llm_or_mock(llm_config)
    decision = invoke_llm(llm, system_prompt=system_prompt, user_prompt=user_prompt)

    # Validate the selected backend exists
    selected = decision.get("selected_backend", "fastapi")
    if selected not in available:
        warn(f"LLM selected unknown backend '{selected}', falling back to 'fastapi'")
        selected = "fastapi"
        decision["selected_backend"] = selected

    reasoning = decision.get("reasoning", "No reasoning provided.")

    info(f"Selected backend: {selected}")
    info(f"Deployment mode: {'local' if is_local else 'cloud'}")
    info(f"Reasoning: {reasoning[:200]}...")

    log.info(
        "strategy_decision",
        reasoning=True,
        selected_backend=selected,
        local_deployment=is_local,
        full_reasoning=reasoning,
        decision=decision,
    )

    return {
        "strategy_decision": decision,
        "strategy_reasoning": reasoning,
    }


def node_backend_plugin(state: PipelineState) -> PipelineState:
    """Node 3: Execute the selected backend plugin to generate code + IaC/compose."""
    config_dict = state["config"]
    metadata_dict = state["model_metadata"]
    decision = state["strategy_decision"]
    selected = decision.get("selected_backend", "fastapi")
    is_local = config_dict.get("deployment", {}).get("cloud") == "local"

    if is_local:
        step("Node 3 — Generating inference code and Docker Compose (local mode)")
    else:
        step("Node 3 — Generating inference code and Terraform")

    # Reconstruct typed objects
    config = DeployConfig(**config_dict)
    metadata = _dict_to_metadata(metadata_dict)

    backend = get_backend(selected, config, metadata)

    # Validate
    if not backend.validate():
        warn(f"Backend '{selected}' validation failed, falling back to 'fastapi'")
        backend = get_backend("fastapi", config, metadata)

    # Generate inference code
    serving_files = backend.generate_inference_code()
    info(f"Generated {len(serving_files)} serving files")

    if is_local:
        # Generate docker-compose for local deployment
        compose_files = backend.generate_local_compose()
        info(f"Generated {len(compose_files)} local deployment files")
        return {
            "generated_serving_files": {**serving_files, **compose_files},
            "generated_terraform_files": {},
        }
    else:
        # Generate Terraform for cloud deployment
        terraform_files = backend.generate_terraform()
        info(f"Generated {len(terraform_files)} Terraform files")
        return {
            "generated_serving_files": serving_files,
            "generated_terraform_files": terraform_files,
        }


def node_deploy_executor(state: PipelineState) -> PipelineState:
    """Node 5: Deploy — locally via Docker or to cloud via Terraform."""
    config_dict = state["config"]
    decision = state["strategy_decision"]
    selected = decision.get("selected_backend", "fastapi")
    is_local = config_dict.get("deployment", {}).get("cloud") == "local"

    config = DeployConfig(**config_dict)
    metadata = _dict_to_metadata(state["model_metadata"])
    backend = get_backend(selected, config, metadata)

    if is_local:
        step("Node 5 — Deploying locally via Docker")
        deploy_result = backend.deploy_local()
        status = deploy_result.get("status", "unknown")
        endpoint = deploy_result.get("endpoint_url", "N/A")
        if status in ("running_locally", "simulated_local"):
            success(f"Local deployment complete! Endpoint: {endpoint}")
        else:
            warn(f"Local deployment finished with status: {status}")
    else:
        step("Node 5 — Deploying infrastructure via Terraform")

        terraform_dir = Path("generated/terraform")

        # Use TerraformRunner (auto-simulates if CLI not available)
        runner = TerraformRunner(terraform_dir, simulate=True)  # POC: always simulate

        info("Running terraform init...")
        runner.init()

        info("Running terraform apply...")
        runner.apply()

        outputs = runner.output()

        # Also call the backend's deploy() for any extra steps
        deploy_result = backend.deploy()

        # Merge terraform outputs with backend deploy result
        deploy_result.update(outputs)

        success(f"Deployment complete! Endpoint: {deploy_result.get('endpoint_url', 'N/A')}")

    return {"deployment_result": deploy_result}


def node_observability(state: PipelineState) -> PipelineState:
    """Node 6: Setup observability and generate deployment summary."""
    step("Node 6 — Configuring observability and generating summary")

    config_dict = state["config"]
    decision = state["strategy_decision"]
    selected = decision.get("selected_backend", "fastapi")
    deploy_result = state.get("deployment_result", {})

    config = DeployConfig(**config_dict)
    metadata = _dict_to_metadata(state["model_metadata"])
    backend = get_backend(selected, config, metadata)

    obs_instructions = backend.setup_observability()

    # Generate deployment summary
    summary = _generate_summary(
        config_dict=config_dict,
        metadata=state["model_metadata"],
        decision=decision,
        reasoning=state.get("strategy_reasoning", ""),
        deploy_result=deploy_result,
        obs_instructions=obs_instructions,
    )

    # Write summary to disk
    summary_path = Path("generated/DEPLOYMENT_SUMMARY.md")
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(summary)
    success(f"Deployment summary written to {summary_path}")

    return {
        "observability_instructions": obs_instructions,
        "deployment_summary": summary,
    }


# ─── Graph construction ───────────────────────────────────────────────────────

def build_graph() -> StateGraph:
    """Construct and compile the LangGraph deployment pipeline.

    Returns:
        A compiled LangGraph ``StateGraph`` ready for ``.invoke()``.
    """
    graph = StateGraph(PipelineState)

    # Add nodes
    graph.add_node("model_analyzer", node_model_analyzer)
    graph.add_node("strategy_agent", node_strategy_agent)
    graph.add_node("backend_plugin", node_backend_plugin)
    graph.add_node("deploy_executor", node_deploy_executor)
    graph.add_node("observability", node_observability)

    # Wire edges: linear pipeline
    graph.set_entry_point("model_analyzer")
    graph.add_edge("model_analyzer", "strategy_agent")
    graph.add_edge("strategy_agent", "backend_plugin")
    graph.add_edge("backend_plugin", "deploy_executor")
    graph.add_edge("deploy_executor", "observability")
    graph.add_edge("observability", END)

    return graph.compile()


def run_pipeline(config: DeployConfig) -> PipelineState:
    """High-level entry: build graph, seed state, and run.

    Args:
        config: Validated deployment configuration.

    Returns:
        Final pipeline state with all outputs.
    """
    # Discover all backend plugins
    discover_backends()

    # Build initial state — serialise config to dict for state transport
    initial_state: PipelineState = {
        "config": config.model_dump(),
        "errors": [],
    }

    compiled = build_graph()
    log.info("pipeline_start")

    final_state = compiled.invoke(initial_state)

    log.info("pipeline_complete")
    return final_state


# ─── Helpers ──────────────────────────────────────────────────────────────────

def _dict_to_metadata(d: Dict[str, Any]) -> ModelMetadata:
    """Reconstruct a ModelMetadata from its dict representation."""
    return ModelMetadata(
        model_path=d.get("model_path", ""),
        detected_framework=d.get("detected_framework", "unknown"),
        model_files=d.get("model_files", []),
        total_size_mb=d.get("total_size_mb", 0.0),
        size_category=d.get("size_category", "unknown"),
        has_tokenizer=d.get("has_tokenizer", False),
        has_config_json=d.get("has_config_json", False),
        has_custom_entrypoint=d.get("has_custom_entrypoint", False),
        entrypoint_path=d.get("entrypoint_path"),
        gpu_recommendation=d.get("gpu_recommendation", "auto"),
        extra=d.get("extra", {}),
    )


def _generate_summary(
    *,
    config_dict: Dict[str, Any],
    metadata: Dict[str, Any],
    decision: Dict[str, Any],
    reasoning: str,
    deploy_result: Dict[str, Any],
    obs_instructions: Dict[str, str],
) -> str:
    """Generate the DEPLOYMENT_SUMMARY.md content."""
    now = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")
    selected = decision.get("selected_backend", "unknown")
    endpoint = deploy_result.get("endpoint_url", "N/A")
    monitoring = deploy_result.get("monitoring_url", "N/A")
    is_local = config_dict.get("deployment", {}).get("cloud") == "local"

    obs_section = ""
    for key, val in obs_instructions.items():
        obs_section += f"### {key.title()}\n{val}\n\n"

    deploy_mode = "🏠 Local (Docker)" if is_local else "☁️ Cloud"
    stop_cmd = deploy_result.get("stop_command", "")

    if is_local:
        redeploy_section = f"""## 🔄 How to Redeploy

```bash
# Re-run the full pipeline
python deploy.py --config deploy_config.yaml

# Or rebuild and restart containers manually
cd generated/serving_code
docker compose up --build -d
```

## 💥 How to Stop Local Containers

```bash
{stop_cmd if stop_cmd else "cd generated/serving_code && docker compose down"}
```"""
        files_section = """## 📁 Generated Files

- `generated/serving_code/` — Inference server code + docker-compose.yaml
- `generated/DEPLOYMENT_SUMMARY.md` — This file"""
    else:
        redeploy_section = """## 🔄 How to Redeploy

```bash
# Re-run the full pipeline
python deploy.py --config deploy_config.yaml

# Or apply Terraform changes only
cd generated/terraform
terraform apply
```

## 💥 How to Destroy Infrastructure

```bash
cd generated/terraform
terraform destroy
```"""
        files_section = """## 📁 Generated Files

- `generated/serving_code/` — Inference server code
- `generated/terraform/` — Infrastructure as Code
- `generated/DEPLOYMENT_SUMMARY.md` — This file"""

    return f"""# LaunchML — Deployment Summary

> Generated at: {now}

---

## 📦 Model Information

| Property | Value |
|----------|-------|
| Path | `{metadata.get('model_path', 'N/A')}` |
| Framework | {metadata.get('detected_framework', 'unknown')} |
| Size | {metadata.get('total_size_mb', 0):.1f} MB ({metadata.get('size_category', 'unknown')}) |
| GPU Recommendation | {metadata.get('gpu_recommendation', 'auto')} |
| Has Tokenizer | {metadata.get('has_tokenizer', False)} |

## 🎯 Deployment Decision

| Property | Value |
|----------|-------|
| Deploy Mode | {deploy_mode} |
| Selected Backend | **{selected}** |
| Instance Type | {decision.get('instance_type', 'N/A')} |
| GPU Type | {decision.get('gpu_type', 'none')} |
| Scaling Strategy | {decision.get('scaling_strategy', 'N/A')} |
| Min Replicas | {decision.get('replicas_min', 1)} |
| Max Replicas | {decision.get('replicas_max', 5)} |

### Reasoning

{reasoning}

## 🌐 Endpoints

| Endpoint | URL |
|----------|-----|
| Prediction | `{endpoint}` |
| Monitoring | `{monitoring}` |

## 📊 Observability

{obs_section}

{redeploy_section}

{files_section}

---

*Generated by [LaunchML](https://github.com/your-org/launch-ml)*
"""
