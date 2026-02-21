#!/usr/bin/env python3
"""
LaunchML — CLI Entrypoint
==========================

Production-quality CLI entrypoint that:
  1. Loads and validates deploy_config.yaml
  2. Runs the LangGraph agent pipeline
  3. Outputs a working deployed endpoint + summary

Usage (after ``pip install .``):
    launch-ml --model-dir ./my_model --config deploy_config.yaml --output-dir ./output
    launch-ml --model-dir ./my_model --config deploy_config.yaml --output-dir ./output --dry-run
    launch-ml --model-dir ./my_model --config deploy_config.yaml --output-dir ./output --backend fastapi

Environment Variables:
    OPENAI_API_KEY      — Required if llm.provider = openai
    ANTHROPIC_API_KEY   — Required if llm.provider = anthropic
    GOOGLE_API_KEY      — Required if llm.provider = gemini

Architecture:
    This file is deliberately thin — it handles CLI parsing and
    delegates everything to ``launchml.core.agent_graph.run_pipeline()``.
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import click
from dotenv import load_dotenv

from launchml.utils.logger import (
    configure_logging,
    console,
    step,
    success,
    error as cli_error,
    info,
)


# ─── CLI ──────────────────────────────────────────────────────────────────────

@click.command(context_settings={"help_option_names": ["-h", "--help"]})
@click.option(
    "--model-dir",
    "model_dir",
    required=True,
    type=click.Path(exists=True, file_okay=False, resolve_path=True),
    help="Path to the model directory containing artefacts and/or predict.py.",
)
@click.option(
    "--config",
    "config_path",
    required=True,
    type=click.Path(exists=True, dir_okay=False, resolve_path=True),
    help="Path to deploy_config.yaml.",
)
@click.option(
    "--output-dir",
    "output_dir",
    required=True,
    type=click.Path(resolve_path=True),
    help="Path to the output directory for generated serving code, Terraform, and summaries.",
)
@click.option(
    "--dry-run",
    is_flag=True,
    default=False,
    help="Run analysis and strategy only — skip actual deployment.",
)
@click.option(
    "--json-logs",
    is_flag=True,
    default=False,
    help="Emit structured JSON logs instead of coloured text.",
)
@click.option(
    "--backend",
    "-b",
    default=None,
    type=click.Choice(["auto", "fastapi", "triton", "seldon", "vertex_ai"],
                       case_sensitive=False),
    help="Force a specific backend (overrides config). auto = let AI decide.",
)
@click.option(
    "--verbose",
    "-v",
    is_flag=True,
    default=False,
    help="Enable verbose (DEBUG) logging.",
)
def main(
    model_dir: str,
    config_path: str,
    output_dir: str,
    dry_run: bool,
    json_logs: bool,
    backend: str | None,
    verbose: bool,
) -> None:
    """🚀 LaunchML — Analyze, reason, and deploy ML models with AI.

    Reads your model directory, uses an AI agent to pick the best serving
    backend, generates production-ready inference code and infrastructure,
    and deploys it — all in one command.

    \b
    Example:
        launch-ml --model-dir ./my_model \\
                  --config deploy_config.yaml \\
                  --output-dir ./output
    """

    # ── 0. Setup ──────────────────────────────────────────────────────────
    load_dotenv()  # load .env if present (never committed)

    import logging
    configure_logging(
        json_output=json_logs,
        level=logging.DEBUG if verbose else logging.INFO,
    )

    _print_banner()
    start_time = time.time()

    # ── 1. Load config ────────────────────────────────────────────────────
    step("Loading configuration")
    from launchml.core.config_loader import load_config, DeployConfig

    config = load_config(config_path)

    # Override model.path with --model-dir CLI argument
    config_dict = config.model_dump()
    config_dict["model"]["path"] = model_dir

    # CLI --backend flag overrides YAML config
    if backend and backend.lower() != "auto":
        config_dict["deployment"]["backend"] = backend.lower()
        info(f"Backend override (CLI): {backend}")

    # Rebuild the immutable config with overrides applied
    config = DeployConfig(**config_dict)

    # Validate model directory
    model_path = Path(model_dir)
    if not model_path.is_dir():
        cli_error(f"Model directory does not exist: {model_path}")
        raise SystemExit(1)

    deploy_mode = "LOCAL (Docker)" if config.deployment.is_local else f"CLOUD ({config.deployment.cloud})"
    backend_info = f"  |  Backend: {config.deployment.backend}" if config.deployment.has_backend_override else ""
    info(f"Model directory: {model_dir}")
    info(f"Output directory: {output_dir}")
    info(f"Deploy mode: {deploy_mode}  |  Region: {config.deployment.region}{backend_info}")
    info(f"LLM: {config.llm.provider}/{config.llm.model}")

    # ── 2. Run pipeline ───────────────────────────────────────────────────
    from launchml.core.agent_graph import run_pipeline

    try:
        final_state = run_pipeline(config, output_dir=output_dir)
    except KeyboardInterrupt:
        cli_error("Pipeline interrupted by user.")
        raise SystemExit(130)
    except Exception as exc:
        cli_error(f"Pipeline failed: {exc}")
        import traceback
        traceback.print_exc()
        raise SystemExit(1)

    # ── 3. Print results ──────────────────────────────────────────────────
    elapsed = time.time() - start_time
    _print_results(final_state, elapsed, dry_run, output_dir)


# ─── Pretty output helpers ────────────────────────────────────────────────────

def _print_banner() -> None:
    console.print(
        "\n[bold cyan]"
        "╔══════════════════════════════════════════════════════════════╗\n"
        "║                          LaunchML                            ║\n"
        "║              AI-Driven Model Deployment Pipeline             ║\n"
        "╚══════════════════════════════════════════════════════════════╝"
        "[/bold cyan]\n"
    )


def _print_results(state: dict, elapsed: float, dry_run: bool, output_dir: str) -> None:
    console.print("\n[bold]" + "═" * 62 + "[/bold]")
    console.print("[bold green]  ✅  Pipeline Complete[/bold green]")
    console.print("[bold]" + "═" * 62 + "[/bold]\n")

    # Strategy decision
    decision = state.get("strategy_decision", {})
    deploy = state.get("deployment_result", {})
    config_dict = state.get("config", {})
    is_local = config_dict.get("deployment", {}).get("cloud") == "local"

    deploy_mode = "🏠 Local (Docker)" if is_local else "☁️ Cloud"
    forced = config_dict.get("deployment", {}).get("backend")
    backend_label = decision.get('selected_backend', 'N/A')
    if forced:
        backend_label += "  [dim](user override)[/dim]"
    console.print(f"  [cyan]Mode:[/cyan]           {deploy_mode}")
    console.print(f"  [cyan]Backend:[/cyan]        {backend_label}")
    console.print(f"  [cyan]Instance:[/cyan]       {decision.get('instance_type', 'N/A')}")
    console.print(f"  [cyan]GPU:[/cyan]            {decision.get('gpu_type', 'none')}")
    console.print(f"  [cyan]Scaling:[/cyan]        {decision.get('scaling_strategy', 'N/A')}")
    console.print()

    if not dry_run:
        endpoint = deploy.get("endpoint_url", "N/A")
        monitoring = deploy.get("monitoring_url", "N/A")
        console.print(f"  [green]Endpoint:[/green]       {endpoint}")
        console.print(f"  [green]Monitoring:[/green]     {monitoring}")
        if is_local:
            stop_cmd = deploy.get("stop_command", "")
            if stop_cmd:
                console.print(f"  [yellow]Stop:[/yellow]           {stop_cmd}")
        # Display sample curl command
        sample_curl = deploy.get("sample_curl", "")
        if sample_curl:
            console.print()
            console.print("  [bold cyan]📋 Sample Request:[/bold cyan]")
            console.print()
            for line in sample_curl.split("\n"):
                console.print(f"    [dim]{line}[/dim]")
    else:
        console.print("  [yellow](dry-run mode — deployment skipped)[/yellow]")

    console.print()
    console.print(f"  [dim]Time elapsed: {elapsed:.1f}s[/dim]")
    console.print(f"  [dim]Summary: {output_dir}/DEPLOYMENT_SUMMARY.md[/dim]")
    console.print()


# ─── Entrypoint ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    main()
