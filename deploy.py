#!/usr/bin/env python3
"""
LaunchML — deploy.py
=====================

Production-quality CLI entrypoint that:
  1. Loads and validates deploy_config.yaml
  2. Runs the LangGraph agent pipeline
  3. Outputs a working deployed endpoint + summary

Usage:
    python deploy.py --config deploy_config.yaml
    python deploy.py --config deploy_config.yaml --dry-run
    python deploy.py --config deploy_config.yaml --json-logs

Environment Variables:
    OPENAI_API_KEY      — Required if llm.provider = openai
    ANTHROPIC_API_KEY   — Required if llm.provider = anthropic
    GOOGLE_API_KEY      — Required if llm.provider = gemini

Architecture:
    This file is deliberately thin — it handles CLI parsing and
    delegates everything to ``core.agent_graph.run_pipeline()``.
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import click
from dotenv import load_dotenv

# Ensure project root is on sys.path for imports
PROJECT_ROOT = Path(__file__).parent.resolve()
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from utils.logger import (
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
    "--config",
    "config_path",
    required=True,
    type=click.Path(exists=True),
    help="Path to deploy_config.yaml",
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
    "--verbose",
    "-v",
    is_flag=True,
    default=False,
    help="Enable verbose (DEBUG) logging.",
)
def main(config_path: str, dry_run: bool, json_logs: bool, verbose: bool) -> None:
    """LaunchML — Analyze, reason, and deploy ML models with AI."""

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
    from core.config_loader import load_config
    config = load_config(config_path)
    deploy_mode = "LOCAL (Docker)" if config.deployment.is_local else f"CLOUD ({config.deployment.cloud})"
    info(f"Deploy mode: {deploy_mode}  |  Region: {config.deployment.region}")
    info(f"LLM: {config.llm.provider}/{config.llm.model}")

    # ── 2. Run pipeline ───────────────────────────────────────────────────
    from core.agent_graph import run_pipeline

    try:
        final_state = run_pipeline(config)
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
    _print_results(final_state, elapsed, dry_run)


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


def _print_results(state: dict, elapsed: float, dry_run: bool) -> None:
    console.print("\n[bold]" + "═" * 62 + "[/bold]")
    console.print("[bold green]  ✅  Pipeline Complete[/bold green]")
    console.print("[bold]" + "═" * 62 + "[/bold]\n")

    # Strategy decision
    decision = state.get("strategy_decision", {})
    deploy = state.get("deployment_result", {})
    config_dict = state.get("config", {})
    is_local = config_dict.get("deployment", {}).get("cloud") == "local"

    deploy_mode = "🏠 Local (Docker)" if is_local else "☁️ Cloud"
    console.print(f"  [cyan]Mode:[/cyan]           {deploy_mode}")
    console.print(f"  [cyan]Backend:[/cyan]        {decision.get('selected_backend', 'N/A')}")
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
    else:
        console.print("  [yellow](dry-run mode — deployment skipped)[/yellow]")

    console.print()
    console.print(f"  [dim]Time elapsed: {elapsed:.1f}s[/dim]")
    console.print(f"  [dim]Summary: generated/DEPLOYMENT_SUMMARY.md[/dim]")
    console.print()


# ─── Entrypoint ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    main()
