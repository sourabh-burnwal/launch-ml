"""
LaunchML Terraform Runner — safe subprocess wrapper for Terraform CLI operations.

Architecture:
    - Wraps `terraform init`, `plan`, `apply`, and `destroy` behind a
      clean Python API.
    - All commands run in the directory containing the generated `.tf` files.
    - stdout / stderr are captured and streamed to the structured logger.
    - For the POC, if Terraform is not installed the runner enters
      *simulation mode* and returns mock outputs so the rest of the pipeline
      can still be exercised end-to-end.
    - No credentials are ever written to disk; Terraform is expected to
      pick them up from environment variables (e.g. GOOGLE_APPLICATION_CREDENTIALS).
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

from launchml.utils.logger import get_logger, console

log = get_logger(__name__)


@dataclass
class TerraformResult:
    """Captures the result of a Terraform command."""
    command: str
    return_code: int
    stdout: str
    stderr: str
    outputs: dict[str, Any] = field(default_factory=dict)
    simulated: bool = False


class TerraformRunner:
    """Execute Terraform commands against a working directory.

    Args:
        working_dir: Path containing ``*.tf`` files.
        auto_approve: Skip interactive approval (``-auto-approve``).
        simulate: Force simulation mode even if Terraform is installed.
    """

    def __init__(
        self,
        working_dir: str | Path,
        *,
        auto_approve: bool = True,
        simulate: bool = False,
    ) -> None:
        self.working_dir = Path(working_dir).resolve()
        self.auto_approve = auto_approve
        self._simulate = simulate or not self._terraform_available()

        if self._simulate:
            log.warning(
                "terraform_not_found_or_simulation_mode",
                msg="Terraform CLI not found or simulation forced — running in mock mode.",
            )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def init(self) -> TerraformResult:
        """Run ``terraform init``."""
        return self._run(["init", "-input=false", "-no-color"])

    def plan(self) -> TerraformResult:
        """Run ``terraform plan``."""
        return self._run(["plan", "-input=false", "-no-color"])

    def apply(self) -> TerraformResult:
        """Run ``terraform apply``."""
        cmd = ["apply", "-input=false", "-no-color"]
        if self.auto_approve:
            cmd.append("-auto-approve")
        return self._run(cmd)

    def destroy(self) -> TerraformResult:
        """Run ``terraform destroy``."""
        cmd = ["destroy", "-input=false", "-no-color"]
        if self.auto_approve:
            cmd.append("-auto-approve")
        return self._run(cmd)

    def output(self) -> dict[str, Any]:
        """Run ``terraform output -json`` and return parsed dict."""
        result = self._run(["output", "-json", "-no-color"])
        if result.simulated:
            return result.outputs
        try:
            raw = json.loads(result.stdout)
            return {k: v.get("value", v) for k, v in raw.items()}
        except (json.JSONDecodeError, AttributeError):
            return {}

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _terraform_available() -> bool:
        return shutil.which("terraform") is not None

    def _run(self, args: list[str]) -> TerraformResult:
        full_cmd = " ".join(["terraform", *args])
        log.info("terraform_cmd", cmd=full_cmd, cwd=str(self.working_dir))

        if self._simulate:
            return self._simulate_run(full_cmd)

        try:
            proc = subprocess.run(
                ["terraform", *args],
                cwd=self.working_dir,
                capture_output=True,
                text=True,
                timeout=600,
                env={**os.environ},  # inherit env (credentials, etc.)
            )
            result = TerraformResult(
                command=full_cmd,
                return_code=proc.returncode,
                stdout=proc.stdout,
                stderr=proc.stderr,
            )
            if proc.returncode != 0:
                log.error("terraform_failed", cmd=full_cmd, stderr=proc.stderr[:500])
            else:
                log.info("terraform_success", cmd=full_cmd)
            return result
        except FileNotFoundError:
            log.error("terraform_not_found")
            return self._simulate_run(full_cmd)
        except subprocess.TimeoutExpired:
            log.error("terraform_timeout", cmd=full_cmd)
            return TerraformResult(
                command=full_cmd, return_code=-1, stdout="", stderr="Timeout"
            )

    def _simulate_run(self, cmd: str) -> TerraformResult:
        """Return a mock result for demo / CI environments."""
        console.print(f"  [dim](simulated)[/dim] {cmd}")
        mock_outputs: dict[str, Any] = {}
        if "apply" in cmd or "output" in cmd:
            mock_outputs = {
                "endpoint_url": "https://launchml-demo.example.com/predict",
                "monitoring_url": "https://launchml-demo.example.com/metrics",
            }
        return TerraformResult(
            command=cmd,
            return_code=0,
            stdout=json.dumps(mock_outputs),
            stderr="",
            outputs=mock_outputs,
            simulated=True,
        )
