"""
LaunchML Model Analyzer — scans a local model directory and produces structured metadata.

This is Node ①  in the LangGraph pipeline.

Detection heuristics:
    1. File-extension scan (`.pt`, `.onnx`, `.pb`, `.safetensors`, etc.)
    2. Presence of HuggingFace artefacts (``config.json``, ``tokenizer.json``)
    3. Directory structure conventions (``saved_model/``, ``variables/``)
    4. Model size estimation (sum of artefact file sizes)
    5. GPU need heuristic based on size + framework
    6. Entrypoint import scanning — detects framework when only a
       ``predict.py`` is present (e.g. models fetched from
       torchvision / HuggingFace Hub at runtime).

The output is a ``ModelMetadata`` dataclass that flows through the rest
of the pipeline.
"""

from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

from utils.logger import get_logger

log = get_logger(__name__)

# ─── Constants ────────────────────────────────────────────────────────────────

_FRAMEWORK_EXTENSIONS: dict[str, list[str]] = {
    "pytorch": [".pt", ".pth", ".bin"],
    "tensorflow": [".pb", ".h5", ".keras"],
    "onnx": [".onnx"],
    "transformers": [".safetensors"],  # also detected via config.json
}

_HF_MARKER_FILES = {"config.json", "tokenizer.json", "tokenizer_config.json", "special_tokens_map.json"}

# Patterns matched against import lines in a custom entrypoint.
# Order matters: later entries can override earlier ones when both match.
_ENTRYPOINT_IMPORT_PATTERNS: list[tuple[str, str]] = [
    (r"\btorch\b", "pytorch"),
    (r"\btorchvision\b", "pytorch"),
    (r"\btensorflow\b", "tensorflow"),
    (r"\bkeras\b", "tensorflow"),
    (r"\bonnxruntime\b", "onnx"),
    (r"\btransformers\b", "transformers"),
    (r"\bhuggingface_hub\b", "transformers"),
    (r"\bdiffusers\b", "transformers"),
]

# Mapping from Python import names to pip package names.
# Used to auto-generate requirements.txt from the user's entrypoint.
_IMPORT_TO_PIP: dict[str, str] = {
    "torch": "torch",
    "torchvision": "torchvision",
    "torchaudio": "torchaudio",
    "tensorflow": "tensorflow",
    "keras": "keras",
    "onnxruntime": "onnxruntime",
    "PIL": "Pillow",
    "cv2": "opencv-python",
    "sklearn": "scikit-learn",
    "skimage": "scikit-image",
    "yaml": "PyYAML",
    "requests": "requests",
    "numpy": "numpy",
    "pandas": "pandas",
    "scipy": "scipy",
    "transformers": "transformers",
    "huggingface_hub": "huggingface-hub",
    "diffusers": "diffusers",
    "accelerate": "accelerate",
    "safetensors": "safetensors",
    "tokenizers": "tokenizers",
    "datasets": "datasets",
    "einops": "einops",
}

_SIZE_THRESHOLDS_MB = {
    "small": 100,      # < 100 MB
    "medium": 1_000,   # < 1 GB
    "large": 5_000,    # < 5 GB
    "xlarge": float("inf"),
}


# ─── Output dataclass ────────────────────────────────────────────────────────

@dataclass
class ModelMetadata:
    """Structured metadata produced by the analyzer."""
    model_path: str
    detected_framework: str                     # pytorch | tensorflow | onnx | transformers | unknown
    model_files: List[str] = field(default_factory=list)
    total_size_mb: float = 0.0
    size_category: str = "unknown"              # small | medium | large | xlarge
    has_tokenizer: bool = False
    has_config_json: bool = False
    has_custom_entrypoint: bool = False
    entrypoint_path: Optional[str] = None
    runtime_download: bool = False              # True when weights are fetched at startup (no local artefacts)
    entrypoint_dependencies: List[str] = field(default_factory=list)  # pip package names
    gpu_recommendation: str = "auto"            # true | false | auto
    extra: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Serialise to a plain dict (JSON-safe)."""
        return {
            "model_path": self.model_path,
            "detected_framework": self.detected_framework,
            "model_files": self.model_files,
            "total_size_mb": round(self.total_size_mb, 2),
            "size_category": self.size_category,
            "has_tokenizer": self.has_tokenizer,
            "has_config_json": self.has_config_json,
            "has_custom_entrypoint": self.has_custom_entrypoint,
            "entrypoint_path": self.entrypoint_path,
            "runtime_download": self.runtime_download,
            "entrypoint_dependencies": self.entrypoint_dependencies,
            "gpu_recommendation": self.gpu_recommendation,
            "extra": self.extra,
        }


# ─── Analyzer ─────────────────────────────────────────────────────────────────

class ModelAnalyzer:
    """Scan a model directory and return ``ModelMetadata``."""

    def __init__(self, model_path: str, *, framework_hint: Optional[str] = None,
                 entrypoint: Optional[str] = None) -> None:
        self.model_path = Path(model_path).expanduser().resolve()
        self.framework_hint = framework_hint
        self.entrypoint = entrypoint

    # ------------------------------------------------------------------
    # Public
    # ------------------------------------------------------------------

    def analyze(self) -> ModelMetadata:
        """Run full analysis and return metadata."""
        log.info("model_analysis_start", path=str(self.model_path))

        all_files = self._walk_files()
        framework = self._detect_framework(all_files)
        model_files = self._filter_model_files(all_files, framework)
        total_mb = self._total_size_mb(all_files)
        size_cat = self._size_category(total_mb)
        has_tok = self._has_tokenizer(all_files)
        has_cfg = self._has_config_json(all_files)
        has_entry, entry_path = self._check_entrypoint()

        # ── Runtime-download detection ────────────────────────────────
        # If we have a custom entrypoint but *no* model weight files,
        # the user likely downloads from torchvision / HuggingFace Hub
        # / a remote registry at startup time.
        is_runtime_download = has_entry and len(model_files) == 0

        # When no model files exist, try to infer the framework by
        # scanning import statements in the entrypoint.
        if framework == "unknown" and entry_path:
            scanned_fw = self._detect_framework_from_entrypoint(entry_path)
            if scanned_fw != "unknown":
                log.info(
                    "framework_from_entrypoint",
                    framework=scanned_fw,
                    entrypoint=entry_path,
                )
                framework = scanned_fw

        gpu_rec = self._gpu_heuristic(
            framework, total_mb, runtime_download=is_runtime_download,
        )

        # Scan entrypoint for pip dependencies
        ep_deps: List[str] = []
        if entry_path:
            ep_deps = self._scan_entrypoint_dependencies(entry_path)
            if ep_deps:
                log.info("entrypoint_deps", deps=ep_deps, entrypoint=entry_path)

        extra: Dict[str, Any] = {}
        if has_cfg:
            extra["config_json"] = self._read_config_json()
        if is_runtime_download:
            extra["note"] = (
                "No model weight files found. The custom entrypoint is "
                "expected to download/initialise the model at startup "
                "(e.g. from torchvision, HuggingFace Hub, etc.)."
            )

        meta = ModelMetadata(
            model_path=str(self.model_path),
            detected_framework=framework,
            model_files=model_files,
            total_size_mb=total_mb,
            size_category=size_cat,
            has_tokenizer=has_tok,
            has_config_json=has_cfg,
            has_custom_entrypoint=has_entry,
            entrypoint_path=entry_path,
            runtime_download=is_runtime_download,
            entrypoint_dependencies=ep_deps,
            gpu_recommendation=gpu_rec,
            extra=extra,
        )
        log.info(
            "model_analysis_complete",
            framework=framework,
            size_mb=round(total_mb, 1),
            gpu=gpu_rec,
            runtime_download=is_runtime_download,
        )
        return meta

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _walk_files(self) -> List[Path]:
        """Recursively list all files under the model path."""
        if not self.model_path.is_dir():
            log.warning("model_path_not_dir", path=str(self.model_path))
            return [self.model_path] if self.model_path.is_file() else []
        return [p for p in self.model_path.rglob("*") if p.is_file()]

    def _detect_framework(self, files: List[Path]) -> str:
        """Detect ML framework from file extensions and markers."""
        # If user gave a hint, trust it (but validate later)
        if self.framework_hint:
            return self.framework_hint

        extension_votes: dict[str, int] = {}
        for f in files:
            for fw, exts in _FRAMEWORK_EXTENSIONS.items():
                if f.suffix.lower() in exts:
                    extension_votes[fw] = extension_votes.get(fw, 0) + 1

        # HuggingFace markers bump transformers vote
        names = {f.name for f in files}
        if names & _HF_MARKER_FILES:
            extension_votes["transformers"] = extension_votes.get("transformers", 0) + 5

        if not extension_votes:
            return "unknown"

        winner = max(extension_votes, key=lambda k: extension_votes[k])

        # If both transformers artefacts and .bin files exist, prefer transformers
        if "transformers" in extension_votes and "pytorch" in extension_votes:
            winner = "transformers"

        return winner

    def _filter_model_files(self, files: List[Path], framework: str) -> List[str]:
        """Return relative paths of artefact files for the detected framework."""
        exts = set()
        for fw_exts in _FRAMEWORK_EXTENSIONS.values():
            exts.update(fw_exts)
        return [
            str(f.relative_to(self.model_path))
            for f in files
            if f.suffix.lower() in exts
        ]

    def _total_size_mb(self, files: List[Path]) -> float:
        total_bytes = sum(f.stat().st_size for f in files if f.is_file())
        return total_bytes / (1024 * 1024)

    @staticmethod
    def _size_category(mb: float) -> str:
        for cat, threshold in _SIZE_THRESHOLDS_MB.items():
            if mb < threshold:
                return cat
        return "xlarge"

    def _has_tokenizer(self, files: List[Path]) -> bool:
        names = {f.name for f in files}
        return bool(names & {"tokenizer.json", "tokenizer_config.json", "vocab.txt"})

    def _has_config_json(self, files: List[Path]) -> bool:
        return any(f.name == "config.json" for f in files)

    def _check_entrypoint(self) -> tuple[bool, Optional[str]]:
        if not self.entrypoint:
            return False, None
        ep = self.model_path / self.entrypoint
        if ep.is_file():
            return True, str(ep)
        # Also check relative to CWD
        ep2 = Path(self.entrypoint)
        if ep2.is_file():
            return True, str(ep2.resolve())
        return False, None

    def _detect_framework_from_entrypoint(self, entrypoint_path: str) -> str:
        """Scan import lines in the entrypoint to infer the ML framework.

        This handles the common case where no model weight files exist
        because the user downloads from torchvision, HuggingFace Hub, etc.
        """
        try:
            source = Path(entrypoint_path).read_text(encoding="utf-8", errors="ignore")
        except OSError:
            return "unknown"

        # Collect only import lines (handles both ``import X`` and ``from X import ...``)
        import_lines = [
            line for line in source.splitlines()
            if re.match(r"^\s*(import |from )", line)
        ]
        if not import_lines:
            return "unknown"

        import_block = "\n".join(import_lines)
        votes: dict[str, int] = {}
        for pattern, fw in _ENTRYPOINT_IMPORT_PATTERNS:
            if re.search(pattern, import_block):
                votes[fw] = votes.get(fw, 0) + 1

        if not votes:
            return "unknown"

        # transformers > pytorch when both present (HuggingFace wraps torch)
        if "transformers" in votes and "pytorch" in votes:
            return "transformers"
        return max(votes, key=lambda k: votes[k])

    def _gpu_heuristic(
        self, framework: str, size_mb: float, *, runtime_download: bool = False,
    ) -> str:
        """Heuristic for GPU need.

        When ``runtime_download`` is True the weights aren't on disk so
        we can't judge by file size — default to ``"auto"`` and let the
        LLM / user decide.
        """
        if runtime_download:
            # Can't judge size; GPU is likely but not certain.
            return "auto"
        if size_mb > 500:
            return "true"
        if framework in ("transformers",) and size_mb > 100:
            return "true"
        if size_mb < 50:
            return "false"
        return "auto"

    def _scan_entrypoint_dependencies(self, entrypoint_path: str) -> List[str]:
        """Extract pip package names from the import statements of the entrypoint.

        Only packages present in ``_IMPORT_TO_PIP`` are returned — standard-library
        modules (``sys``, ``os``, ``io``, …) are silently ignored.
        """
        try:
            source = Path(entrypoint_path).read_text(encoding="utf-8", errors="ignore")
        except OSError:
            return []

        pip_pkgs: set[str] = set()
        for line in source.splitlines():
            stripped = line.strip()
            if stripped.startswith("import "):
                # import torch, numpy  →  ["torch", "numpy"]
                modules = stripped[len("import "):].split(",")
                for mod in modules:
                    top = mod.strip().split(".")[0].split(" ")[0]
                    if top in _IMPORT_TO_PIP:
                        pip_pkgs.add(_IMPORT_TO_PIP[top])
            elif stripped.startswith("from "):
                # from torchvision import models  →  "torchvision"
                parts = stripped[len("from "):].split()
                if parts:
                    top = parts[0].split(".")[0]
                    if top in _IMPORT_TO_PIP:
                        pip_pkgs.add(_IMPORT_TO_PIP[top])
        return sorted(pip_pkgs)

    def _read_config_json(self) -> Dict[str, Any]:
        """Best-effort read of HuggingFace config.json."""
        cfg_path = self.model_path / "config.json"
        if not cfg_path.is_file():
            return {}
        try:
            with open(cfg_path) as fh:
                return json.load(fh)
        except Exception:
            return {}
