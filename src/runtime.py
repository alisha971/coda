"""Runtime inspection and provenance for notebooks run on Colab or locally.

Two jobs:

1. **A resource report** at the top of every notebook: disk, memory, CPU, GPU,
   and which artifacts already exist, so an expensive stage is loaded rather
   than recomputed and a session that is about to run out of disk says so first.
2. **A code fingerprint.** The Colab copy of the repository lives on Drive and
   has no git history, so the commit cannot identify the code that produced a
   result. The fingerprint is a SHA-256 over every source, configuration and
   requirements file. It goes into every run record, and after results are
   copied back it is checked against the committed tree.
"""

from __future__ import annotations

import hashlib
import json
import os
import platform
import shutil
import subprocess
import sys
from datetime import datetime, timezone
from importlib import metadata
from pathlib import Path
from typing import Any

from src.paths import Paths, _on_colab, paths

#: Files whose contents define the code that ran.
FINGERPRINT_GLOBS: tuple[str, ...] = (
    "src/**/*.py", "configs/*.yaml", "requirements-colab.txt",
)

#: Packages whose versions are recorded with every run.
RECORDED_PACKAGES: tuple[str, ...] = (
    "numpy", "pandas", "pyarrow", "scipy", "praat-parselmouth", "soundfile",
    "requests", "pyyaml", "matplotlib", "scikit-learn", "statsmodels",
    "torch", "transformers", "sentence-transformers",
)


def code_fingerprint(root: Path | None = None) -> str:
    """SHA-256 over the code and configuration, independent of file timestamps.

    Paths are hashed relative to the root with forward slashes and line endings
    normalised, so a Windows checkout and a Drive copy of the same tree agree.
    """
    root = root or paths().root
    files = sorted(
        {p for g in FINGERPRINT_GLOBS for p in root.glob(g) if p.is_file()
         and "__pycache__" not in p.parts}
    )
    h = hashlib.sha256()
    for f in files:
        rel = f.relative_to(root).as_posix()
        body = f.read_bytes().replace(b"\r\n", b"\n")
        h.update(rel.encode() + b"\0" + hashlib.sha256(body).digest())
    return h.hexdigest()[:16]


def package_versions() -> dict[str, str | None]:
    out: dict[str, str | None] = {}
    for name in RECORDED_PACKAGES:
        try:
            out[name] = metadata.version(name)
        except metadata.PackageNotFoundError:
            out[name] = None
    return out


def _memory_gb() -> tuple[float | None, float | None]:
    try:
        import psutil  # type: ignore

        vm = psutil.virtual_memory()
        return vm.total / 1e9, vm.available / 1e9
    except ImportError:
        return None, None


def _gpu() -> str | None:
    if shutil.which("nvidia-smi") is None:
        return None
    out = subprocess.run(
        ["nvidia-smi", "--query-gpu=name,memory.total", "--format=csv,noheader"],
        capture_output=True, text=True, check=False,
    )
    return out.stdout.strip() or None


def hardware() -> dict[str, Any]:
    p = paths()
    probe = p.root if p.root.exists() else Path.cwd()
    usage = shutil.disk_usage(probe)
    total, free = usage.total / 1e9, usage.free / 1e9
    scratch = p.audio_scratch
    scratch.mkdir(parents=True, exist_ok=True)
    s_usage = shutil.disk_usage(scratch)
    ram_total, ram_free = _memory_gb()
    return {
        "runtime": "colab" if _on_colab() else "local",
        "python": platform.python_version(),
        "platform": platform.platform(),
        "cpu_count": os.cpu_count(),
        "ram_total_gb": ram_total,
        "ram_available_gb": ram_free,
        "gpu": _gpu(),
        "project_disk_total_gb": total,
        "project_disk_free_gb": free,
        "scratch_disk_free_gb": s_usage.free / 1e9,
    }


def run_record(stage: str, extra: dict[str, Any] | None = None) -> dict[str, Any]:
    """Provenance written beside every artifact a notebook produces."""
    return {
        "stage": stage,
        "created_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "code_fingerprint": code_fingerprint(),
        "hardware": hardware(),
        "packages": package_versions(),
        "argv": sys.argv[:1],
        **(extra or {}),
    }


def write_run_record(path: Path, stage: str, extra: dict[str, Any] | None = None) -> dict:
    rec = run_record(stage, extra)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(rec, indent=2, default=str), encoding="utf-8")
    return rec


def artifact_status(p: Paths | None = None) -> dict[str, bool]:
    """Which pipeline artifacts already exist under the project root."""
    p = p or paths()
    return {
        "manifest": p.manifest.exists(),
        "metadata": (p.raw_metadata / "filelist.csv").exists(),
        "transcripts": p.raw_transcript.exists() and any(p.raw_transcript.glob("*.jsonl")),
        "turns": (p.processed / "turns.parquet").exists(),
        "audio_pass_sessions": len(list(p.audio_pass.glob("*/DONE.json")))
        if p.audio_pass.exists() else 0,
        "channel_qc": (p.processed / "channel_qc.parquet").exists(),
    }


def runtime_report(min_free_gb: float = 5.0) -> dict[str, Any]:
    """Print the resource and artifact report that opens every notebook."""
    hw = hardware()
    arts = artifact_status()
    print(f"runtime        {hw['runtime']}   python {hw['python']}   cpus {hw['cpu_count']}")
    if hw["ram_total_gb"]:
        print(f"memory         {hw['ram_available_gb']:.1f} GB free of {hw['ram_total_gb']:.1f} GB")
    print(f"gpu            {hw['gpu'] or 'none (not needed for this stage unless stated)'}")
    print(f"project disk   {hw['project_disk_free_gb']:.1f} GB free   root = {paths().root}")
    print(f"scratch disk   {hw['scratch_disk_free_gb']:.1f} GB free   (audio lives here briefly)")
    print(f"code           fingerprint {code_fingerprint()}")
    print("artifacts      " + ", ".join(f"{k}={v}" for k, v in arts.items()))
    if hw["scratch_disk_free_gb"] < min_free_gb:
        print(f"WARNING: less than {min_free_gb} GB free on the scratch disk.")
    return {"hardware": hw, "artifacts": arts}


__all__ = [
    "code_fingerprint", "package_versions", "hardware", "run_record",
    "write_run_record", "artifact_status", "runtime_report",
]
