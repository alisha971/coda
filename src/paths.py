"""One project root, one directory layout, on every runtime.

Notebooks run on Colab from a copy of this repository on Google Drive, and their
outputs are copied back into the repository by hand. That only works if the
layout is identical in both places, so no path in the project is written out
anywhere except here, and every path is relative to a single resolved root.

Root resolution, first match wins:

1. the ``CODA_ROOT`` environment variable;
2. on Colab, ``/content/drive/MyDrive/coda``;
3. otherwise the nearest ancestor of the working directory holding this
   project's ``pyproject.toml``.

Audio is the one exception to "everything under the root". It is streamed to the
runtime's local disk and deleted after measurement (process-and-discard), so it
never reaches Drive, whose free tier is smaller than the pilot's audio.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

COLAB_DRIVE_ROOT = Path("/content/drive/MyDrive/coda")
COLAB_SCRATCH = Path("/content/coda_scratch")


def _on_colab() -> bool:
    try:
        import google.colab  # type: ignore  # noqa: F401
    except ImportError:
        return False
    return True


def _is_project_root(path: Path) -> bool:
    toml = path / "pyproject.toml"
    return toml.is_file() and 'name = "coda"' in toml.read_text(encoding="utf-8")


def project_root() -> Path:
    """Resolve the project root. Raises if none can be found."""
    env = os.environ.get("CODA_ROOT")
    if env:
        return Path(env).resolve()
    if _on_colab():
        return COLAB_DRIVE_ROOT
    here = Path.cwd().resolve()
    for candidate in (here, *here.parents):
        if _is_project_root(candidate):
            return candidate
    raise FileNotFoundError(
        "Project root not found. Set CODA_ROOT, or run from inside the checkout."
    )


@dataclass(frozen=True)
class Paths:
    """Every directory the pipeline reads or writes.

    The git status of each tree follows ``.gitignore``: ``data/`` is ignored
    (corpus-derived and large), while ``manifests/``, ``results/`` and
    ``figures/`` are tracked evidence.
    """

    root: Path

    # -- tracked -----------------------------------------------------------
    @property
    def configs(self) -> Path:
        return self.root / "configs"

    @property
    def manifests(self) -> Path:
        return self.root / "manifests"

    @property
    def manifest(self) -> Path:
        return self.manifests / "coda_download_manifest.csv"

    @property
    def results(self) -> Path:
        return self.root / "results"

    @property
    def figures(self) -> Path:
        return self.root / "figures"

    # -- ignored: corpus-derived ------------------------------------------
    @property
    def data(self) -> Path:
        return self.root / "data"

    @property
    def raw_metadata(self) -> Path:
        return self.data / "raw" / "metadata"

    @property
    def raw_transcript(self) -> Path:
        return self.data / "raw" / "transcript"

    @property
    def raw_vad(self) -> Path:
        return self.data / "raw" / "vad"

    @property
    def processed(self) -> Path:
        return self.data / "processed"

    @property
    def audio_pass(self) -> Path:
        """Per-session outputs of the audio pass: tracks, QC rows, fetch log."""
        return self.processed / "audio_pass"

    # -- runtime-local, never synced ---------------------------------------
    @property
    def audio_scratch(self) -> Path:
        """Where audio lives for the seconds it takes to measure it."""
        if _on_colab():
            return COLAB_SCRATCH / "audio"
        return self.root / "tmp" / "audio"  # ignored by .gitignore (tmp/)

    def load_config(self, name: str) -> dict:
        """A tracked YAML configuration, e.g. ``load_config("corpus")``."""
        import yaml

        path = self.configs / (name if name.endswith(".yaml") else f"{name}.yaml")
        return yaml.safe_load(path.read_text(encoding="utf-8"))

    def ensure(self) -> Paths:
        for d in (
            self.manifests, self.results, self.figures, self.raw_metadata,
            self.raw_transcript, self.raw_vad, self.processed, self.audio_pass,
            self.audio_scratch,
        ):
            d.mkdir(parents=True, exist_ok=True)
        return self


def paths() -> Paths:
    """The layout rooted at :func:`project_root`."""
    return Paths(project_root())


__all__ = ["Paths", "paths", "project_root", "COLAB_DRIVE_ROOT", "COLAB_SCRATCH"]
