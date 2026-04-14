from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any
import json


@dataclass(frozen=True)
class ReleaseConfig:
    root_dir: Path
    manifest_path: Path
    manifest_dir: Path
    manifest: dict[str, Any]


def package_root() -> Path:
    return Path(__file__).resolve().parents[2]


def default_manifest_path() -> Path:
    return package_root() / "configs" / "release_manifest.json"


def load_release_config(manifest_path: str | Path | None = None) -> ReleaseConfig:
    path = Path(manifest_path) if manifest_path is not None else default_manifest_path()
    with path.open("r", encoding="utf-8") as fh:
        manifest = json.load(fh)
    return ReleaseConfig(
        root_dir=package_root(),
        manifest_path=path,
        manifest_dir=path.parent.resolve(),
        manifest=manifest,
    )
