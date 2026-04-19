"""Public release package for SAM2Galaxy GNN inference."""

from .artifacts import ReleaseArtifacts, load_release_artifacts

__all__ = [
    "ReleaseArtifacts",
    "load_release_artifacts",
]

__version__ = "0.1.2"
