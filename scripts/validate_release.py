#!/usr/bin/env python3

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from sam2galaxy_gnn.artifacts import load_release_artifacts, validate_artifacts


def main() -> int:
    artifacts = load_release_artifacts()
    missing = validate_artifacts(artifacts)
    if missing:
        print("Missing release artifacts:")
        for item in missing:
            print(f"- {item}")
        return 1
    print("Release scaffold validation passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
