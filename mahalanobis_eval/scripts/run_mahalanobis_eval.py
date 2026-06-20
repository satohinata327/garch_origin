#!/usr/bin/env python3
from __future__ import annotations

import runpy
from pathlib import Path


def main() -> None:
    repo_root = Path(__file__).resolve().parents[3]
    evaluator = repo_root / "timegan_remake1" / "mahalanobis_eval" / "scripts" / "run_mahalanobis_eval.py"
    if not evaluator.exists():
        raise FileNotFoundError(f"Mahalanobis evaluator not found: {evaluator}")
    namespace = runpy.run_path(str(evaluator))
    namespace["main"]()


if __name__ == "__main__":
    main()
