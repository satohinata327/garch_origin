#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import shutil
import subprocess
from pathlib import Path

from garch_utils import load_config


def copy_generated_as_mask(generated_dir: Path, mask_dir: Path) -> list[Path]:
    mask_dir.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []
    for idx, path in enumerate(sorted(generated_dir.glob("*.csv")), start=1):
        output_path = mask_dir / f"mask{idx}_garch.csv"
        shutil.copy2(path, output_path)
        written.append(output_path)
    if not written:
        raise FileNotFoundError(f"No generated CSV files found in {generated_dir}")
    return written


def relabel_generator_column(path: Path, generator: str) -> None:
    if not path.exists():
        return
    with path.open(newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        if reader.fieldnames is None or "generator" not in reader.fieldnames:
            return
        rows = list(reader)
        fields = reader.fieldnames
    for row in rows:
        row["generator"] = generator
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def relabel_outputs(output_dir: Path, generator: str) -> None:
    for path in [
        output_dir / "features" / "each_mask_features.csv",
        output_dir / "results" / "feature_zscores.csv",
        output_dir / "results" / "mahalanobis_distances.csv",
        output_dir / "results" / "mask_distance_positions.csv",
    ]:
        relabel_generator_column(path, generator)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="garch_origin/config/garch11_baseline.json")
    parser.add_argument("--generated-dir", default=None)
    parser.add_argument("--work-mask-dir", default=None)
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--mahalanobis-script", default="garch_origin/mahalanobis_eval/scripts/run_mahalanobis_eval.py")
    parser.add_argument("--train-csv", default=None)
    args = parser.parse_args()

    config = load_config(args.config)
    run_dir = Path(config.get("output_dir", "garch_origin/runs/garch11_baseline"))
    generated_dir = Path(args.generated_dir) if args.generated_dir else run_dir / "generated"
    mask_dir = Path(args.work_mask_dir) if args.work_mask_dir else run_dir / "evaluation" / "mahalanobis_input"
    output_dir = Path(args.output_dir) if args.output_dir else run_dir / "evaluation" / "mahalanobis_results"
    train_csv = args.train_csv if args.train_csv else config["train_csv"]
    mahalanobis_config = config.get("mahalanobis", {})

    paths = copy_generated_as_mask(generated_dir, mask_dir)
    print(f"Prepared {len(paths)} generated files for Mahalanobis evaluation")

    cmd = [
        "python3",
        args.mahalanobis_script,
        "--train-csv",
        train_csv,
        "--mask-dir",
        str(mask_dir),
        "--output-dir",
        str(output_dir),
        "--window-length",
        str(mahalanobis_config.get("window_length", 1260)),
        "--stride",
        str(mahalanobis_config.get("stride", 126)),
    ]
    print("Running:", " ".join(cmd))
    subprocess.run(cmd, check=True)
    relabel_outputs(output_dir, "garch")
    summary_path = output_dir / "results" / "summary.txt"
    if summary_path.exists():
        summary = summary_path.read_text(encoding="utf-8")
        summary = summary.replace(
            "# TimeGAN Mahalanobis evaluation result",
            "# GARCH Mahalanobis evaluation result",
            1,
        )
        summary = summary.replace(",unknown,", ",garch,")
        summary_path.write_text(summary, encoding="utf-8")
    print(f"Saved evaluation results to {output_dir}")


if __name__ == "__main__":
    main()
