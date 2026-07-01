#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from garch_utils import load_config, read_train_csv  # noqa: E402


def read_pair_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        fieldnames = set(reader.fieldnames or [])
        missing = {"sp500", "DGS10"} - fieldnames
        if missing:
            raise ValueError(f"Missing columns in {path}: {sorted(missing)}")
        return [{"sp500": row["sp500"], "DGS10": row["DGS10"]} for row in reader]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="garch_origin/config/garch11_correlogram_design.json")
    parser.add_argument(
        "--generated-dir",
        default="garch_origin/runs/garch11_correlogram_design/generated",
        type=Path,
    )
    parser.add_argument("--output-dir", default="garch_origin/my_mixed_data", type=Path)
    parser.add_argument("--output-name", default="mixed_data_garch.csv")
    parser.add_argument("--length", type=int, default=1260)
    parser.add_argument("--num-generated", type=int, default=10)
    parser.add_argument("--real-start", type=int, default=0)
    args = parser.parse_args()

    if args.length <= 0:
        raise ValueError("--length must be positive")
    if args.num_generated <= 0:
        raise ValueError("--num-generated must be positive")
    if args.real_start < 0:
        raise ValueError("--real-start must be non-negative")

    config = load_config(args.config)
    features = list(config.get("features", ["sp500", "DGS10"]))
    if features != ["sp500", "DGS10"]:
        raise ValueError(f"This mixed-data writer expects ['sp500', 'DGS10'], got {features}")

    generated_paths = sorted(args.generated_dir.glob("garch_generated_*.csv"))[: args.num_generated]
    if len(generated_paths) != args.num_generated:
        raise FileNotFoundError(
            f"Expected {args.num_generated} generated files in {args.generated_dir}, found {len(generated_paths)}"
        )

    series: list[list[dict[str, str]]] = []
    for path in generated_paths:
        rows = read_pair_csv(path)
        if len(rows) < args.length:
            raise ValueError(f"Too few rows in {path}: {len(rows)}")
        series.append(rows[: args.length])

    train = read_train_csv(config["train_csv"], features)
    real_end = args.real_start + args.length
    if real_end > len(train):
        raise ValueError(
            f"Requested real window [{args.real_start}, {real_end}) but train has {len(train)} rows"
        )
    real_rows = [
        {"sp500": f"{float(row.sp500):.10g}", "DGS10": f"{float(row.DGS10):.10g}"}
        for row in train.iloc[args.real_start:real_end].itertuples(index=False)
    ]
    series.append(real_rows)

    fieldnames: list[str] = []
    for idx in range(1, len(series) + 1):
        fieldnames.extend([f"mask{idx}_sp500", f"mask{idx}_DGS10"])

    args.output_dir.mkdir(parents=True, exist_ok=True)
    output_path = args.output_dir / args.output_name
    with output_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row_idx in range(args.length):
            out_row: dict[str, str] = {}
            for mask_idx, rows in enumerate(series, start=1):
                out_row[f"mask{mask_idx}_sp500"] = rows[row_idx]["sp500"]
                out_row[f"mask{mask_idx}_DGS10"] = rows[row_idx]["DGS10"]
            writer.writerow(out_row)

    print(f"Wrote {output_path}")
    print(f"generated masks: mask1-mask{args.num_generated}")
    print(f"real mask: mask{args.num_generated + 1}")
    print(f"real source: {config['train_csv']} rows {args.real_start}-{real_end - 1}")


if __name__ == "__main__":
    main()
