#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import html
import math
import sys
from pathlib import Path

import numpy as np

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from garch_utils import load_config, read_train_csv  # noqa: E402


def autocorrelation(values: np.ndarray, max_lag: int) -> list[float]:
    xs = np.asarray(values, dtype=np.float64)
    xs = xs[np.isfinite(xs)]
    centered = xs - float(np.mean(xs))
    denom = float(np.dot(centered, centered))
    if denom <= 0.0:
        return [math.nan for _ in range(max_lag + 1)]

    acf = [1.0]
    for lag in range(1, max_lag + 1):
        if lag >= len(centered):
            acf.append(math.nan)
            continue
        numerator = float(np.dot(centered[:-lag], centered[lag:]))
        acf.append(numerator / denom)
    return acf


def write_acf_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = ["feature", "series", "lag", "acf"]
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def fmt(value: float) -> str:
    if not math.isfinite(value):
        return "nan"
    return f"{value:.6f}"


def svg_escape(text: object) -> str:
    return html.escape(str(text), quote=True)


def save_correlogram_svg(
    path: Path,
    acf_by_panel: dict[tuple[str, str], list[float]],
    features: list[str],
    series_names: list[str],
    max_lag: int,
    n_rows: int,
) -> None:
    panel_w = 470
    panel_h = 250
    margin_left = 58
    margin_right = 18
    margin_top = 42
    margin_bottom = 42
    gap_x = 36
    gap_y = 38
    title_h = 56
    width = margin_left + len(features) * panel_w + (len(features) - 1) * gap_x + margin_right
    height = title_h + len(series_names) * panel_h + (len(series_names) - 1) * gap_y + 26
    conf = 1.96 / math.sqrt(max(n_rows, 1))

    def panel_origin(row_idx: int, col_idx: int) -> tuple[float, float]:
        x = margin_left + col_idx * (panel_w + gap_x)
        y = title_h + row_idx * (panel_h + gap_y)
        return float(x), float(y)

    lines = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="#ffffff"/>',
        f'<text x="{width / 2:.1f}" y="30" text-anchor="middle" font-size="22" font-family="Arial" fill="#222">Train data correlogram</text>',
        f'<text x="{width / 2:.1f}" y="50" text-anchor="middle" font-size="12" font-family="Arial" fill="#666">ACF with approx. 95% confidence band +/- {conf:.4f}</text>',
    ]

    for row_idx, series_name in enumerate(series_names):
        for col_idx, feature in enumerate(features):
            x0, y0 = panel_origin(row_idx, col_idx)
            plot_w = panel_w - 76
            plot_h = panel_h - 74
            px0 = x0 + 44
            py0 = y0 + 30
            zero_y = py0 + plot_h / 2.0
            scale_y = plot_h / 2.0
            acf = acf_by_panel[(feature, series_name)]

            lines.extend(
                [
                    f'<text x="{x0 + panel_w / 2:.1f}" y="{y0 + 18:.1f}" text-anchor="middle" font-size="14" font-family="Arial" fill="#222">{svg_escape(feature)} - {svg_escape(series_name)}</text>',
                    f'<line x1="{px0:.1f}" y1="{zero_y:.1f}" x2="{px0 + plot_w:.1f}" y2="{zero_y:.1f}" stroke="#777" stroke-width="1"/>',
                    f'<line x1="{px0:.1f}" y1="{py0:.1f}" x2="{px0:.1f}" y2="{py0 + plot_h:.1f}" stroke="#333" stroke-width="1"/>',
                    f'<line x1="{px0:.1f}" y1="{py0 + plot_h:.1f}" x2="{px0 + plot_w:.1f}" y2="{py0 + plot_h:.1f}" stroke="#333" stroke-width="1"/>',
                    f'<rect x="{px0:.1f}" y="{zero_y - conf * scale_y:.1f}" width="{plot_w:.1f}" height="{2 * conf * scale_y:.1f}" fill="#d9e8ff" opacity="0.8"/>',
                    f'<text x="{px0 - 8:.1f}" y="{py0 + 4:.1f}" text-anchor="end" font-size="10" font-family="Arial" fill="#555">1</text>',
                    f'<text x="{px0 - 8:.1f}" y="{zero_y + 4:.1f}" text-anchor="end" font-size="10" font-family="Arial" fill="#555">0</text>',
                    f'<text x="{px0 - 8:.1f}" y="{py0 + plot_h + 4:.1f}" text-anchor="end" font-size="10" font-family="Arial" fill="#555">-1</text>',
                    f'<text x="{px0:.1f}" y="{py0 + plot_h + 22:.1f}" text-anchor="middle" font-size="10" font-family="Arial" fill="#555">0</text>',
                    f'<text x="{px0 + plot_w:.1f}" y="{py0 + plot_h + 22:.1f}" text-anchor="middle" font-size="10" font-family="Arial" fill="#555">{max_lag}</text>',
                ]
            )

            for lag, value in enumerate(acf):
                if not math.isfinite(value):
                    continue
                x = px0 + plot_w * lag / max(max_lag, 1)
                y = zero_y - max(-1.0, min(1.0, value)) * scale_y
                color = "#1f77b4" if lag > 0 else "#333333"
                width_attr = "2" if lag > 0 else "3"
                lines.append(
                    f'<line x1="{x:.2f}" y1="{zero_y:.2f}" x2="{x:.2f}" y2="{y:.2f}" stroke="{color}" stroke-width="{width_attr}"/>'
                )
                if lag > 0:
                    lines.append(f'<circle cx="{x:.2f}" cy="{y:.2f}" r="2.2" fill="{color}"/>')

            selected = [1, 5, 10, 20, 40, max_lag]
            label_parts = []
            for lag in selected:
                if lag <= max_lag:
                    label_parts.append(f"lag{lag}={fmt(acf[lag])}")
            lines.append(
                f'<text x="{px0 + plot_w / 2:.1f}" y="{y0 + panel_h - 8:.1f}" text-anchor="middle" font-size="10" font-family="Arial" fill="#555">{svg_escape(", ".join(label_parts))}</text>'
            )

    lines.append("</svg>")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="garch_origin/config/garch11_baseline.json")
    parser.add_argument("--max-lag", type=int, default=60)
    parser.add_argument("--output-dir", default=None)
    args = parser.parse_args()

    if args.max_lag < 1:
        raise ValueError("--max-lag must be at least 1")

    config = load_config(args.config)
    features = list(config.get("features", ["sp500", "DGS10"]))
    output_root = Path(config.get("output_dir", "garch_origin/runs/garch11_baseline"))
    output_dir = Path(args.output_dir) if args.output_dir else output_root / "diagnostics" / "correlogram"
    train = read_train_csv(config["train_csv"], features)

    series_builders = {
        "return": lambda values: values,
        "absolute_return": np.abs,
        "squared_return": np.square,
    }

    rows: list[dict[str, object]] = []
    acf_by_panel: dict[tuple[str, str], list[float]] = {}
    for feature in features:
        values = train[feature].to_numpy(dtype=np.float64)
        for series_name, builder in series_builders.items():
            transformed = np.asarray(builder(values), dtype=np.float64)
            acf = autocorrelation(transformed, args.max_lag)
            acf_by_panel[(feature, series_name)] = acf
            for lag, value in enumerate(acf):
                rows.append({"feature": feature, "series": series_name, "lag": lag, "acf": fmt(value)})

    write_acf_csv(output_dir / "acf_values.csv", rows)
    save_correlogram_svg(
        output_dir / "train_correlogram.svg",
        acf_by_panel,
        features,
        list(series_builders.keys()),
        args.max_lag,
        len(train),
    )

    summary = [
        "# Train data correlogram",
        "",
        f"train_csv: {config['train_csv']}",
        f"rows: {len(train)}",
        f"features: {', '.join(features)}",
        f"max_lag: {args.max_lag}",
        "",
        "Selected ACF values:",
    ]
    for feature in features:
        for series_name in series_builders:
            acf = acf_by_panel[(feature, series_name)]
            selected = []
            for lag in [1, 5, 10, 20, 40, args.max_lag]:
                if lag <= args.max_lag:
                    selected.append(f"lag{lag}={fmt(acf[lag])}")
            summary.append(f"- {feature} {series_name}: " + ", ".join(selected))
    (output_dir / "summary.md").write_text("\n".join(summary) + "\n", encoding="utf-8")

    print(f"Saved correlogram outputs to {output_dir}")


if __name__ == "__main__":
    main()
