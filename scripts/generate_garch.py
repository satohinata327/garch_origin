#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from garch_utils import (  # noqa: E402
    ensure_run_dirs,
    fit_garch11,
    load_config,
    params_to_json,
    read_train_csv,
    save_generated_csv,
    save_json,
    simulate_garch11,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="garch_origin/config/garch11_baseline.json")
    args = parser.parse_args()

    config = load_config(args.config)
    features = list(config.get("features", ["sp500", "DGS10"]))
    output_dir = Path(config.get("output_dir", "garch_origin/runs/garch11_baseline"))
    dirs = ensure_run_dirs(output_dir)
    save_json(dirs["config"] / "used_config.json", config)

    scale_factor = float(config.get("scale_factor", 100.0))
    df = read_train_csv(config["train_csv"], features)
    scaled = df.to_numpy(dtype=np.float64) * scale_factor

    params_by_feature = {}
    std_residuals = []
    for idx, feature in enumerate(features):
        params, variances, residual_values = fit_garch11(scaled[:, idx])
        params_by_feature[feature] = params
        std_residuals.append(residual_values)

    residual_matrix = np.column_stack(std_residuals)
    residual_corr = np.corrcoef(residual_matrix, rowvar=False)
    rng = np.random.default_rng(int(config.get("seed", 42)))
    num_generated = int(config.get("num_generated", 10))
    generated_length = int(config.get("generated_length", 1260))
    burn_in = int(config.get("burn_in", 500))

    for idx in range(1, num_generated + 1):
        generated_scaled = simulate_garch11(
            params_by_feature=params_by_feature,
            residual_corr=residual_corr,
            features=features,
            length=generated_length,
            burn_in=burn_in,
            rng=rng,
        )
        generated = generated_scaled / scale_factor
        save_generated_csv(dirs["generated"] / f"garch_generated_{idx:03d}.csv", generated, features)

    save_json(
        dirs["data"] / "fitted_params.json",
        {
            "model": "univariate_garch11_with_empirical_residual_correlation",
            "scale_factor": scale_factor,
            "features": features,
            "params": params_to_json(params_by_feature),
            "standardized_residual_correlation": residual_corr.tolist(),
            "train_csv": config["train_csv"],
            "n_train_rows": int(len(df)),
        },
    )

    log_text = "\n".join(
        [
            "GARCH(1,1) baseline generation completed.",
            f"train_csv: {config['train_csv']}",
            f"output_dir: {output_dir}",
            f"features: {features}",
            f"num_generated: {num_generated}",
            f"generated_length: {generated_length}",
        ]
    )
    (dirs["logs"] / "generate_log.txt").write_text(log_text + "\n", encoding="utf-8")
    print(log_text)


if __name__ == "__main__":
    main()
