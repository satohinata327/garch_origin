#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import numpy as np

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from garch_utils import (  # noqa: E402
    ensure_run_dirs,
    fit_garch11,
    fit_spec_from_config,
    load_config,
    params_to_json,
    persistence_half_life,
    read_train_csv,
    save_generated_csv,
    save_json,
    simulate_garch11_detailed,
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
    fit_specs = {}
    persistence_summary = {}
    std_residuals = []
    for idx, feature in enumerate(features):
        fit_spec = fit_spec_from_config(config, feature)
        params, variances, residual_values = fit_garch11(scaled[:, idx], fit_spec)
        params_by_feature[feature] = params
        fit_specs[feature] = fit_spec
        persistence = params.alpha + params.beta
        persistence_summary[feature] = {
            "profile": fit_spec.profile,
            "alpha_plus_beta": persistence,
            "volatility_half_life_lags": persistence_half_life(params.alpha, params.beta),
        }
        std_residuals.append(residual_values)

    residual_matrix = np.column_stack(std_residuals)
    residual_corr = np.corrcoef(residual_matrix, rowvar=False)
    has_profile_fit = bool(config.get("garch_fit"))
    exact_standardized_residual_corr = bool(config.get("exact_standardized_residual_corr", False))
    rng = np.random.default_rng(int(config.get("seed", 42)))
    num_generated = int(config.get("num_generated", 10))
    generated_length = int(config.get("generated_length", 1260))
    burn_in = int(config.get("burn_in", 500))
    innovation_distribution = str(config.get("innovation_distribution", "normal"))
    t_copula_config = dict(config.get("t_copula", {}))
    t_degrees_of_freedom = float(t_copula_config.get("degrees_of_freedom", 6.0))

    generated_corr_rows = []
    for idx in range(1, num_generated + 1):
        generated_scaled, generated_std_residuals = simulate_garch11_detailed(
            params_by_feature=params_by_feature,
            residual_corr=residual_corr,
            features=features,
            length=generated_length,
            burn_in=burn_in,
            rng=rng,
            exact_standardized_residual_corr=exact_standardized_residual_corr,
            innovation_distribution=innovation_distribution,
            t_degrees_of_freedom=t_degrees_of_freedom,
        )
        generated = generated_scaled / scale_factor
        save_generated_csv(dirs["generated"] / f"garch_generated_{idx:03d}.csv", generated, features)
        generated_corr = np.corrcoef(generated_std_residuals, rowvar=False)
        for row_idx, left in enumerate(features):
            for col_idx, right in enumerate(features):
                generated_corr_rows.append(
                    {
                        "file": f"garch_generated_{idx:03d}.csv",
                        "left": left,
                        "right": right,
                        "target_train_standardized_residual_corr": f"{float(residual_corr[row_idx, col_idx]):.12g}",
                        "generated_standardized_residual_corr": f"{float(generated_corr[row_idx, col_idx]):.12g}",
                        "abs_error": f"{abs(float(generated_corr[row_idx, col_idx] - residual_corr[row_idx, col_idx])):.12g}",
                    }
                )

    corr_diag_path = dirs["data"] / "generated_standardized_residual_correlations.csv"
    with corr_diag_path.open("w", newline="", encoding="utf-8") as f:
        fieldnames = [
            "file",
            "left",
            "right",
            "target_train_standardized_residual_corr",
            "generated_standardized_residual_corr",
            "abs_error",
        ]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(generated_corr_rows)

    fitted_payload = {
        "model": "univariate_garch11_with_empirical_residual_correlation",
        "scale_factor": scale_factor,
        "features": features,
        "params": params_to_json(params_by_feature),
        "standardized_residual_correlation": residual_corr.tolist(),
        "train_csv": config["train_csv"],
        "n_train_rows": int(len(df)),
        "innovation_distribution": innovation_distribution,
    }
    if innovation_distribution.lower() in {"t", "student_t", "student-t", "t_copula", "t-copula"}:
        fitted_payload["t_copula"] = {"degrees_of_freedom": t_degrees_of_freedom}
    if has_profile_fit or exact_standardized_residual_corr:
        fitted_payload.update(
            {
                "fit_specs": {feature: vars(spec) for feature, spec in fit_specs.items()},
                "persistence_summary": persistence_summary,
                "standardized_residual_correlation_source": "train_data_garch_standardized_residuals",
                "exact_standardized_residual_corr_in_generation": exact_standardized_residual_corr,
            }
        )
    save_json(dirs["data"] / "fitted_params.json", fitted_payload)

    log_text = "\n".join(
        [
            "GARCH(1,1) baseline generation completed.",
            f"train_csv: {config['train_csv']}",
            f"output_dir: {output_dir}",
            f"features: {features}",
            f"num_generated: {num_generated}",
            f"generated_length: {generated_length}",
            f"innovation_distribution: {innovation_distribution}",
            f"t_degrees_of_freedom: {t_degrees_of_freedom}",
            f"exact_standardized_residual_corr: {exact_standardized_residual_corr}",
            f"standardized_residual_corr_diagnostics: {corr_diag_path}",
        ]
    )
    (dirs["logs"] / "generate_log.txt").write_text(log_text + "\n", encoding="utf-8")
    print(log_text)


if __name__ == "__main__":
    main()
