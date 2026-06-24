from __future__ import annotations

import csv
import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


@dataclass
class GarchParams:
    mu: float
    omega: float
    alpha: float
    beta: float
    unconditional_variance: float


@dataclass
class GarchFitSpec:
    min_persistence: float = 0.0
    initial_max_persistence: float = 0.995
    max_persistence: float = 0.998999
    alpha_min: float = 0.02
    alpha_max: float = 0.20
    beta_min: float = 0.70
    beta_max: float = 0.975
    profile: str = "default"


def load_config(path: str | Path) -> dict[str, Any]:
    with Path(path).open(encoding="utf-8") as f:
        return json.load(f)


def save_json(path: str | Path, data: dict[str, Any]) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with Path(path).open("w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


def ensure_run_dirs(output_dir: str | Path) -> dict[str, Path]:
    root = Path(output_dir)
    dirs = {
        "root": root,
        "config": root / "config",
        "data": root / "data",
        "generated": root / "generated",
        "evaluation": root / "evaluation",
        "logs": root / "logs",
    }
    for path in dirs.values():
        path.mkdir(parents=True, exist_ok=True)
    return dirs


def read_train_csv(path: str | Path, features: list[str]) -> pd.DataFrame:
    df = pd.read_csv(path)
    if "date" in df.columns:
        df["date"] = pd.to_datetime(df["date"])
        df = df.sort_values("date")
    missing = [col for col in features if col not in df.columns]
    if missing:
        raise ValueError(f"Missing columns in {path}: {missing}")
    out = df[features].apply(pd.to_numeric, errors="coerce")
    out = out.replace([np.inf, -np.inf], np.nan).dropna()
    if len(out) < 500:
        raise ValueError(f"Too few usable rows in {path}: {len(out)}")
    return out


def compute_conditional_variances(residuals: np.ndarray, omega: float, alpha: float, beta: float) -> np.ndarray:
    variances = np.empty_like(residuals, dtype=np.float64)
    unconditional = omega / max(1.0 - alpha - beta, 1e-8)
    variances[0] = max(unconditional, np.var(residuals), 1e-10)
    for t in range(1, len(residuals)):
        variances[t] = omega + alpha * residuals[t - 1] ** 2 + beta * variances[t - 1]
        variances[t] = max(variances[t], 1e-10)
    return variances


def garch_negative_loglik(theta: np.ndarray, y: np.ndarray, spec: GarchFitSpec | None = None) -> float:
    mu, omega, alpha, beta = theta
    spec = spec or GarchFitSpec()
    persistence = alpha + beta
    if (
        omega <= 0.0
        or alpha < 0.0
        or beta < 0.0
        or persistence < spec.min_persistence
        or persistence > spec.max_persistence
        or persistence >= 0.999
    ):
        return 1e100
    residuals = y - mu
    variances = compute_conditional_variances(residuals, omega, alpha, beta)
    return float(0.5 * np.sum(np.log(2.0 * math.pi) + np.log(variances) + residuals**2 / variances))


def candidate_grid(sample_var: float, spec: GarchFitSpec | None = None) -> list[tuple[float, float, float]]:
    spec = spec or GarchFitSpec()
    candidates: list[tuple[float, float, float]] = []
    for alpha in np.linspace(spec.alpha_min, spec.alpha_max, 19):
        for beta in np.linspace(spec.beta_min, spec.beta_max, 28):
            persistence = alpha + beta
            if spec.min_persistence <= persistence < spec.initial_max_persistence:
                omega = max((1.0 - alpha - beta) * sample_var, 1e-12)
                candidates.append((float(omega), float(alpha), float(beta)))
    return candidates


def fit_spec_from_config(config: dict[str, Any], feature: str) -> GarchFitSpec:
    garch_config = config.get("garch_fit", {})
    default_spec = dict(garch_config.get("default", {}))
    feature_spec = dict(garch_config.get("features", {}).get(feature, {}))
    values = {**default_spec, **feature_spec}
    return GarchFitSpec(
        min_persistence=float(values.get("min_persistence", 0.0)),
        initial_max_persistence=float(
            values.get("initial_max_persistence", values.get("max_persistence", 0.995))
        ),
        max_persistence=float(values.get("max_persistence", 0.998999)),
        alpha_min=float(values.get("alpha_min", 0.02)),
        alpha_max=float(values.get("alpha_max", 0.20)),
        beta_min=float(values.get("beta_min", 0.70)),
        beta_max=float(values.get("beta_max", 0.975)),
        profile=str(values.get("profile", "default")),
    )


def persistence_half_life(alpha: float, beta: float) -> float | None:
    persistence = alpha + beta
    if persistence <= 0.0 or persistence >= 1.0:
        return None
    return math.log(0.5) / math.log(persistence)


def fit_garch11(y: np.ndarray, spec: GarchFitSpec | None = None) -> tuple[GarchParams, np.ndarray, np.ndarray]:
    spec = spec or GarchFitSpec()
    values = np.asarray(y, dtype=np.float64)
    sample_var = float(np.var(values, ddof=1))
    if sample_var <= 0.0:
        raise ValueError("Cannot fit GARCH to a constant series")

    mu = float(np.mean(values))
    best_score = math.inf
    candidates = candidate_grid(sample_var, spec)
    if not candidates:
        raise ValueError(f"No GARCH candidates satisfy fit spec: {spec}")
    best_params = candidates[0]
    for omega, alpha, beta in candidates:
        score = garch_negative_loglik(np.array([mu, omega, alpha, beta]), values, spec)
        if score < best_score:
            best_score = score
            best_params = (omega, alpha, beta)

    alpha_radius = 0.02
    beta_radius = 0.03
    for _ in range(5):
        _, best_alpha, best_beta = best_params
        improved = False
        for alpha in np.linspace(max(1e-6, best_alpha - alpha_radius), min(0.5, best_alpha + alpha_radius), 9):
            for beta in np.linspace(max(1e-6, best_beta - beta_radius), min(0.998, best_beta + beta_radius), 9):
                persistence = alpha + beta
                if persistence < spec.min_persistence or persistence > spec.max_persistence or persistence >= 0.999:
                    continue
                omega = max((1.0 - alpha - beta) * sample_var, 1e-12)
                score = garch_negative_loglik(np.array([mu, omega, alpha, beta]), values, spec)
                if score < best_score:
                    best_score = score
                    best_params = (float(omega), float(alpha), float(beta))
                    improved = True
        alpha_radius *= 0.5
        beta_radius *= 0.5
        if not improved:
            continue

    omega, alpha, beta = best_params
    residuals = values - mu
    variances = compute_conditional_variances(residuals, omega, alpha, beta)
    std_residuals = residuals / np.sqrt(variances)
    params = GarchParams(
        mu=mu,
        omega=omega,
        alpha=alpha,
        beta=beta,
        unconditional_variance=omega / max(1.0 - alpha - beta, 1e-8),
    )
    return params, variances, std_residuals


def exact_sample_corr_innovations(length: int, corr: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    if length < corr.shape[0] + 2:
        raise ValueError("length is too short to impose an exact sample correlation")

    target = nearest_positive_definite_corr(corr)
    raw = rng.standard_normal((length, target.shape[0]))
    raw = raw - raw.mean(axis=0, keepdims=True)
    raw_std = raw.std(axis=0, ddof=1)
    raw_std[raw_std <= 0.0] = 1.0
    standardized = raw / raw_std

    sample_corr = nearest_positive_definite_corr(np.corrcoef(standardized, rowvar=False))
    sample_chol = np.linalg.cholesky(sample_corr)
    target_chol = np.linalg.cholesky(target)
    whitened = standardized @ np.linalg.inv(sample_chol).T
    matched = whitened @ target_chol.T
    matched = matched - matched.mean(axis=0, keepdims=True)
    matched_std = matched.std(axis=0, ddof=1)
    matched_std[matched_std <= 0.0] = 1.0
    return matched / matched_std


def nearest_positive_definite_corr(corr: np.ndarray) -> np.ndarray:
    matrix = np.asarray(corr, dtype=np.float64)
    matrix = (matrix + matrix.T) / 2.0
    np.fill_diagonal(matrix, 1.0)
    jitter = 1e-10
    for _ in range(8):
        try:
            np.linalg.cholesky(matrix)
            return matrix
        except np.linalg.LinAlgError:
            matrix = matrix + np.eye(matrix.shape[0]) * jitter
            jitter *= 10.0
    return np.eye(matrix.shape[0])


def simulate_garch11_detailed(
    params_by_feature: dict[str, GarchParams],
    residual_corr: np.ndarray,
    features: list[str],
    length: int,
    burn_in: int,
    rng: np.random.Generator,
    exact_standardized_residual_corr: bool = False,
) -> tuple[np.ndarray, np.ndarray]:
    corr = nearest_positive_definite_corr(residual_corr)
    total_length = length + burn_in
    values = np.zeros((total_length, len(features)), dtype=np.float64)
    residuals = np.zeros((total_length, len(features)), dtype=np.float64)
    variances = np.zeros((total_length, len(features)), dtype=np.float64)

    for col, feature in enumerate(features):
        params = params_by_feature[feature]
        variances[0, col] = max(params.unconditional_variance, 1e-10)
        values[0, col] = params.mu

    if exact_standardized_residual_corr:
        burn_innovations = exact_sample_corr_innovations(max(burn_in, len(features) + 2), corr, rng)[:burn_in]
        kept_innovations = exact_sample_corr_innovations(length, corr, rng)
        innovations = np.vstack([burn_innovations, kept_innovations])
    else:
        chol = np.linalg.cholesky(corr)
        innovations = rng.standard_normal((total_length, len(features))) @ chol.T

    for t in range(1, total_length):
        for col, feature in enumerate(features):
            params = params_by_feature[feature]
            variances[t, col] = (
                params.omega
                + params.alpha * residuals[t - 1, col] ** 2
                + params.beta * variances[t - 1, col]
            )
            variances[t, col] = max(variances[t, col], 1e-10)
            residuals[t, col] = math.sqrt(variances[t, col]) * innovations[t, col]
            values[t, col] = params.mu + residuals[t, col]

    return values[burn_in:], innovations[burn_in:]


def simulate_garch11(
    params_by_feature: dict[str, GarchParams],
    residual_corr: np.ndarray,
    features: list[str],
    length: int,
    burn_in: int,
    rng: np.random.Generator,
    exact_standardized_residual_corr: bool = False,
) -> np.ndarray:
    values, _ = simulate_garch11_detailed(
        params_by_feature=params_by_feature,
        residual_corr=residual_corr,
        features=features,
        length=length,
        burn_in=burn_in,
        rng=rng,
        exact_standardized_residual_corr=exact_standardized_residual_corr,
    )
    return values


def save_generated_csv(path: str | Path, values: np.ndarray, features: list[str]) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with Path(path).open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(features)
        for row in values:
            writer.writerow([f"{float(x):.10g}" if math.isfinite(float(x)) else "" for x in row])


def params_to_json(params_by_feature: dict[str, GarchParams]) -> dict[str, dict[str, float]]:
    return {feature: asdict(params) for feature, params in params_by_feature.items()}
