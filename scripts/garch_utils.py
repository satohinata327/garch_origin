from __future__ import annotations

import csv
import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy.optimize import minimize


@dataclass
class GarchParams:
    mu: float
    omega: float
    alpha: float
    beta: float
    unconditional_variance: float


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


def garch_negative_loglik(theta: np.ndarray, y: np.ndarray) -> float:
    mu, omega, alpha, beta = theta
    if omega <= 0.0 or alpha < 0.0 or beta < 0.0 or alpha + beta >= 0.999:
        return 1e100
    residuals = y - mu
    variances = compute_conditional_variances(residuals, omega, alpha, beta)
    return float(0.5 * np.sum(np.log(2.0 * math.pi) + np.log(variances) + residuals**2 / variances))


def fit_garch11(y: np.ndarray) -> tuple[GarchParams, np.ndarray, np.ndarray]:
    values = np.asarray(y, dtype=np.float64)
    sample_var = float(np.var(values, ddof=1))
    if sample_var <= 0.0:
        raise ValueError("Cannot fit GARCH to a constant series")

    x0 = np.array([float(np.mean(values)), sample_var * 0.05, 0.05, 0.9], dtype=np.float64)
    bounds = [
        (None, None),
        (1e-12, sample_var * 100.0),
        (1e-8, 0.999),
        (1e-8, 0.999),
    ]
    constraints = ({"type": "ineq", "fun": lambda theta: 0.999 - theta[2] - theta[3]},)
    result = minimize(
        garch_negative_loglik,
        x0,
        args=(values,),
        method="SLSQP",
        bounds=bounds,
        constraints=constraints,
        options={"maxiter": 1000, "ftol": 1e-10},
    )
    if not result.success:
        raise RuntimeError(f"GARCH optimization failed: {result.message}")

    mu, omega, alpha, beta = [float(x) for x in result.x]
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


def simulate_garch11(
    params_by_feature: dict[str, GarchParams],
    residual_corr: np.ndarray,
    features: list[str],
    length: int,
    burn_in: int,
    rng: np.random.Generator,
) -> np.ndarray:
    corr = nearest_positive_definite_corr(residual_corr)
    chol = np.linalg.cholesky(corr)
    total_length = length + burn_in
    values = np.zeros((total_length, len(features)), dtype=np.float64)
    residuals = np.zeros((total_length, len(features)), dtype=np.float64)
    variances = np.zeros((total_length, len(features)), dtype=np.float64)

    for col, feature in enumerate(features):
        params = params_by_feature[feature]
        variances[0, col] = max(params.unconditional_variance, 1e-10)
        values[0, col] = params.mu

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

    return values[burn_in:]


def save_generated_csv(path: str | Path, values: np.ndarray, features: list[str]) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with Path(path).open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(features)
        for row in values:
            writer.writerow([f"{float(x):.10g}" if math.isfinite(float(x)) else "" for x in row])


def params_to_json(params_by_feature: dict[str, GarchParams]) -> dict[str, dict[str, float]]:
    return {feature: asdict(params) for feature, params in params_by_feature.items()}
