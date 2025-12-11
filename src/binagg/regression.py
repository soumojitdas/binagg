"""
Algorithm 2: DP BinAgg for Linear Regression.

This module implements differentially private linear regression using the
binning-aggregation framework with bias correction and valid confidence intervals.

Reference:
    Lin, S., Slavković, A., & Bhoomireddy, D. R. (2025).
    "Differentially Private Linear Regression and Synthetic Data Generation
    with Statistical Guarantees." arXiv:2510.16974v1
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Tuple

import numpy as np
from numpy.linalg import inv
from scipy.stats import t as t_dist

from binagg.binning import (
    BinAggResult,
    PrivatizedAggregates,
    privatize_aggregates,
    privtree_binning,
)
from binagg.privacy import allocate_budget


@dataclass
class DPRegressionResult:
    """
    Result from differentially private linear regression.

    Attributes
    ----------
    coefficients : np.ndarray
        Bias-corrected DP coefficient estimates β̃. Shape: (d,).
    standard_errors : np.ndarray
        Standard errors from sandwich estimator. Shape: (d,).
    confidence_intervals : np.ndarray
        Confidence intervals for each coefficient. Shape: (d, 2).
    naive_coefficients : np.ndarray
        Naive WLS estimates without bias correction. Shape: (d,).
    naive_standard_errors : np.ndarray
        Standard errors without DP noise correction. Shape: (d,).
    n_bins : int
        Number of bins used (after filtering).
    n_samples_original : int
        Original number of samples.
    alpha : float
        Significance level used for confidence intervals.
    privacy_budget : float
        Total μ-GDP budget used.
    """

    coefficients: np.ndarray
    standard_errors: np.ndarray
    confidence_intervals: np.ndarray
    naive_coefficients: np.ndarray
    naive_standard_errors: np.ndarray
    n_bins: int
    n_samples_original: int
    alpha: float
    privacy_budget: float


def dp_linear_regression(
    X: np.ndarray,
    y: np.ndarray,
    x_bounds: List[Tuple[float, float]],
    y_bounds: Tuple[float, float],
    mu: float,
    theta: float = 0.0,
    alpha: float = 0.05,
    budget_ratios: Tuple[float, float, float, float] = (1, 3, 3, 3),
    min_count: int = 2,
    random_state: Optional[int] = None,
) -> DPRegressionResult:
    """
    Algorithm 2: DP BinAgg for Linear Regression.

    Performs differentially private linear regression with bias correction
    and asymptotic confidence intervals.

    Parameters
    ----------
    X : np.ndarray
        Feature matrix of shape (n, d).
    y : np.ndarray
        Label vector of shape (n,).
    x_bounds : list of tuple
        Per-feature bounds as [(L_1, U_1), ..., (L_d, U_d)].
    y_bounds : tuple
        Bounds on y as (y_min, y_max).
    mu : float
        Total privacy budget in μ-GDP.
    theta : float, optional
        PrivTree splitting threshold. Default is 0.
    alpha : float, optional
        Significance level for confidence intervals. Default is 0.05 (95% CI).
    budget_ratios : tuple of float, optional
        Privacy budget ratios for (binning, count, sum_x, sum_y).
        Default is (1, 3, 3, 3).
    min_count : int, optional
        Minimum noisy count to keep a bin. Default is 2.
    random_state : int, optional
        Random seed for reproducibility.

    Returns
    -------
    DPRegressionResult
        Contains coefficient estimates, standard errors, and confidence intervals.

    Notes
    -----
    The total privacy guarantee is:
        sqrt(μ_bin² + μ_c² + μ_s² + μ_t²) = μ

    The bias-corrected estimator is:
        β̃ = (S̃ᵀW̃S̃ - D̃)⁻¹ S̃ᵀW̃t̃

    where D̃ is the bias correction matrix from Theorem 4.2.

    Examples
    --------
    >>> X = np.random.uniform(0, 1, (100, 2))
    >>> y = X @ [1.5, 2.0] + np.random.normal(0, 0.5, 100)
    >>> result = dp_linear_regression(
    ...     X, y,
    ...     x_bounds=[(0, 1), (0, 1)],
    ...     y_bounds=(-2, 5),
    ...     mu=1.0
    ... )
    >>> result.coefficients.shape
    (2,)
    """
    n_samples, n_features = X.shape
    y = np.asarray(y).flatten()

    # Compute y sensitivity
    y_bound = max(abs(y_bounds[0]), abs(y_bounds[1]))

    # Allocate privacy budget: (binning, count, sum_x, sum_y)
    mu_bin, mu_c, mu_s, mu_t = allocate_budget(mu, budget_ratios)

    # Step 1: Binning (Algorithm 1 - part 1)
    bin_result = privtree_binning(
        X, y, x_bounds, mu_bin, theta=theta, random_state=random_state
    )

    # Step 2: Privatize aggregates (Algorithm 1 - part 2)
    # Combine count, sum_x, sum_y budgets
    mu_agg = np.sqrt(mu_c**2 + mu_s**2 + mu_t**2)
    agg_ratios = (mu_c / mu_agg, mu_s / mu_agg, mu_t / mu_agg)

    priv_agg = privatize_aggregates(
        bin_result,
        y_bound=y_bound,
        mu_agg=mu_agg,
        budget_ratios=agg_ratios,
        min_count=min_count,
        random_state=random_state,
    )

    # Step 3: Compute bias-corrected WLS estimator
    beta_dp, beta_naive, se_dp, se_naive = _compute_dp_wls(
        priv_agg, n_features
    )

    # Step 4: Compute confidence intervals
    K = priv_agg.n_bins
    df = K - n_features  # degrees of freedom
    if df > 0:
        t_crit = t_dist.ppf(1 - alpha / 2, df)
    else:
        t_crit = t_dist.ppf(1 - alpha / 2, 1)  # fallback

    ci_lower = beta_dp - t_crit * se_dp
    ci_upper = beta_dp + t_crit * se_dp
    confidence_intervals = np.column_stack([ci_lower, ci_upper])

    return DPRegressionResult(
        coefficients=beta_dp,
        standard_errors=se_dp,
        confidence_intervals=confidence_intervals,
        naive_coefficients=beta_naive,
        naive_standard_errors=se_naive,
        n_bins=K,
        n_samples_original=n_samples,
        alpha=alpha,
        privacy_budget=mu,
    )


def _compute_dp_wls(
    priv_agg: PrivatizedAggregates,
    n_features: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Compute bias-corrected weighted least squares estimator.

    Returns
    -------
    tuple
        (beta_dp, beta_naive, se_dp, se_naive)
    """
    K = priv_agg.n_bins
    d = n_features

    tilde_S = priv_agg.noisy_sum_x  # (K, d)
    tilde_t = priv_agg.noisy_sum_y  # (K,)
    tilde_W = np.diag(1.0 / priv_agg.noisy_counts)  # (K, K)
    sens_x = priv_agg.sensitivity_x  # (K, d)
    epsilon_s = priv_agg.epsilon_s

    # Compute bias correction matrix D̃
    # D̃ = (1/K) Σ_k w̃_k * D_k where D_k = diag(Δ_k² / ε_s²)
    D_k_list = []
    D = np.zeros((d, d))
    for k in range(K):
        D_k = np.diag(sens_x[k] ** 2 / epsilon_s**2)
        D_k_list.append(D_k)
        D += (1.0 / priv_agg.noisy_counts[k]) * D_k
    D /= K

    # Bias-corrected estimator: β̃ = (S̃ᵀW̃S̃ - D̃)⁻¹ S̃ᵀW̃t̃
    StWS = tilde_S.T @ tilde_W @ tilde_S
    StWt = tilde_S.T @ tilde_W @ tilde_t

    try:
        beta_dp = inv(StWS - D) @ StWt
    except np.linalg.LinAlgError:
        # Fallback: add small regularization
        beta_dp = inv(StWS - D + 1e-6 * np.eye(d)) @ StWt

    # Naive estimator (without bias correction)
    try:
        beta_naive = inv(StWS) @ StWt
    except np.linalg.LinAlgError:
        beta_naive = inv(StWS + 1e-6 * np.eye(d)) @ StWt

    # Compute sandwich covariance estimator for bias-corrected estimator
    se_dp = _compute_sandwich_se(
        tilde_S, tilde_t, tilde_W, beta_dp, D_k_list, D, K, d
    )

    # Naive standard errors (ignoring DP noise)
    try:
        # Using σ² = 1 as placeholder (proper estimation would need residuals)
        Sigma_naive = inv(StWS)
        se_naive = np.sqrt(np.diag(Sigma_naive))
    except np.linalg.LinAlgError:
        se_naive = np.full(d, np.nan)

    return beta_dp, beta_naive, se_dp, se_naive


def _compute_sandwich_se(
    tilde_S: np.ndarray,
    tilde_t: np.ndarray,
    tilde_W: np.ndarray,
    beta: np.ndarray,
    D_k_list: List[np.ndarray],
    D: np.ndarray,
    K: int,
    d: int,
) -> np.ndarray:
    """
    Compute standard errors using the sandwich covariance estimator.

    From Theorem 4.2:
        Σ̃ = M̃⁻¹ H̃ M̃⁻¹

    where:
        M̃ = (1/K)(S̃ᵀW̃S̃) - D̃
        H̃ = (1/(K(K-d))) Σ_k Q̃_k Q̃_kᵀ
        Q̃_k = s̃_k w̃_k (t̃_k - s̃_kᵀβ̃) + w̃_k D_k β̃
    """
    # Compute M̃
    StWS = tilde_S.T @ tilde_W @ tilde_S
    M_tilde = StWS / K - D

    # Compute Q_k for each bin
    Q_list = []
    for k in range(K):
        s_k = tilde_S[k, :]  # (d,)
        w_k = tilde_W[k, k]  # scalar
        t_k = tilde_t[k]  # scalar
        D_k = D_k_list[k]  # (d, d)

        residual = t_k - s_k @ beta
        Q_k = s_k * w_k * residual + w_k * (D_k @ beta)
        Q_list.append(Q_k)

    # Compute H̃
    Q_array = np.array(Q_list)  # (K, d)
    denom = K * max(K - d, 1)  # Avoid division by zero
    H_tilde = (Q_array.T @ Q_array) / denom

    # Compute Σ̃ = M̃⁻¹ H̃ M̃⁻¹
    try:
        M_inv = inv(M_tilde)
        Sigma_tilde = M_inv @ H_tilde @ M_inv
        se = np.sqrt(np.diag(Sigma_tilde))
    except np.linalg.LinAlgError:
        se = np.full(d, np.nan)

    return se


def dp_regression_from_aggregates(
    priv_agg: PrivatizedAggregates,
    n_features: int,
    alpha: float = 0.05,
    mu: float = 1.0,
) -> DPRegressionResult:
    """
    Compute DP regression from pre-computed privatized aggregates.

    This is useful when you want to reuse the same privatized data
    for multiple analyses.

    Parameters
    ----------
    priv_agg : PrivatizedAggregates
        Pre-computed privatized aggregates.
    n_features : int
        Number of features d.
    alpha : float, optional
        Significance level. Default is 0.05.
    mu : float, optional
        Privacy budget used (for reporting). Default is 1.0.

    Returns
    -------
    DPRegressionResult
        Regression results.
    """
    beta_dp, beta_naive, se_dp, se_naive = _compute_dp_wls(priv_agg, n_features)

    K = priv_agg.n_bins
    df = max(K - n_features, 1)
    t_crit = t_dist.ppf(1 - alpha / 2, df)

    ci_lower = beta_dp - t_crit * se_dp
    ci_upper = beta_dp + t_crit * se_dp
    confidence_intervals = np.column_stack([ci_lower, ci_upper])

    return DPRegressionResult(
        coefficients=beta_dp,
        standard_errors=se_dp,
        confidence_intervals=confidence_intervals,
        naive_coefficients=beta_naive,
        naive_standard_errors=se_naive,
        n_bins=K,
        n_samples_original=int(np.sum(priv_agg.true_counts)),
        alpha=alpha,
        privacy_budget=mu,
    )
