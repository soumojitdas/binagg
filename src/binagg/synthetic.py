"""
Algorithm 3: BinAgg for Synthetic Data Generation.

This module implements differentially private synthetic data generation
using the binning-aggregation framework.

Reference:
    Lin, S., Slavković, A., & Bhoomireddy, D. R. (2025).
    "Differentially Private Linear Regression and Synthetic Data Generation
    with Statistical Guarantees." arXiv:2510.16974v1
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Tuple

import numpy as np

from binagg.binning import BinAggResult, privtree_binning
from binagg.privacy import allocate_budget


@dataclass
class SyntheticDataResult:
    """
    Result from synthetic data generation.

    Attributes
    ----------
    X_synthetic : np.ndarray
        Synthetic feature matrix. Shape: (n_syn, d).
    y_synthetic : np.ndarray
        Synthetic label vector. Shape: (n_syn,).
    n_samples : int
        Total number of synthetic samples generated.
    n_bins_used : int
        Number of bins used (after filtering low-count bins).
    samples_per_bin : np.ndarray
        Number of samples generated from each bin.
    privacy_budget : float
        Total μ-GDP budget used.
    """

    X_synthetic: np.ndarray
    y_synthetic: np.ndarray
    n_samples: int
    n_bins_used: int
    samples_per_bin: np.ndarray
    privacy_budget: float


def generate_synthetic_data(
    X: np.ndarray,
    y: np.ndarray,
    x_bounds: List[Tuple[float, float]],
    y_bounds: Tuple[float, float],
    mu: float,
    theta: float = 0.0,
    budget_ratios: Tuple[float, float, float, float] = (1, 3, 3, 3),
    min_count: int = 2,
    clip_output: bool = True,
    random_state: Optional[int] = None,
) -> SyntheticDataResult:
    """
    Algorithm 3: BinAgg for Synthetic Data Generation.

    Generates differentially private synthetic data that preserves the
    joint (X, y) distribution suitable for downstream regression tasks.

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
    budget_ratios : tuple of float, optional
        Privacy budget ratios for (binning, count, sum_x, sum_y).
        Default is (1, 3, 3, 3).
    min_count : int, optional
        Minimum noisy count to generate samples from a bin. Default is 2.
    clip_output : bool, optional
        Whether to clip synthetic data to bounds. Default is True.
    random_state : int, optional
        Random seed for reproducibility.

    Returns
    -------
    SyntheticDataResult
        Contains synthetic features and labels.

    Notes
    -----
    For each bin k with noisy count c̃_k ≥ min_count, we generate c̃_k
    synthetic samples:

        x̃^(k,i) = (s_k + ξˣ) / c̃_k,  ξˣ ~ N(0, c̃_k Δ_k² / ε_s²)
        ỹ^(k,i) = (t_k + ξʸ) / c̃_k,  ξʸ ~ N(0, c̃_k B_y² / ε_t²)

    The total privacy guarantee is sqrt(μ_bin² + μ_c² + μ_s² + μ_t²) = μ.

    By Corollary 3.1 in the paper, aggregating the synthetic data within
    each bin yields the same distribution as the privatized aggregates
    in Algorithm 2.

    Examples
    --------
    >>> X = np.random.uniform(0, 1, (100, 2))
    >>> y = X @ [1.5, 2.0] + np.random.normal(0, 0.5, 100)
    >>> result = generate_synthetic_data(
    ...     X, y,
    ...     x_bounds=[(0, 1), (0, 1)],
    ...     y_bounds=(-2, 5),
    ...     mu=1.0
    ... )
    >>> result.X_synthetic.shape[1]
    2
    """
    if random_state is not None:
        np.random.seed(random_state)

    n_samples, n_features = X.shape
    y = np.asarray(y).flatten()

    # Compute y sensitivity
    y_bound = max(abs(y_bounds[0]), abs(y_bounds[1]))

    # Allocate privacy budget
    mu_bin, mu_c, mu_s, mu_t = allocate_budget(mu, budget_ratios)

    # Step 1: Binning
    bin_result = privtree_binning(
        X, y, x_bounds, mu_bin, theta=theta, random_state=random_state
    )

    # Step 2: Generate synthetic data from each bin
    X_syn_list = []
    y_syn_list = []
    samples_per_bin = []

    # Compute per-bin epsilons
    # Total aggregation budget: sqrt(μ_c² + μ_s² + μ_t²)
    # Split equally among count, sum_x, sum_y
    epsilon_c = mu_c
    epsilon_s = mu_s
    epsilon_t = mu_t

    K = bin_result.n_bins
    d = bin_result.n_features
    bins_array = np.array(bin_result.bins)
    sens_x = np.maximum(np.abs(bins_array[:, :, 0]), np.abs(bins_array[:, :, 1]))

    bins_used = 0
    for k in range(K):
        c_k = bin_result.counts[k]
        s_k = bin_result.sum_x[k]  # (d,)
        t_k = bin_result.sum_y[k]  # scalar
        delta_s_k = sens_x[k]  # (d,)

        # Step 2a: Privatize count
        tilde_c_k = int(np.round(c_k + np.random.normal(0, 1.0 / epsilon_c)))

        if tilde_c_k < min_count:
            samples_per_bin.append(0)
            continue  # Skip this bin

        bins_used += 1

        # Step 2b: Generate synthetic features
        # x̃^(k,i) = (s_k + ξˣ) / c̃_k,  ξˣ ~ N(0, c̃_k Δ_k² / ε_s²)
        cov_x = (tilde_c_k * delta_s_k**2) / (epsilon_s**2)
        noise_x = np.random.normal(0, np.sqrt(cov_x), size=(tilde_c_k, d))
        x_syn_k = (s_k + noise_x) / tilde_c_k

        # Step 2c: Generate synthetic labels
        # ỹ^(k,i) = (t_k + ξʸ) / c̃_k,  ξʸ ~ N(0, c̃_k B_y² / ε_t²)
        var_y = (tilde_c_k * y_bound**2) / (epsilon_t**2)
        noise_y = np.random.normal(0, np.sqrt(var_y), size=tilde_c_k)
        y_syn_k = (t_k + noise_y) / tilde_c_k

        X_syn_list.append(x_syn_k)
        y_syn_list.append(y_syn_k)
        samples_per_bin.append(tilde_c_k)

    # Concatenate all synthetic samples
    if X_syn_list:
        X_synthetic = np.vstack(X_syn_list)
        y_synthetic = np.concatenate(y_syn_list)
    else:
        X_synthetic = np.empty((0, d))
        y_synthetic = np.empty((0,))

    # Optionally clip to bounds
    if clip_output and len(X_synthetic) > 0:
        for i, (lb, ub) in enumerate(x_bounds):
            X_synthetic[:, i] = np.clip(X_synthetic[:, i], lb, ub)
        y_synthetic = np.clip(y_synthetic, y_bounds[0], y_bounds[1])

    return SyntheticDataResult(
        X_synthetic=X_synthetic,
        y_synthetic=y_synthetic,
        n_samples=len(y_synthetic),
        n_bins_used=bins_used,
        samples_per_bin=np.array(samples_per_bin),
        privacy_budget=mu,
    )


def generate_synthetic_from_binning(
    bin_result: BinAggResult,
    y_bound: float,
    mu_synth: float,
    budget_ratios: Tuple[float, float, float] = (1, 1, 1),
    min_count: int = 2,
    x_bounds: Optional[List[Tuple[float, float]]] = None,
    y_bounds: Optional[Tuple[float, float]] = None,
    clip_output: bool = True,
    random_state: Optional[int] = None,
) -> SyntheticDataResult:
    """
    Generate synthetic data from pre-computed binning result.

    This allows reusing the same binning for multiple synthetic datasets
    (though each generation consumes additional privacy budget).

    Parameters
    ----------
    bin_result : BinAggResult
        Pre-computed binning result from privtree_binning.
    y_bound : float
        Bound on |y| values (sensitivity for sum_y).
    mu_synth : float
        Privacy budget for synthetic data generation (excludes binning).
    budget_ratios : tuple of float, optional
        Ratios for (count, sum_x, sum_y). Default is (1, 1, 1).
    min_count : int, optional
        Minimum noisy count to keep a bin. Default is 2.
    x_bounds : list of tuple, optional
        Bounds for clipping. If None, no clipping on X.
    y_bounds : tuple, optional
        Bounds for clipping y. If None, no clipping on y.
    clip_output : bool, optional
        Whether to clip output. Default is True.
    random_state : int, optional
        Random seed.

    Returns
    -------
    SyntheticDataResult
        Contains synthetic features and labels.
    """
    if random_state is not None:
        np.random.seed(random_state)

    K = bin_result.n_bins
    d = bin_result.n_features

    # Split privacy budget
    r_c, r_s, r_t = budget_ratios
    norm_factor = np.sqrt(r_c**2 + r_s**2 + r_t**2)
    epsilon_c = mu_synth * r_c / norm_factor
    epsilon_s = mu_synth * r_s / norm_factor
    epsilon_t = mu_synth * r_t / norm_factor

    X_syn_list = []
    y_syn_list = []
    samples_per_bin = []
    bins_used = 0

    for k in range(K):
        c_k = bin_result.counts[k]
        s_k = bin_result.sum_x[k]
        t_k = bin_result.sum_y[k]
        delta_s_k = bin_result.sensitivity_x[k]

        # Privatize count
        tilde_c_k = int(np.round(c_k + np.random.normal(0, 1.0 / epsilon_c)))

        if tilde_c_k < min_count:
            samples_per_bin.append(0)
            continue

        bins_used += 1

        # Generate synthetic features
        cov_x = (tilde_c_k * delta_s_k**2) / (epsilon_s**2)
        noise_x = np.random.normal(0, np.sqrt(cov_x), size=(tilde_c_k, d))
        x_syn_k = (s_k + noise_x) / tilde_c_k

        # Generate synthetic labels
        var_y = (tilde_c_k * y_bound**2) / (epsilon_t**2)
        noise_y = np.random.normal(0, np.sqrt(var_y), size=tilde_c_k)
        y_syn_k = (t_k + noise_y) / tilde_c_k

        X_syn_list.append(x_syn_k)
        y_syn_list.append(y_syn_k)
        samples_per_bin.append(tilde_c_k)

    # Concatenate
    if X_syn_list:
        X_synthetic = np.vstack(X_syn_list)
        y_synthetic = np.concatenate(y_syn_list)
    else:
        X_synthetic = np.empty((0, d))
        y_synthetic = np.empty((0,))

    # Clip if requested
    if clip_output and len(X_synthetic) > 0:
        if x_bounds is not None:
            for i, (lb, ub) in enumerate(x_bounds):
                X_synthetic[:, i] = np.clip(X_synthetic[:, i], lb, ub)
        if y_bounds is not None:
            y_synthetic = np.clip(y_synthetic, y_bounds[0], y_bounds[1])

    return SyntheticDataResult(
        X_synthetic=X_synthetic,
        y_synthetic=y_synthetic,
        n_samples=len(y_synthetic),
        n_bins_used=bins_used,
        samples_per_bin=np.array(samples_per_bin),
        privacy_budget=mu_synth,
    )
