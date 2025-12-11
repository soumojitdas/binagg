"""
Algorithm 1: DP Binning-Aggregation Preparation.

This module implements the PrivTree-based binning strategy for partitioning
the feature space and computing aggregated statistics (counts, sums) per bin.

Reference:
    Zhang, J., Xiao, X., & Xie, X. (2016). "PrivTree: A differentially private
    algorithm for hierarchical decompositions." SIGMOD.
"""

from __future__ import annotations

import random
from dataclasses import dataclass
from typing import List, Optional, Tuple

import numpy as np

from binagg.privacy import mu_to_epsilon


@dataclass
class BinAggResult:
    """
    Result from the binning-aggregation preparation step.

    Attributes
    ----------
    bins : list of list of tuple
        K bins, each containing d (lower, upper) bounds per feature.
        Shape: (K, d, 2) conceptually.
    sum_x : np.ndarray
        Sum of features per bin. Shape: (K, d).
    sum_y : np.ndarray
        Sum of labels per bin. Shape: (K,).
    counts : np.ndarray
        Count of samples per bin. Shape: (K,).
    sensitivity_x : np.ndarray
        Sensitivity per bin per feature: max(|L_ki|, |U_ki|). Shape: (K, d).
    n_bins : int
        Number of bins K.
    n_features : int
        Number of features d.
    """

    bins: List[List[Tuple[float, float]]]
    sum_x: np.ndarray
    sum_y: np.ndarray
    counts: np.ndarray
    sensitivity_x: np.ndarray
    n_bins: int
    n_features: int


@dataclass
class PrivatizedAggregates:
    """
    Privatized bin-level aggregates after adding noise.

    Attributes
    ----------
    bins : list of list of tuple
        Remaining bins after filtering (those with noisy count >= min_count).
    noisy_counts : np.ndarray
        Privatized counts per bin. Shape: (K',).
    noisy_sum_x : np.ndarray
        Privatized sum of features per bin. Shape: (K', d).
    noisy_sum_y : np.ndarray
        Privatized sum of labels per bin. Shape: (K',).
    sensitivity_x : np.ndarray
        Sensitivity per remaining bin. Shape: (K', d).
    true_counts : np.ndarray
        Original counts for remaining bins. Shape: (K',).
    epsilon_c : float
        Privacy parameter used for count privatization.
    epsilon_s : float
        Privacy parameter used for sum_x privatization.
    epsilon_t : float
        Privacy parameter used for sum_y privatization.
    n_bins : int
        Number of remaining bins K'.
    """

    bins: List[List[Tuple[float, float]]]
    noisy_counts: np.ndarray
    noisy_sum_x: np.ndarray
    noisy_sum_y: np.ndarray
    sensitivity_x: np.ndarray
    true_counts: np.ndarray
    epsilon_c: float
    epsilon_s: float
    epsilon_t: float
    n_bins: int


def privtree_binning(
    X: np.ndarray,
    y: np.ndarray,
    x_bounds: List[Tuple[float, float]],
    mu_bin: float,
    theta: float = 0.0,
    branching_factor: int = 2,
    min_bins: Optional[int] = None,
    random_state: Optional[int] = None,
) -> BinAggResult:
    """
    Algorithm 1: DP Binning-Aggregation Preparation (binning step).

    Creates a differentially private partition of the feature space using
    the PrivTree algorithm, then computes aggregated statistics per bin.

    Parameters
    ----------
    X : np.ndarray
        Feature matrix of shape (n, d).
    y : np.ndarray
        Label vector of shape (n,).
    x_bounds : list of tuple
        Per-feature bounds as [(L_1, U_1), (L_2, U_2), ..., (L_d, U_d)].
    mu_bin : float
        Privacy budget (μ-GDP) for the binning step.
    theta : float, optional
        Splitting threshold. Default is 0. Negative values lead to more splits.
    branching_factor : int, optional
        Branching factor for tree splits. Default is 2 (binary).
    min_bins : int, optional
        Minimum number of bins to create. Default is d+1.
    random_state : int, optional
        Random seed for reproducibility.

    Returns
    -------
    BinAggResult
        Contains bins, sums, counts, and sensitivities.

    Notes
    -----
    The PrivTree algorithm satisfies ε-DP with:
        λ = (2β - 1) / ((β - 1) * ε)
        δ = λ * ln(β)

    where β is the branching factor and ε = mu_to_epsilon(mu_bin).
    """
    if random_state is not None:
        np.random.seed(random_state)
        random.seed(random_state)

    n_samples, n_features = X.shape
    y = np.asarray(y).flatten()

    if len(x_bounds) != n_features:
        raise ValueError(
            f"x_bounds must have {n_features} entries, got {len(x_bounds)}"
        )

    if min_bins is None:
        min_bins = n_features + 1

    # PrivTree configuration
    beta = branching_factor
    epsilon = mu_to_epsilon(mu_bin)
    lambda_ = (2 * beta - 1) / (beta - 1) / epsilon
    delta_depth = lambda_ * np.log(beta)

    # Initialize bins with slightly extended upper bounds for inclusivity
    initial_bins = [(lb, ub + 1e-8) for (lb, ub) in x_bounds]
    bins_queue = [initial_bins]
    depths_queue = [0]

    x_bins_list: List[List[Tuple[float, float]]] = []
    sum_x_list: List[np.ndarray] = []
    sum_y_list: List[float] = []
    counts_list: List[int] = []

    # PrivTree recursive partitioning
    while bins_queue:
        current_bins = bins_queue.pop(0)
        current_depth = depths_queue.pop(0)

        # Find points in this bin
        bin_mask = np.all(
            [
                (X[:, i] >= current_bins[i][0]) & (X[:, i] <= current_bins[i][1])
                for i in range(n_features)
            ],
            axis=0,
        )

        X_bin = X[bin_mask]
        y_bin = y[bin_mask]
        n_bin = len(X_bin)

        # Compute aggregates
        sum_y_bin = float(np.sum(y_bin)) if n_bin > 0 else 0.0
        sum_x_bin = np.sum(X_bin, axis=0) if n_bin > 0 else np.zeros(n_features)

        # PrivTree scoring with depth penalty
        raw_score = n_bin - delta_depth * current_depth
        biased_score = max(raw_score, theta - delta_depth)
        noisy_score = biased_score + np.random.laplace(scale=lambda_)

        if noisy_score > theta:
            # Split: find dimension with largest width
            widths = [right - left for (left, right) in current_bins]
            split_dim = int(np.argmax(widths))
            midpoint = (current_bins[split_dim][0] + current_bins[split_dim][1]) / 2

            left_bin = list(current_bins)
            right_bin = list(current_bins)
            left_bin[split_dim] = (current_bins[split_dim][0], midpoint)
            right_bin[split_dim] = (midpoint, current_bins[split_dim][1])

            bins_queue.extend([left_bin, right_bin])
            depths_queue.extend([current_depth + 1, current_depth + 1])
        else:
            # Leaf node: store this bin
            x_bins_list.append(current_bins)
            sum_x_list.append(sum_x_bin)
            sum_y_list.append(sum_y_bin)
            counts_list.append(n_bin)

    # Post-processing: ensure at least min_bins
    if len(x_bins_list) == 0:
        # Edge case: initialize with full domain
        bin_mask = np.all(
            [
                (X[:, i] >= initial_bins[i][0]) & (X[:, i] <= initial_bins[i][1])
                for i in range(n_features)
            ],
            axis=0,
        )
        X_bin = X[bin_mask]
        y_bin = y[bin_mask]
        x_bins_list = [initial_bins]
        sum_x_list = [np.sum(X_bin, axis=0) if len(X_bin) > 0 else np.zeros(n_features)]
        sum_y_list = [float(np.sum(y_bin)) if len(y_bin) > 0 else 0.0]
        counts_list = [len(X_bin)]

    # Force additional splits if needed
    while len(x_bins_list) < min_bins:
        x_bins_list, sum_x_list, sum_y_list, counts_list = _split_largest_bin(
            X, y, x_bins_list, sum_x_list, sum_y_list, counts_list, n_features
        )

    # Compute sensitivity for each bin
    bins_array = np.array(x_bins_list)  # Shape: (K, d, 2)
    sensitivity_x = np.maximum(
        np.abs(bins_array[:, :, 0]), np.abs(bins_array[:, :, 1])
    )

    return BinAggResult(
        bins=x_bins_list,
        sum_x=np.array(sum_x_list),
        sum_y=np.array(sum_y_list),
        counts=np.array(counts_list),
        sensitivity_x=sensitivity_x,
        n_bins=len(x_bins_list),
        n_features=n_features,
    )


def _split_largest_bin(
    X: np.ndarray,
    y: np.ndarray,
    x_bins_list: List[List[Tuple[float, float]]],
    sum_x_list: List[np.ndarray],
    sum_y_list: List[float],
    counts_list: List[int],
    n_features: int,
) -> Tuple[List, List, List, List]:
    """Split the bin with largest volume to increase bin count."""
    # Find bin with largest volume
    volumes = [
        np.prod([right - left for (left, right) in bin_bounds])
        for bin_bounds in x_bins_list
    ]
    max_idx = int(np.argmax(volumes))
    current_bin = x_bins_list[max_idx]

    # Split along widest dimension
    widths = [right - left for (left, right) in current_bin]
    max_width = max(widths)
    max_dims = [i for i, w in enumerate(widths) if w == max_width]
    split_dim = random.choice(max_dims)

    left_edge, right_edge = current_bin[split_dim]
    midpoint = (left_edge + right_edge) / 2

    # Create child bins
    left_bin = list(current_bin)
    right_bin = list(current_bin)
    left_bin[split_dim] = (left_edge, midpoint)
    right_bin[split_dim] = (midpoint, right_edge)

    # Find points in current bin
    bin_mask = np.all(
        [
            (X[:, i] >= current_bin[i][0]) & (X[:, i] <= current_bin[i][1])
            for i in range(n_features)
        ],
        axis=0,
    )
    X_bin = X[bin_mask]
    y_bin = y[bin_mask]

    # Split points
    left_mask = X_bin[:, split_dim] < midpoint
    right_mask = ~left_mask

    X_left, y_left = X_bin[left_mask], y_bin[left_mask]
    X_right, y_right = X_bin[right_mask], y_bin[right_mask]

    # Compute aggregates for children
    count_left = len(X_left)
    count_right = len(X_right)
    sum_x_left = np.sum(X_left, axis=0) if count_left > 0 else np.zeros(n_features)
    sum_y_left = float(np.sum(y_left)) if count_left > 0 else 0.0
    sum_x_right = np.sum(X_right, axis=0) if count_right > 0 else np.zeros(n_features)
    sum_y_right = float(np.sum(y_right)) if count_right > 0 else 0.0

    # Replace parent with children
    x_bins_list.pop(max_idx)
    sum_x_list.pop(max_idx)
    sum_y_list.pop(max_idx)
    counts_list.pop(max_idx)

    x_bins_list.extend([left_bin, right_bin])
    sum_x_list.extend([sum_x_left, sum_x_right])
    sum_y_list.extend([sum_y_left, sum_y_right])
    counts_list.extend([count_left, count_right])

    return x_bins_list, sum_x_list, sum_y_list, counts_list


def privatize_aggregates(
    result: BinAggResult,
    y_bound: float,
    mu_agg: float,
    budget_ratios: Tuple[float, float, float] = (1.0, 1.0, 1.0),
    min_count: int = 2,
    random_state: Optional[int] = None,
) -> PrivatizedAggregates:
    """
    Privatize the bin-level aggregates (counts, sum_x, sum_y).

    This completes Algorithm 1 by adding Gaussian noise to the aggregated
    statistics and filtering bins with low noisy counts.

    Parameters
    ----------
    result : BinAggResult
        Output from privtree_binning.
    y_bound : float
        Bound on |y| values (sensitivity for sum_y).
    mu_agg : float
        Privacy budget (μ-GDP) for aggregation step.
    budget_ratios : tuple of float, optional
        Ratios for splitting budget among (count, sum_x, sum_y).
        Default is (1, 1, 1) for equal split.
    min_count : int, optional
        Minimum noisy count to keep a bin. Default is 2.
    random_state : int, optional
        Random seed for reproducibility.

    Returns
    -------
    PrivatizedAggregates
        Contains privatized counts, sums, and filtered bins.

    Notes
    -----
    The privacy budget is split as:
        ε_c = μ_agg * r_c / sqrt(r_c² + r_s² + r_t²)
        ε_s = μ_agg * r_s / sqrt(r_c² + r_s² + r_t²)
        ε_t = μ_agg * r_t / sqrt(r_c² + r_s² + r_t²)

    where (r_c, r_s, r_t) are the budget_ratios.
    """
    if random_state is not None:
        np.random.seed(random_state)

    K = result.n_bins
    d = result.n_features

    # Split privacy budget
    r_c, r_s, r_t = budget_ratios
    norm_factor = np.sqrt(r_c**2 + r_s**2 + r_t**2)
    epsilon_c = mu_agg * r_c / norm_factor
    epsilon_s = mu_agg * r_s / norm_factor
    epsilon_t = mu_agg * r_t / norm_factor

    # Privatize counts: c̃_k = round(c_k + N(0, 1/ε_c²))
    noise_c = np.random.normal(0, 1.0 / epsilon_c, size=K)
    noisy_counts = np.rint(result.counts + noise_c).astype(int)

    # Privatize sum_x: s̃_k = s_k + N(0, Δ_k²/ε_s²)
    noise_x = np.random.normal(0, result.sensitivity_x / epsilon_s)
    noisy_sum_x = result.sum_x + noise_x

    # Privatize sum_y: t̃_k = t_k + N(0, B_y²/ε_t²)
    noise_y = np.random.normal(0, y_bound / epsilon_t, size=K)
    noisy_sum_y = result.sum_y + noise_y

    # Filter bins with noisy count < min_count
    keep_mask = noisy_counts >= min_count
    kept_indices = np.where(keep_mask)[0]

    filtered_bins = [result.bins[i] for i in kept_indices]
    filtered_noisy_counts = np.maximum(1, noisy_counts[keep_mask])  # Ensure >= 1
    filtered_noisy_sum_x = noisy_sum_x[keep_mask]
    filtered_noisy_sum_y = noisy_sum_y[keep_mask]
    filtered_sensitivity_x = result.sensitivity_x[keep_mask]
    filtered_true_counts = result.counts[keep_mask]

    return PrivatizedAggregates(
        bins=filtered_bins,
        noisy_counts=filtered_noisy_counts,
        noisy_sum_x=filtered_noisy_sum_x,
        noisy_sum_y=filtered_noisy_sum_y,
        sensitivity_x=filtered_sensitivity_x,
        true_counts=filtered_true_counts,
        epsilon_c=epsilon_c,
        epsilon_s=epsilon_s,
        epsilon_t=epsilon_t,
        n_bins=len(filtered_bins),
    )
