"""
Validation Script: Compare binagg implementation against paper's simulation results.

The original notebook (Confidence Intervals.ipynb) shows:
- n=1000, d=5, mu=1.0, theta=0
- Coverage should be ~95% for valid confidence intervals
- Mean number of bins ~22
"""

import numpy as np
from scipy.stats import t as t_dist
import sys

# Add src to path for local testing
sys.path.insert(0, 'src')

from binagg import dp_linear_regression

def run_validation(n_simulations=100, verbose=True):
    """
    Run validation simulation matching the paper's setup.

    Paper settings:
    - n_samples = 1000
    - n_features = 5
    - mu = 1.0
    - theta = 0
    - x in [0, 1]
    - true_beta in [1, 2]
    """

    n_samples = 1000
    n_features = 5
    mu = 1.0
    theta = 0
    alpha = 0.05

    x_lb, x_ub = 0, 1
    y_lb, y_ub = 0, 7  # From original notebook for d=5

    x_bounds = [(x_lb, x_ub)] * n_features
    y_bounds = (y_lb, y_ub)

    np.random.seed(42)

    # Fixed true coefficients (like in the paper)
    true_coeffs = np.random.uniform(1, 2, size=n_features)

    # Generate fixed X (like in the paper)
    X = np.random.uniform(x_lb, x_ub, size=(n_samples, n_features))

    coverages = []
    estimates = []
    theoretical_ses = []  # Store SE from each simulation
    n_bins_list = []

    print(f"Running {n_simulations} simulations...")
    print(f"Settings: n={n_samples}, d={n_features}, mu={mu}, theta={theta}")
    print(f"True coefficients: {true_coeffs.round(3)}")
    print("-" * 60)

    for sim in range(n_simulations):
        # Generate new Y for each simulation
        Y = X @ true_coeffs + np.random.normal(0, 1, size=n_samples)
        Y = np.clip(Y, y_lb, y_ub)

        # Run our implementation
        result = dp_linear_regression(
            X, Y,
            x_bounds=x_bounds,
            y_bounds=y_bounds,
            mu=mu,
            theta=theta,
            alpha=alpha,
            random_state=sim
        )

        # Check coverage
        K = result.n_bins
        t_alpha = t_dist.ppf(1 - alpha / 2, K)
        lower = result.coefficients - t_alpha * result.standard_errors
        upper = result.coefficients + t_alpha * result.standard_errors
        cover = (true_coeffs >= lower) & (true_coeffs <= upper)

        coverages.append(cover)
        estimates.append(result.coefficients)
        theoretical_ses.append(result.standard_errors)  # Collect theoretical SE
        n_bins_list.append(K)

        if verbose and (sim + 1) % 20 == 0:
            print(f"  Completed {sim + 1}/{n_simulations} simulations...")

    # Compute summary statistics
    coverages = np.array(coverages)
    estimates = np.array(estimates)
    theoretical_ses = np.array(theoretical_ses)

    mean_coverage = np.mean(coverages, axis=0)
    overall_coverage = np.mean(coverages)
    mean_bias = np.mean(estimates - true_coeffs, axis=0)
    empirical_se = np.std(estimates, axis=0)
    avg_theoretical_se = np.mean(theoretical_ses, axis=0)  # Average of estimated SEs
    mean_bins = np.mean(n_bins_list)

    print("\n" + "=" * 70)
    print("VALIDATION RESULTS")
    print("=" * 70)

    print(f"\nNumber of bins: mean={mean_bins:.1f}, min={min(n_bins_list)}, max={max(n_bins_list)}")

    print(f"\nPer-coefficient results:")
    print(f"{'Coef':<8} {'True':<8} {'Bias':<10} {'Emp SE':<12} {'Theo SE':<12} {'Coverage':<10}")
    print("-" * 70)
    for i in range(n_features):
        print(f"beta_{i:<3} {true_coeffs[i]:<8.3f} {mean_bias[i]:<10.4f} "
              f"{empirical_se[i]:<12.4f} {avg_theoretical_se[i]:<12.4f} {mean_coverage[i]:<10.3f}")

    print(f"\nOverall coverage: {overall_coverage:.3f}")
    print(f"Expected coverage: 0.950")

    # SE comparison
    se_ratio = avg_theoretical_se / empirical_se
    print(f"\nSE Ratio (Theo/Emp): {se_ratio.round(3)}")
    print("(Ratio close to 1.0 indicates well-calibrated standard errors)")

    # Validation check
    print("\n" + "=" * 70)
    print("VALIDATION CHECK")
    print("=" * 70)

    passed = True

    # Coverage should be approximately 95% (allow 85-100% for small samples)
    if 0.85 <= overall_coverage <= 1.0:
        print(f"[PASS] Coverage {overall_coverage:.3f} is in acceptable range [0.85, 1.0]")
    else:
        print(f"[FAIL] Coverage {overall_coverage:.3f} is outside acceptable range")
        passed = False

    # Mean bins should be reasonable (paper shows ~22 for this setting)
    if 10 <= mean_bins <= 50:
        print(f"[PASS] Mean bins {mean_bins:.1f} is in acceptable range [10, 50]")
    else:
        print(f"[FAIL] Mean bins {mean_bins:.1f} is outside acceptable range")
        passed = False

    # Bias should be small (relative to true coefficients)
    max_rel_bias = np.max(np.abs(mean_bias / true_coeffs))
    if max_rel_bias < 0.5:  # Less than 50% relative bias
        print(f"[PASS] Max relative bias {max_rel_bias:.3f} is acceptable")
    else:
        print(f"[FAIL] Max relative bias {max_rel_bias:.3f} is too large")
        passed = False

    # SE ratio should be close to 1 (between 0.5 and 2.0)
    mean_se_ratio = np.mean(se_ratio)
    if 0.5 <= mean_se_ratio <= 2.0:
        print(f"[PASS] Mean SE ratio {mean_se_ratio:.3f} is in acceptable range [0.5, 2.0]")
    else:
        print(f"[FAIL] Mean SE ratio {mean_se_ratio:.3f} is outside acceptable range")
        passed = False

    print("\n" + "=" * 70)
    if passed:
        print("OVERALL: VALIDATION PASSED")
    else:
        print("OVERALL: VALIDATION FAILED")
    print("=" * 70)

    return passed, {
        'coverage': overall_coverage,
        'mean_bins': mean_bins,
        'mean_bias': mean_bias,
        'empirical_se': empirical_se,
        'avg_theoretical_se': avg_theoretical_se,
        'se_ratio': se_ratio
    }


if __name__ == "__main__":
    # Run with fewer simulations for quick validation
    # Use n_simulations=500+ for more accurate results
    passed, results = run_validation(n_simulations=1000)

    if not passed:
        sys.exit(1)
