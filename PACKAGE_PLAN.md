# BinAgg Package Development Plan

## Project Overview

**Goal**: Consolidate the BinAgg (Binning-Aggregation for Differentially Private Regression) codebase into a clean, well-tested, open-source Python package.

**Paper Reference**: arXiv:2510.16974v1 - "Differentially Private Linear Regression and Synthetic Data Generation with Statistical Guarantees"

**Authors**: Shurong Lin, Aleksandra Slavković, Deekshith Reddy Bhoomireddy (Pennsylvania State University)

---

## Current State

### Implemented Algorithms (Scattered Across Notebooks)

| Algorithm | Paper Page | Current Location | Status |
|-----------|------------|------------------|--------|
| Algorithm 1: DP Binning-Aggregation Preparation | p.8 | `privtree_binning_XY_with_theta()` in both notebooks + `DP_fy_sum_x_y_counts()` in CI notebook | ✅ Implemented |
| Algorithm 2: DP BinAgg for Linear Regression | p.9 | `models_estimation_se()` in Confidence Intervals.ipynb | ✅ Implemented |
| Algorithm 3: BinAgg for Synthetic Data | p.10 | `DP_fy_sum_x_y_counts_syn_data()` in VER 11 Synthetic data.ipynb | ✅ Implemented |

### Supporting Functions

| Function | Purpose | Location |
|----------|---------|----------|
| `mu_to_epsilon()` | GDP μ → ε conversion | Both notebooks |
| `delta_from_gdp()` | Compute δ from GDP parameters | VER 11 notebook |
| `mu_from_eps_delta()` | Convert (ε,δ)-DP to μ-GDP | VER 11 notebook |
| `eps_from_mu_delta()` | Convert μ-GDP to ε given δ | VER 11 notebook |

### Available Datasets for Testing

| Dataset | File | Size (n, d) | Target Variable |
|---------|------|-------------|-----------------|
| Intrusion Detection | `intrusion detection.csv` | (182, 4) | Number of Barriers |
| Auction Verification | `auction.csv` | (2043, 8) | Property Price |
| Air Quality UCI | `AirQualityUCI.csv` | (9357, 12) | CO concentration |
| Wine Quality (Red) | `winequality-red.csv` | (1599, 11) | Quality rating |
| Wine Quality (White) | `winequality-white.csv` | (4898, 11) | Quality rating |
| Energy Data | `energydata_complete.csv` | (19735, 27) | Appliances usage |

---

## Proposed Package Structure

```
binagg/
├── pyproject.toml              # Package configuration (PEP 517/518)
├── README.md                   # Package documentation
├── LICENSE                     # MIT or Apache 2.0
├── CHANGELOG.md                # Version history
│
├── src/
│   └── binagg/
│       ├── __init__.py         # Package exports
│       ├── privacy.py          # Module: Privacy parameter conversions
│       ├── binning.py          # Module: Algorithm 1 - DP Binning-Aggregation
│       ├── regression.py       # Module: Algorithm 2 - DP Linear Regression
│       ├── synthetic.py        # Module: Algorithm 3 - Synthetic Data Generation
│       └── utils.py            # Shared utilities
│
├── tests/
│   ├── __init__.py
│   ├── conftest.py             # Pytest fixtures (datasets, common params)
│   ├── test_privacy.py         # Tests for privacy conversions
│   ├── test_binning.py         # Tests for Algorithm 1
│   ├── test_regression.py      # Tests for Algorithm 2
│   ├── test_synthetic.py       # Tests for Algorithm 3
│   └── test_integration.py     # End-to-end tests with real datasets
│
├── examples/
│   ├── basic_usage.py          # Simple example
│   ├── regression_with_ci.py   # Algorithm 2 example
│   └── synthetic_data.py       # Algorithm 3 example
│
└── data/                       # Test datasets (or download scripts)
    └── README.md               # Dataset sources and licenses
```

---

## Module Specifications

### Module 1: `privacy.py` - Privacy Parameter Conversions

```python
# Core functions to implement:
def mu_to_epsilon(mu: float) -> float:
    """Convert GDP μ parameter to ε."""

def epsilon_to_mu(epsilon: float) -> float:
    """Convert ε to GDP μ parameter."""

def delta_from_gdp(mu: float, eps: float) -> float:
    """Compute δ from GDP parameters."""

def mu_from_eps_delta(eps: float, delta: float) -> float:
    """Convert (ε,δ)-DP to μ-GDP."""

def eps_from_mu_delta(mu: float, delta: float) -> float:
    """Convert μ-GDP to ε given δ."""

def compose_gdp(*mus: float) -> float:
    """Compose multiple μ-GDP mechanisms: sqrt(sum(μ²))."""
```

### Module 2: `binning.py` - Algorithm 1: DP Binning-Aggregation

```python
# Core class/functions to implement:
@dataclass
class BinAggResult:
    """Result from binning-aggregation."""
    bins: List[List[Tuple[float, float]]]  # K bins, each with d (lower, upper) bounds
    sum_x: np.ndarray                       # (K, d) sum of features per bin
    sum_y: np.ndarray                       # (K,) sum of labels per bin
    counts: np.ndarray                      # (K,) count per bin
    sensitivity_x: np.ndarray               # (K, d) sensitivity per bin per feature
    n_bins: int                             # Number of bins K

def privtree_binning(
    X: np.ndarray,
    y: np.ndarray,
    x_bounds: List[Tuple[float, float]],
    mu_bin: float,
    theta: float = 0.0,
    branching_factor: int = 2,
    min_bins: Optional[int] = None  # defaults to d+1
) -> BinAggResult:
    """
    Algorithm 1: DP Binning-Aggregation Preparation.

    Creates DP bins using PrivTree and computes aggregated statistics.
    """

def privatize_aggregates(
    result: BinAggResult,
    y_bound: float,
    mu_c: float,
    mu_s: float,
    mu_t: float,
    min_count: int = 2
) -> PrivatizedAggregates:
    """
    Privatize the bin-level aggregates (counts, sums).

    Returns privatized counts, sum_x, sum_y with bins filtered by min_count.
    """
```

### Module 3: `regression.py` - Algorithm 2: DP Linear Regression

```python
# Core class/functions to implement:
@dataclass
class DPRegressionResult:
    """Result from DP linear regression."""
    coefficients: np.ndarray           # β̃ estimates
    standard_errors: np.ndarray        # SE(β̃) from sandwich estimator
    confidence_intervals: np.ndarray   # (d, 2) lower/upper bounds
    naive_coefficients: np.ndarray     # β̃_naive (without bias correction)
    naive_standard_errors: np.ndarray  # SE without DP correction
    n_bins: int                        # K
    privacy_budget: float              # Total μ used

def dp_linear_regression(
    X: np.ndarray,
    y: np.ndarray,
    x_bounds: List[Tuple[float, float]],
    y_bounds: Tuple[float, float],
    mu: float,
    theta: float = 0.0,
    alpha: float = 0.05,
    budget_split: Tuple[float, float, float, float] = (1, 3, 3, 3)  # bin:count:x:y
) -> DPRegressionResult:
    """
    Algorithm 2: DP BinAgg for Linear Regression.

    Full pipeline: binning → aggregation → privatization → bias-corrected WLS.
    Returns coefficients with confidence intervals.
    """

def compute_bias_correction_matrix(
    sensitivity_x: np.ndarray,
    noisy_counts: np.ndarray,
    epsilon_x: float
) -> np.ndarray:
    """Compute the bias correction matrix D̃."""

def compute_sandwich_covariance(
    tilde_S: np.ndarray,
    tilde_t: np.ndarray,
    tilde_W: np.ndarray,
    beta: np.ndarray,
    D_k_list: List[np.ndarray],
    K: int,
    d: int
) -> np.ndarray:
    """Compute the sandwich covariance estimator Σ̃ = M̃⁻¹H̃M̃⁻¹."""
```

### Module 4: `synthetic.py` - Algorithm 3: Synthetic Data Generation

```python
# Core class/functions to implement:
@dataclass
class SyntheticDataResult:
    """Result from synthetic data generation."""
    X_synthetic: np.ndarray    # (n_syn, d) synthetic features
    y_synthetic: np.ndarray    # (n_syn,) synthetic labels
    n_samples: int             # Total synthetic samples generated
    n_bins_used: int           # Bins after filtering
    privacy_budget: float      # Total μ used

def generate_synthetic_data(
    X: np.ndarray,
    y: np.ndarray,
    x_bounds: List[Tuple[float, float]],
    y_bounds: Tuple[float, float],
    mu: float,
    theta: float = 0.0,
    budget_split: Tuple[float, float, float, float] = (1, 3, 3, 3)
) -> SyntheticDataResult:
    """
    Algorithm 3: BinAgg for Synthetic Data.

    Full pipeline: binning → aggregation → synthetic sample generation.
    """

def generate_samples_from_bin(
    sum_x: np.ndarray,
    sum_y: float,
    count: int,
    sensitivity_x: np.ndarray,
    sensitivity_y: float,
    epsilon_x: float,
    epsilon_y: float
) -> Tuple[np.ndarray, np.ndarray]:
    """Generate synthetic samples for a single bin."""
```

---

## Implementation Tasks

### Phase 1: Project Setup
- [ ] Create package directory structure
- [ ] Set up `pyproject.toml` with dependencies
- [ ] Create README.md with basic documentation
- [ ] Set up pytest configuration
- [ ] Add LICENSE file (recommend Apache 2.0 or MIT)

### Phase 2: Core Modules
- [ ] **privacy.py**: Implement all GDP ↔ (ε,δ)-DP conversions
- [ ] **binning.py**: Consolidate and refactor `privtree_binning_XY_with_theta()`
- [ ] **binning.py**: Implement `privatize_aggregates()`
- [ ] **regression.py**: Implement `dp_linear_regression()` from `models_estimation_se()`
- [ ] **regression.py**: Implement bias correction and sandwich covariance
- [ ] **synthetic.py**: Implement `generate_synthetic_data()` from `DP_fy_sum_x_y_counts_syn_data()`
- [ ] **utils.py**: Add shared utilities (clipping, validation, etc.)

### Phase 3: Testing
- [ ] **test_privacy.py**: Unit tests for all conversion functions
- [ ] **test_binning.py**: Tests for PrivTree binning
  - [ ] Test bin count ≥ d+1
  - [ ] Test sensitivity calculation
  - [ ] Test with synthetic data (known ground truth)
- [ ] **test_regression.py**: Tests for Algorithm 2
  - [ ] Test bias correction matrix computation
  - [ ] Test coefficient estimation accuracy
  - [ ] Test confidence interval coverage (Monte Carlo)
- [ ] **test_synthetic.py**: Tests for Algorithm 3
  - [ ] Test output shape and bounds
  - [ ] Test that synthetic data preserves regression relationship
- [ ] **test_integration.py**: End-to-end tests with real datasets
  - [ ] Intrusion Detection dataset (small, d=4)
  - [ ] Auction dataset (medium, d=8)
  - [ ] Wine Quality dataset (combined red+white)
  - [ ] Air Quality dataset (larger, d=12)
  - [ ] Energy dataset (large, d=27)

### Phase 4: Documentation & Examples
- [ ] Write docstrings for all public functions
- [ ] Create `examples/basic_usage.py`
- [ ] Create `examples/regression_with_ci.py`
- [ ] Create `examples/synthetic_data.py`
- [ ] Write comprehensive README.md

### Phase 5: Polish & Release
- [ ] Add type hints throughout
- [ ] Run linting (ruff/flake8)
- [ ] Run type checking (mypy)
- [ ] Set up CI/CD (GitHub Actions)
- [ ] Publish to PyPI (optional)

---

## Test Plan Details

### Unit Tests for Privacy Module (`test_privacy.py`)

```python
def test_mu_to_epsilon_known_values():
    """Test μ→ε conversion against known values from paper."""
    # μ=1 should give ε≈1.25 (approximately)

def test_roundtrip_conversion():
    """Test that μ→ε→μ returns original value."""

def test_composition():
    """Test that compose_gdp gives sqrt(sum of squares)."""
```

### Unit Tests for Binning Module (`test_binning.py`)

```python
def test_minimum_bins_created():
    """Ensure at least d+1 bins are created."""

def test_all_points_assigned():
    """Ensure no data points are lost in binning."""

def test_sensitivity_calculation():
    """Test sensitivity = max(|L|, |U|) per bin."""

def test_laplace_noise_scale():
    """Test that Laplace noise uses correct λ parameter."""
```

### Integration Tests with Real Datasets (`test_integration.py`)

```python
@pytest.fixture
def intrusion_dataset():
    """Load intrusion detection dataset (n=182, d=4)."""

@pytest.fixture
def auction_dataset():
    """Load auction dataset (n=2043, d=8)."""

def test_regression_on_intrusion(intrusion_dataset):
    """Test Algorithm 2 on small dataset."""
    X, y, bounds = intrusion_dataset
    result = dp_linear_regression(X, y, bounds, mu=1.0)
    assert result.coefficients.shape == (4,)
    assert all(np.isfinite(result.standard_errors))

def test_synthetic_preserves_relationship(auction_dataset):
    """Test that regression on synthetic ≈ regression on original."""
    X, y, bounds = auction_dataset
    syn_result = generate_synthetic_data(X, y, bounds, mu=1.0)

    # Fit OLS on both
    beta_orig = np.linalg.lstsq(X, y, rcond=None)[0]
    beta_syn = np.linalg.lstsq(syn_result.X_synthetic, syn_result.y_synthetic, rcond=None)[0]

    # Should be reasonably close (within privacy noise)
    assert np.allclose(beta_orig, beta_syn, rtol=0.5)

def test_confidence_interval_coverage():
    """Monte Carlo test: 95% CI should cover true β ~95% of time."""
    # Run 1000 simulations, check empirical coverage
```

---

## Dependencies

```toml
[project]
dependencies = [
    "numpy>=1.20",
    "scipy>=1.7",
    "scikit-learn>=1.0",  # Optional, for comparison
]

[project.optional-dependencies]
dev = [
    "pytest>=7.0",
    "pytest-cov",
    "pandas",  # For loading CSV datasets
    "matplotlib",  # For examples
]
```

---

## Known Implementation Differences from Paper

| Aspect | Paper | Current Implementation | Action Needed |
|--------|-------|----------------------|---------------|
| Privacy parameterization | Uses μ-GDP throughout | Mixes μ and ε | Standardize to μ-GDP |
| Budget split | Flexible (μ_bin, μ_c, μ_s, μ_t) | Fixed 1:3:3:3 via √3 | Make configurable |
| Bounds format | Per-feature bounds | Two versions exist | Consolidate to per-feature |
| Algorithm 1 separation | Separate from Alg 2/3 | Combined with privatization | Keep combined (practical) |

---

## Progress Tracking

| Task | Status | Date | Notes |
|------|--------|------|-------|
| Initial codebase analysis | ✅ Done | | All 3 algorithms identified |
| Package plan created | ✅ Done | | This document |
| Project setup | ⬜ Not started | | |
| privacy.py | ⬜ Not started | | |
| binning.py | ⬜ Not started | | |
| regression.py | ⬜ Not started | | |
| synthetic.py | ⬜ Not started | | |
| Unit tests | ⬜ Not started | | |
| Integration tests | ⬜ Not started | | |
| Documentation | ⬜ Not started | | |
| Release | ⬜ Not started | | |

---

## References

1. Paper: arXiv:2510.16974v1
2. PrivTree: Zhang et al. (2016) - "PrivTree: A Differentially Private Algorithm for Hierarchical Decompositions"
3. GDP: Dong et al. (2022) - "Gaussian Differential Privacy"
