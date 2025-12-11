# Publishing to TestPyPI and PyPI

This guide walks you through publishing the `binagg` package to TestPyPI (for testing) and eventually PyPI (for production).

## Prerequisites

### 1. Install Build Tools

```bash
pip install build twine
```

### 2. Create TestPyPI Account

1. Go to https://test.pypi.org/account/register/
2. Create an account
3. Verify your email
4. Go to Account Settings → API tokens
5. Create a new API token with scope "Entire account"
6. **Save the token** (starts with `pypi-`)

### 3. Create PyPI Account (for production later)

Same steps at https://pypi.org/account/register/

## Step-by-Step Publishing

### Step 1: Update Version Number

Edit `src/binagg/__init__.py` and update the version:

```python
__version__ = "0.1.0"  # Change this for each release
```

Also update `pyproject.toml`:

```toml
version = "0.1.0"
```

### Step 2: Clean Previous Builds

```bash
# On Windows (PowerShell)
Remove-Item -Recurse -Force dist, build, *.egg-info -ErrorAction SilentlyContinue
Remove-Item -Recurse -Force src\*.egg-info -ErrorAction SilentlyContinue

# On Windows (Command Prompt)
rmdir /s /q dist build 2>nul
for /d %i in (*.egg-info) do rmdir /s /q "%i" 2>nul
for /d %i in (src\*.egg-info) do rmdir /s /q "%i" 2>nul

# On Linux/Mac
rm -rf dist/ build/ *.egg-info src/*.egg-info
```

### Step 3: Build the Package

```bash
python -m build
```

This creates two files in `dist/`:
- `binagg-0.1.0.tar.gz` (source distribution)
- `binagg-0.1.0-py3-none-any.whl` (wheel)

### Step 4: Upload to TestPyPI

```bash
python -m twine upload --repository testpypi dist/*
```

When prompted:
- Username: `__token__`
- Password: Your API token (the one starting with `pypi-`)

**Or use a config file** (`~/.pypirc` on Linux/Mac, `%USERPROFILE%\.pypirc` on Windows):

```ini
[testpypi]
username = __token__
password = pypi-your-token-here
```

Then just run:
```bash
python -m twine upload --repository testpypi dist/*
```

### Step 5: Test Installation from TestPyPI

```bash
# Create a fresh virtual environment
python -m venv test_env
test_env\Scripts\activate  # Windows
# source test_env/bin/activate  # Linux/Mac

# Install from TestPyPI
pip install --index-url https://test.pypi.org/simple/ --extra-index-url https://pypi.org/simple/ binagg

# Test it works
python -c "from binagg import dp_linear_regression; print('Success!')"
```

**Note:** `--extra-index-url https://pypi.org/simple/` is needed because TestPyPI doesn't have all dependencies (numpy, scipy).

### Step 6: Publish to Production PyPI (When Ready)

```bash
python -m twine upload dist/*
```

Then install normally:
```bash
pip install binagg
```

## Versioning Guidelines

Use [Semantic Versioning](https://semver.org/):

- `0.1.0` - Initial release
- `0.1.1` - Bug fixes
- `0.2.0` - New features (backward compatible)
- `1.0.0` - First stable release
- `1.1.0` - New features in stable version
- `2.0.0` - Breaking changes

## Complete Publishing Checklist

- [ ] Update version in `src/binagg/__init__.py`
- [ ] Update version in `pyproject.toml`
- [ ] Run all tests: `pytest tests/ -v`
- [ ] Update CHANGELOG (if you have one)
- [ ] Clean old builds
- [ ] Build: `python -m build`
- [ ] Upload to TestPyPI: `twine upload --repository testpypi dist/*`
- [ ] Test install from TestPyPI
- [ ] Upload to PyPI: `twine upload dist/*`
- [ ] Create GitHub release/tag

## Troubleshooting

### "File already exists" Error

You can't upload the same version twice. Bump the version number:
- `0.1.0` → `0.1.1` for a fix
- `0.1.0` → `0.1.0.post1` for a re-upload (not recommended)

### "Invalid token" Error

Make sure:
1. You're using `__token__` as the username (literally)
2. The password is the full token including `pypi-` prefix
3. You created the token for TestPyPI (not PyPI) if uploading to TestPyPI

### Dependencies Not Found

TestPyPI doesn't have all packages. Use:
```bash
pip install --index-url https://test.pypi.org/simple/ \
    --extra-index-url https://pypi.org/simple/ \
    binagg
```

## Quick Commands Reference

```bash
# Build
python -m build

# Upload to TestPyPI
python -m twine upload --repository testpypi dist/*

# Upload to PyPI
python -m twine upload dist/*

# Install from TestPyPI
pip install -i https://test.pypi.org/simple/ --extra-index-url https://pypi.org/simple/ binagg

# Install from PyPI
pip install binagg
```

## Sharing with Testers

Send testers these instructions:

```
To install binagg for testing:

pip install -i https://test.pypi.org/simple/ --extra-index-url https://pypi.org/simple/ binagg

Then try:

from binagg import dp_linear_regression
import numpy as np

X = np.random.uniform(0, 10, (100, 2))
y = X @ [1.5, 2.0] + np.random.normal(0, 1, 100)

result = dp_linear_regression(
    X, y,
    x_bounds=[(0, 10), (0, 10)],
    y_bounds=(-5, 30),
    mu=1.0
)
print("Coefficients:", result.coefficients)
```
