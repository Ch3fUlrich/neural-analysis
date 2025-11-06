# numba and umap-learn Python 3.14 Compatibility

## Status: BLOCKED - Waiting for upstream package updates

## Problem
Python 3.14 is too new for the numba/llvmlite ecosystem. Pre-built wheels are not available for cp314 (CPython 3.14).

## Root Cause
- `llvmlite 0.45.1` has no pre-built wheels for Python 3.14
- Building from source requires **LLVM 20**
- Ubuntu apt repositories only have up to LLVM 18 (stable)
- Dependency chain: `umap-learn` → `pynndescent` → `llvmlite`

## Attempted Solutions
1. ❌ Building with LLVM 14 - Rejected by cmake (needs LLVM 20)
2. ❌ Building with LLVM 18 - Not tested yet, but likely incompatible
3. ❌ Removing dependencies - User needs numba for performance-critical computations

## Impact
- `numba>=0.59` - Commented out temporarily
- `umap-learn>=0.5.3` - Commented out temporarily
- Code already handles optional imports gracefully (see `src/neural_analysis/metrics/similarity.py` and `distance.py`)

## Solutions (in priority order)

### Option 1: Wait for upstream wheels (RECOMMENDED)
- Wait for numba/llvmlite maintainers to publish Python 3.14 wheels
- Estimated timeline: 1-3 months after Python 3.14 stable release
- **Action**: Monitor PyPI for updates

### Option 2: Downgrade to Python 3.12
- Python 3.12 has pre-built wheels for all dependencies
- Lose Python 3.14 features but CI passes immediately
- **Action**: Change `python-version: ["3.14"]` to `["3.12"]` in `.github/workflows/ci.yml`

### Option 3: Build LLVM 20 from source in CI
- Very slow (LLVM compile time: 30-60 minutes)
- Complex setup, increases CI complexity
- Not recommended for regular CI runs
- **Action**: Create separate workflow for testing with custom LLVM build

### Option 4: Use older llvmlite version
- Find llvmlite version compatible with LLVM 18 or 14
- May require downgrading numba as well
- Risk: Incompatibility with other dependencies
- **Action**: Research llvmlite version history and LLVM requirements

## Related Files
- `pyproject.toml` - Dependencies commented out with TODO
- `.github/workflows/ci.yml` - Only cmake installed (LLVM removed)
- `src/neural_analysis/metrics/similarity.py` - Graceful degradation implemented
- `src/neural_analysis/metrics/distance.py` - Graceful degradation implemented

## Next Steps
1. Monitor llvmlite PyPI releases for Python 3.14 wheels
2. When available, uncomment dependencies in `pyproject.toml`
3. Run `uv lock` to update lock file
4. Test CI workflow passes
5. Verify performance improvements from numba JIT

## References
- llvmlite PyPI: https://pypi.org/project/llvmlite/
- numba PyPI: https://pypi.org/project/numba/
- LLVM releases: https://apt.llvm.org/
