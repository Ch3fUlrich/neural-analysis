# Repository Structure

## Current Structure

```
neural-analysis/
│
├── .github/
│   └── workflows/
│       └── ci.yml          # GitHub Actions CI pipeline
│
├── docs/                   # Documentation
│   ├── folder_structure.md # This file - Repository structure
│   ├── project_goal.md     # Project mission and goals
│   ├── python_testing_setup.md # Testing and quality guidelines
│   ├── local_ci_testing.md # Local CI with act guide
│   ├── setup_script_usage.md # Setup script comprehensive guide
│   └── claude.md           # Instructions for AI assistants
│
├── scripts/                # Automation scripts
│   ├── setup_env.sh        # Environment bootstrap script
│   └── run_ci_locally.sh   # Local CI runner with act
│
├── src/                    # Main package source code
│   └── neural_analysis/
│       ├── __init__.py
│       └── example.py      # Example module with utilities
│
├── tests/                  # Unit and integration tests
│   ├── test_example.py     # Tests for example module
│   └── test_placeholder.py # Initial placeholder test
│
├── .gitignore              # Git ignore patterns
├── .pre-commit-config.yaml # Pre-commit hooks configuration
├── .python-version         # Python version specification (3.14)
├── CONTRIBUTING.md         # Contribution guidelines
├── LICENSE                 # MIT License
├── README.md               # Project overview and quick start
├── pyproject.toml          # Project metadata and dependencies
└── uv.lock                 # Locked dependency versions

```

## Planned Structure (Future Expansion)

When the project grows, it will follow this structure:

```
neural_analysis_repo/
│
├── data/                   # Raw and processed data
│   ├── raw/                # Original unmodified datasets
│   ├── processed/          # Preprocessed datasets ready for analysis
│   └── external/           # External data or reference datasets
│
├── notebooks/              # Jupyter notebooks for exploration and demos
│   └── examples.ipynb
│
├── src/                    # All source code for analysis
│   ├── __init__.py
│   ├── utils/              # General utility functions (file IO, logging, etc.)
│   │   ├── __init__.py
│   │   ├── io_utils.py
│   │   └── math_utils.py
│   │
│   ├── preprocessing/      # Data cleaning, normalization, filtering
│   │   ├── __init__.py
│   │   └── signal_processing.py
│   │
│   ├── analysis/           # Core analysis methods
│   │   ├── __init__.py
│   │   ├── embedding.py    # Neural embedding / dimensionality reduction
│   │   ├── connectivity.py # Functional or structural connectivity analysis
│   │   └── spike_analysis.py
│   │
│   ├── plotting/           # Plotting functions / figure templates
│   │   ├── __init__.py
│   │   ├── raster_plot.py
│   │   └── summary_figures.py
│   │
│   └── models/             # Optional: ML/Deep Learning models
│       ├── __init__.py
│       ├── autoencoder.py
│       └── classifier.py
│
├── tests/                  # Unit tests for all modules
│   ├── __init__.py
│   ├── test_utils.py
│   ├── test_embedding.py
│   └── test_plotting.py
│
├── docs/                   # Documentation, methodology notes
│
├── results/                # Generated outputs (plots, embeddings, tables)
│   ├── figures/
│   └── tables/
│
├── requirements.txt        # Python dependencies
├── setup.py / pyproject.toml # Package info
├── README.md
└── .gitignore
```

## 🚨 Critical Workflow Rules

### NEVER Push Without CI Passing

**Before every push to GitHub:**
```bash
# MANDATORY - Run local CI
./scripts/run_ci_locally.sh

# If act/Docker not available, run checks manually:
uv run -- ruff check src tests
uv run -- mypy src tests
uv run -- pytest -v
```

**Why?**
- Catches issues before they reach GitHub
- Saves CI minutes
- Maintains code quality standards
- Prevents broken builds on main branch

### Branch Protection

- ✅ Main branch is protected
- ✅ Direct pushes to main are blocked
- ✅ All changes must go through pull requests
- ✅ CI must pass before merging
- ✅ Use feature branches: `feat/`, `fix/`, `chore/`

### Workflow Steps

1. **Create feature branch**
   ```bash
   git checkout -b feat/your-feature
   ```

2. **Make changes and test locally**
   ```bash
   # Make code changes
   # Add tests
   # Run local CI
   ./scripts/run_ci_locally.sh
   ```

3. **Commit changes**
   ```bash
   git add .
   git commit -m "feat: description"
   ```

4. **Push to feature branch**
   ```bash
   git push origin feat/your-feature
   ```

5. **Create pull request**
   - GitHub Actions will run automatically
   - All checks must pass
   - Review and merge when approved

## Project Goal
