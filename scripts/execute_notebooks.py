#!/usr/bin/env python3
"""
Execute all example notebooks to verify changes work correctly.

This script runs all notebooks in the examples/ directory and can optionally
save them with fresh outputs. Useful for:
- Verifying code changes don't break notebooks
- Refreshing notebook outputs
- CI/CD testing
"""

import argparse
import sys
from pathlib import Path

try:
    import nbformat
    from nbconvert.preprocessors import CellExecutionError, ExecutePreprocessor
except ImportError:
    print(
        "ERROR: nbformat and nbconvert required. Install with: pip install nbformat nbconvert"
    )
    sys.exit(1)


def execute_notebook(
    notebook_path: Path,
    timeout: int = 600,
    save_output: bool = False,
    allow_errors: bool = False,
) -> bool:
    """Execute a Jupyter notebook and optionally save the results.

    Parameters
    ----------
    notebook_path : Path
        Path to the notebook file
    timeout : int, default=600
        Execution timeout per notebook in seconds
    save_output : bool, default=False
        If True, save the executed notebook with outputs
        If False, execute but don't modify the original file
    allow_errors : bool, default=False
        If True, continue execution even if a cell fails
        If False, stop on first error

    Returns
    -------
    bool
        True if execution succeeded (or partially succeeded with allow_errors=True)
        False if execution failed completely
    """
    print(f"\n{'=' * 60}")
    print(f"Executing: {notebook_path.name}")
    print(f"{'=' * 60}")

    if not notebook_path.exists():
        print(f"ERROR: Notebook not found: {notebook_path}")
        return False

    try:
        # Read notebook
        with open(notebook_path, encoding="utf-8") as f:
            nb = nbformat.read(f, as_version=4)

        # Create executor
        ep = ExecutePreprocessor(
            timeout=timeout,
            kernel_name="python3",
            allow_errors=allow_errors,
        )

        # Execute the notebook
        try:
            ep.preprocess(nb, {"metadata": {"path": str(notebook_path.parent)}})
            status = "✓ SUCCESS"
            success = True
        except CellExecutionError as e:
            print(f"⚠️  Cell execution error in {notebook_path.name}:")
            # Handle different exception formats
            cell_info = ""
            if hasattr(e, "cell_index"):
                cell_info = f"Cell #{e.cell_index + 1}: "
            elif hasattr(e, "traceback") and e.traceback:
                # Try to extract cell info from traceback
                cell_info = "Cell execution failed: "
            else:
                cell_info = "Cell execution failed: "
            print(f"   {cell_info}{str(e)[:200]}")
            status = "⚠️  PARTIAL (error in execution)"
            success = allow_errors  # Only count as success if errors allowed

        # Save the executed notebook if requested
        if save_output:
            with open(notebook_path, "w", encoding="utf-8") as f:
                nbformat.write(nb, f)
            print(f"{status}: {notebook_path.name} (saved with outputs)")
        else:
            print(f"{status}: {notebook_path.name}")

        return success

    except Exception as e:
        print(f"✗ ERROR executing {notebook_path.name}: {e}")
        import traceback

        traceback.print_exc()
        return False


def main() -> int:
    """Execute all notebooks in the examples directory or specified notebooks.

    Returns
    -------
    int
        Exit code: 0 if all notebooks passed, 1 otherwise
    """
    parser = argparse.ArgumentParser(
        description="Execute Jupyter notebooks to verify changes work correctly"
    )
    parser.add_argument(
        "--notebooks",
        nargs="+",
        type=str,
        help="Specific notebooks to execute (default: all in examples/)",
    )
    parser.add_argument(
        "--timeout",
        type=int,
        default=600,
        help="Execution timeout per notebook in seconds (default: 600)",
    )
    parser.add_argument(
        "--save-output",
        action="store_true",
        help="Save executed notebooks with outputs (default: False)",
    )
    parser.add_argument(
        "--allow-errors",
        action="store_true",
        help="Continue execution even if a cell fails (default: False)",
    )
    parser.add_argument(
        "--examples-dir",
        type=str,
        default=None,
        help="Directory containing notebooks (default: examples/)",
    )

    args = parser.parse_args()

    # Get the project root
    script_dir = Path(__file__).parent
    project_root = script_dir.parent

    # Determine which notebooks to execute
    if args.notebooks:
        # Use specified notebooks
        notebooks = []
        for nb_name in args.notebooks:
            nb_path = project_root / "examples" / nb_name
            if not nb_path.exists():
                # Try relative to project root
                nb_path = project_root / nb_name
            if not nb_path.exists():
                print(f"ERROR: Notebook not found: {nb_name}")
                return 1
            notebooks.append(nb_path)
    else:
        # Find all notebooks in examples directory
        examples_dir = (
            project_root / args.examples_dir
            if args.examples_dir
            else project_root / "examples"
        )

        if not examples_dir.exists():
            print(f"ERROR: Examples directory not found: {examples_dir}")
            return 1

        notebooks = sorted(examples_dir.glob("*.ipynb"))

        if not notebooks:
            print(f"ERROR: No notebooks found in {examples_dir}")
            return 1

    print(f"Found {len(notebooks)} notebook(s) to execute")
    print(f"Execution timeout per notebook: {args.timeout} seconds")
    if args.save_output:
        print(
            "⚠️  Will save executed notebooks with outputs (original files will be modified)"
        )

    # Execute each notebook
    results = []
    for nb_path in notebooks:
        success = execute_notebook(
            nb_path,
            timeout=args.timeout,
            save_output=args.save_output,
            allow_errors=args.allow_errors,
        )
        results.append((nb_path.name, success))

    # Print summary
    print(f"\n{'=' * 60}")
    print("Execution Summary")
    print(f"{'=' * 60}")
    for name, success in results:
        status = "✓ PASSED" if success else "✗ FAILED"
        print(f"{status}: {name}")

    success_count = sum(1 for _, success in results if success)
    total_count = len(results)
    print(f"\n{success_count}/{total_count} notebook(s) executed successfully")

    if success_count < total_count:
        print("\n❌ Some notebooks failed to execute")
        return 1
    else:
        print("\n✅ All notebooks executed successfully!")
        return 0


if __name__ == "__main__":
    sys.exit(main())
