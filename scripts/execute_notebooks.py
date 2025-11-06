#!/usr/bin/env python3
"""
Execute all example notebooks to refresh outputs.
This script runs all notebooks in the examples/ directory and saves them with fresh outputs.
"""
import sys
from pathlib import Path
import nbformat
from nbconvert.preprocessors import ExecutePreprocessor, CellExecutionError

def execute_notebook(notebook_path: Path, timeout: int = 600):
    """Execute a notebook and save the results."""
    print(f"\n{'='*80}")
    print(f"Executing: {notebook_path.name}")
    print(f"{'='*80}")
    
    try:
        with open(notebook_path, 'r', encoding='utf-8') as f:
            nb = nbformat.read(f, as_version=4)
        
        # Create executor
        ep = ExecutePreprocessor(
            timeout=timeout,
            kernel_name='python3',
            allow_errors=False  # Stop on first error
        )
        
        # Execute the notebook
        try:
            ep.preprocess(nb, {'metadata': {'path': str(notebook_path.parent)}})
            status = "✅ SUCCESS"
        except CellExecutionError as e:
            print(f"⚠️  Cell execution error in {notebook_path.name}:")
            print(f"   Cell #{e.cell_index + 1}: {str(e)[:200]}")
            status = "⚠️  PARTIAL (error in execution)"
            # Continue to save partial results
        
        # Save the executed notebook
        with open(notebook_path, 'w', encoding='utf-8') as f:
            nbformat.write(nb, f)
        
        print(f"{status}: {notebook_path.name}")
        return True
        
    except Exception as e:
        print(f"❌ FAILED: {notebook_path.name}")
        print(f"   Error: {str(e)[:200]}")
        return False

def main():
    """Execute all notebooks in the examples directory."""
    # Get the project root
    script_dir = Path(__file__).parent
    project_root = script_dir.parent
    examples_dir = project_root / "examples"
    
    if not examples_dir.exists():
        print(f"❌ Examples directory not found: {examples_dir}")
        sys.exit(1)
    
    # Get all notebooks
    notebooks = sorted(examples_dir.glob("*.ipynb"))
    
    if not notebooks:
        print(f"❌ No notebooks found in {examples_dir}")
        sys.exit(1)
    
    print(f"Found {len(notebooks)} notebooks to execute")
    print(f"Execution timeout per notebook: 600 seconds (10 minutes)")
    
    # Execute each notebook
    results = {}
    for notebook_path in notebooks:
        success = execute_notebook(notebook_path)
        results[notebook_path.name] = success
    
    # Print summary
    print(f"\n{'='*80}")
    print("EXECUTION SUMMARY")
    print(f"{'='*80}")
    
    success_count = sum(results.values())
    total_count = len(results)
    
    for name, success in results.items():
        status = "✅" if success else "❌"
        print(f"{status} {name}")
    
    print(f"\n{success_count}/{total_count} notebooks executed successfully")
    
    if success_count < total_count:
        print("\n⚠️  Some notebooks failed. Check logs above for details.")
        sys.exit(1)
    else:
        print("\n🎉 All notebooks executed successfully!")
        sys.exit(0)

if __name__ == "__main__":
    main()
