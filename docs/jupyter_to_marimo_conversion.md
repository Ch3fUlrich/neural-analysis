# Jupyter to Marimo Notebook Conversion Guide

This guide explains how to automatically convert a Jupyter notebook (`.ipynb`) to a marimo notebook (`.py`).

## Overview

Marimo notebooks are Python files that use decorators (`@app.cell`) to define cells. Unlike Jupyter, marimo has a strict execution model where:
- All cells are Python code cells
- Markdown is displayed using `mo.md()` function calls
- Variables must be explicitly returned from cells to be available in other cells
- No `if __name__ == "__main__": app.run()` block is needed

## Conversion Steps

### 1. Initial Conversion

Use marimo's built-in conversion tool:

```bash
marimo convert examples/metrics_examples.ipynb examples/metrics_examples_marimo_nb.py
```

This creates a basic marimo notebook structure.

### 2. Fix Markdown Cells

**Critical**: Markdown cells should **call** `mo.md()` but **not return** it. The pattern is:

**Correct:**
```python
@app.cell
def _(mo):
    mo.md(r"""# Title""")
    return
```

**Wrong:**
```python
@app.cell
def _(mo):
    return mo.md(r"""# Title""")  # Don't return mo.md()
```

### 3. Extract and Copy Markdown Content

Extract all markdown cells from the Jupyter notebook:

```python
import json

with open('notebook.ipynb', 'r') as f:
    nb = json.load(f)

markdown_cells = []
for i, cell in enumerate(nb['cells']):
    if cell['cell_type'] == 'markdown':
        source = ''.join(cell['source'])
        markdown_cells.append((i, source))
```

### 4. Replace Empty Cells with Markdown

Find all empty cells in the marimo notebook:
```python
@app.cell
def _():
    return
```

Replace them with proper markdown cells:
```python
@app.cell
def _(mo):
    mo.md(r"""
# Your markdown content here
""")
    return
```

### 5. Fix Variable Scoping

- **Local variables**: Prefix with `_` (e.g., `_fig`, `_grid`, `_plot_specs`)
- **Shared variables**: Return them explicitly from cells
- **Loop variables**: Make them private if used across cells

### 6. Fix Imports

- Ensure `marimo` is imported at the top: `import marimo`
- Create a cell that imports and returns `mo`:
  ```python
  @app.cell
  def _():
      import marimo as mo
      return (mo,)
  ```
- The `mo` variable must be returned as a tuple `(mo,)` to be available to other cells

### 7. Remove Unnecessary Code

- Remove `if __name__ == "__main__": app.run()` blocks
- Remove duplicate imports
- Remove any `__mo` imports that marimo convert might add

### 8. Test the Notebook

```bash
# Check for errors
marimo check examples/metrics_examples_marimo_nb.py

# Run the notebook
marimo run examples/metrics_examples_marimo_nb.py
```

## Common Issues and Fixes

### Issue 1: Markdown Not Displaying

**Problem**: Markdown cells show nothing

**Solution**: Ensure you're **calling** `mo.md()` but **not returning** it:
```python
mo.md(r"""...""")         # ✓ Correct - call mo.md()
return                     # ✓ Correct - return nothing
return mo.md(r"""...""")  # ✗ Wrong - don't return mo.md()
```

### Issue 2: Multiple Definitions Error

**Problem**: `critical[multiple-definitions]: Variable 'x' is defined in multiple cells`

**Solution**: Make local variables private by prefixing with `_`:
```python
_fig = ...  # Private to this cell
grid = ...  # Shared across cells (return it)
```

### Issue 3: Import Errors

**Problem**: `ImportError: cannot import name 'X' from 'module'`

**Solution**: 
- Check that the function is exported in `__init__.py`
- Add to `__all__` list
- Add to lazy import handler if using lazy imports

### Issue 4: Variable Not Found

**Problem**: `NameError: name 'x' is not defined`

**Solution**: Ensure the variable is returned from the cell where it's defined:
```python
@app.cell
def _():
    x = 42
    return x  # Must return to make available to other cells
```

## Automated Conversion Script

A complete automated conversion script is available at `scripts/convert_jupyter_to_marimo.py`.

**Usage:**
```bash
python scripts/convert_jupyter_to_marimo.py input.ipynb output.py
python scripts/convert_jupyter_to_marimo.py input.ipynb output.py --overwrite
```

**Features:**
- Automatically extracts all markdown cells from Jupyter notebook
- Uses marimo's built-in conversion tool
- Replaces empty cells with proper markdown cells
- Ensures `mo` is properly imported
- Removes unnecessary code blocks
- Preserves the full structure of the original notebook

**Example:**
```bash
python scripts/convert_jupyter_to_marimo.py \
    examples/metrics_examples.ipynb \
    examples/metrics_examples_marimo_nb.py
```

The script handles:
1. Extracting markdown cells from `.ipynb` files
2. Running `marimo convert` to create the initial structure
3. Replacing empty cells with markdown cells using the correct pattern:
   ```python
   @app.cell
   def _(mo):
       mo.md(r"""...""")
       return
   ```
4. Ensuring `mo` is imported and returned as `(mo,)`
5. Cleaning up unnecessary blocks

## Testing Checklist

- [ ] Run `marimo check` - no critical errors
- [ ] All markdown cells display correctly
- [ ] All code cells execute without errors
- [ ] Variables are properly scoped (no multiple-definitions errors)
- [ ] Imports work correctly
- [ ] Notebook runs with `marimo run`

## Key Differences from Jupyter

| Feature | Jupyter | Marimo |
|---------|---------|--------|
| Cell types | Code, Markdown, Raw | All are Python cells |
| Markdown | Separate cell type | `return mo.md(...)` |
| Variable scope | Global by default | Must return to share |
| Execution | Run cells independently | Reactive execution model |
| File format | `.ipynb` (JSON) | `.py` (Python) |

## References

- [Marimo Documentation](https://docs.marimo.io/)
- [Marimo Markdown API](https://docs.marimo.io/api/markdown/#marimo.md)

