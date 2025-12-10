#!/usr/bin/env python3
"""
Automated script to convert Jupyter notebooks (.ipynb) to marimo notebooks (.py).

This script:
1. Uses marimo's built-in conversion tool
2. Extracts all markdown cells from the Jupyter notebook
3. Replaces empty cells in the marimo notebook with proper markdown cells
4. Ensures mo is properly imported
5. Removes unnecessary code blocks
6. Preserves the full structure of the original notebook

Usage:
    python scripts/convert_jupyter_to_marimo.py input.ipynb output.py
    python scripts/convert_jupyter_to_marimo.py examples/metrics_examples.ipynb examples/metrics_examples_marimo_nb.py
"""

import json
import re
import subprocess
import sys
from pathlib import Path
from typing import List, Tuple


def extract_markdown_cells(ipynb_path: Path) -> List[Tuple[int, str]]:
    """
    Extract all markdown cells from Jupyter notebook.

    Args:
        ipynb_path: Path to the Jupyter notebook (.ipynb file)

    Returns:
        List of tuples (cell_index, markdown_content)
    """
    with open(ipynb_path, "r", encoding="utf-8") as f:
        nb = json.load(f)

    markdown_cells = []
    for i, cell in enumerate(nb["cells"]):
        if cell["cell_type"] == "markdown":
            source = "".join(cell["source"])
            markdown_cells.append((i, source))

    return markdown_cells


def ensure_standard_header(content: str) -> str:
    """
    Ensure the notebook starts with the standard header structure:
    - import marimo
    - __generated_with = "0.18.3"
    - app = marimo.App()
    - @app.cell with mo import (using __() function name and return mo)

    Args:
        content: The marimo notebook content

    Returns:
        Updated content with standard header
    """
    # Standard header structure (exact format as specified)
    standard_header = """import marimo


__generated_with = "0.18.3"

app = marimo.App(width="full")

@app.cell(hide_code=True)
def __():

    import marimo as mo

    return mo

"""

    # Find where the actual cells start (after app creation and any initial cells)
    # Look for the first @app.cell that is NOT the mo import cell
    app_match = re.search(r"app = marimo\.App\(\)\s*\n", content)
    if not app_match:
        # If no app creation found, prepend the entire header
        return standard_header + content

    # Find the first content cell (not the mo import)
    after_app = app_match.end()

    # Look for the first @app.cell that uses mo (markdown or other cells)
    first_content_cell = re.search(r"@app\.cell", content[after_app:])
    if first_content_cell:
        # Replace everything from the start up to (but not including) the first content cell
        insert_pos = after_app + first_content_cell.start()
        return standard_header + content[insert_pos:]
    else:
        # No cells found, just replace the beginning
        return standard_header + content[after_app:]


def ensure_mo_import(content: str) -> str:
    """
    Ensure marimo is imported as mo and returned from a cell.
    Always adds the import cell right after app creation, before any other cells.
    This ensures mo is always available for markdown cells and other usage.

    Args:
        content: The marimo notebook content

    Returns:
        Updated content with mo import (always added if missing)
    """
    # Find the position right after app creation
    app_creation_match = re.search(r"app = marimo\.App\(\)\s*\n", content)
    if not app_creation_match:
        return content

    after_app = app_creation_match.end()

    # Remove ALL existing mo import cells (we'll add it in the right place)
    # Use a precise pattern that matches the exact cell structure
    mo_import_cell_pattern = r"@app\.cell\s*\n\s*def _\(\):\s*\n\s*import marimo as mo\s*\n\s*return \(mo,\)\s*\n"
    # Remove all occurrences
    while re.search(mo_import_cell_pattern, content, re.MULTILINE):
        content = re.sub(
            mo_import_cell_pattern, "\n", content, flags=re.MULTILINE, count=1
        )

    # Check if mo import is already in the right place (right after app creation)
    check_region = content[after_app : after_app + 200]
    if "import marimo as mo" in check_region and "@app.cell" in check_region:
        # Already in the right place
        return content

    # Find the first cell after app creation
    first_cell_match = re.search(r"@app\.cell", content[after_app:])

    # Insert mo import cell right after app creation, before any other cells
    mo_import_cell = """

@app.cell
def _():
    import marimo as mo

    return (mo,)

"""
    if first_cell_match:
        # Insert before first cell
        insert_pos = after_app + first_cell_match.start()
        content = content[:insert_pos] + mo_import_cell + content[insert_pos:]
    else:
        # No cells found, just add after app creation
        pattern = r"(app = marimo\.App\(\)\s*\n)"
        replacement = r"""\1""" + mo_import_cell
        content = re.sub(pattern, replacement, content)

    return content


def replace_empty_cells_with_markdown(
    content: str, markdown_cells: List[Tuple[int, str]]
) -> str:
    """
    Replace empty cells in marimo notebook with markdown cells.

    Args:
        content: The marimo notebook content
        markdown_cells: List of (index, markdown_content) tuples

    Returns:
        Updated content with markdown cells
    """
    # Find all empty cells: @app.cell followed by def _(): followed by return
    # Match with multiline to handle newlines properly
    pattern = r"(@app\.cell\s*\n\s*def _\(\):\s*\n\s*return\s*\n\s*\n)"
    matches = list(re.finditer(pattern, content, re.MULTILINE))

    if not matches:
        return content

    # Replace in reverse order to preserve positions
    replaced_count = 0
    for i, match in enumerate(reversed(matches)):
        if i < len(markdown_cells):
            md_content = markdown_cells[len(markdown_cells) - 1 - i][1]

            # Escape triple quotes in markdown if present
            if '"""' in md_content:
                md_content = md_content.replace('"""', '\\"\\"\\"')

            # Create markdown cell - note: mo.md() is called but NOT returned
            replacement = f'''@app.cell
def _(mo):
    mo.md(
        r"""
{md_content}
"""
    )
    return

'''
            start, end = match.span()
            content = content[:start] + replacement + content[end:]
            replaced_count += 1

    return content


def remove_unnecessary_blocks(content: str) -> str:
    """
    Remove unnecessary code blocks from marimo notebook.

    Args:
        content: The marimo notebook content

    Returns:
        Updated content with unnecessary blocks removed
    """
    # Remove if __name__ == "__main__" blocks (handle various formats)
    # Pattern 1: Complete block at end of file
    content = re.sub(
        r'\n\nif __name__ == "__main__":\s*app\.run\(\)\s*$',
        "",
        content,
        flags=re.MULTILINE,
    )
    # Pattern 2: Broken/incomplete blocks
    content = re.sub(
        r'\n\nif __name__ == "__main__"\s*\n',
        "\n",
        content,
    )
    content = re.sub(
        r"\n:\s*app\.run\(\)\s*$",
        "",
        content,
        flags=re.MULTILINE,
    )

    # Remove any __mo imports that marimo convert might add incorrectly
    content = re.sub(r"import marimo as __mo\s*\n", "", content)

    return content


def convert_jupyter_to_marimo(
    ipynb_path: Path, output_path: Path, overwrite: bool = False
) -> None:
    """
    Convert Jupyter notebook to marimo notebook format.

    Args:
        ipynb_path: Path to input Jupyter notebook (.ipynb)
        output_path: Path to output marimo notebook (.py)
        overwrite: Whether to overwrite existing output file
    """
    if not ipynb_path.exists():
        raise FileNotFoundError(f"Input notebook not found: {ipynb_path}")

    if output_path.exists() and not overwrite:
        raise FileExistsError(
            f"Output file already exists: {output_path}. "
            "Use overwrite=True to replace it."
        )

    print(f"Converting {ipynb_path} to {output_path}...")

    # Step 1: Extract markdown cells from Jupyter notebook
    print("  Extracting markdown cells from Jupyter notebook...")
    markdown_cells = extract_markdown_cells(ipynb_path)
    print(f"  Found {len(markdown_cells)} markdown cells")

    # Step 2: Use marimo convert to create initial marimo notebook
    print("  Running marimo convert...")
    try:
        result = subprocess.run(
            ["marimo", "convert", str(ipynb_path), "-o", str(output_path)],
            capture_output=True,
            text=True,
            check=True,
        )
    except subprocess.CalledProcessError as e:
        raise RuntimeError(
            f"marimo convert failed: {e.stderr}\n"
            "Make sure marimo is installed: pip install marimo"
        ) from e
    except FileNotFoundError:
        raise RuntimeError(
            "marimo command not found. " "Install marimo: pip install marimo"
        )

    # Step 3: Read the converted file
    print("  Reading converted notebook...")
    with open(output_path, "r", encoding="utf-8") as f:
        content = f.read()

    # Step 4: Replace the beginning with the standard structure
    print("  Setting up standard notebook header...")
    content = ensure_standard_header(content)

    # Step 5: Replace empty cells with markdown (after mo import is ensured)
    print("  Replacing empty cells with markdown...")
    content = replace_empty_cells_with_markdown(content, markdown_cells)

    # Step 6: Remove unnecessary blocks
    print("  Cleaning up unnecessary code blocks...")
    content = remove_unnecessary_blocks(content)

    # Step 7: Write the final file
    print("  Writing final notebook...")
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(content)

    print(f"✓ Successfully converted {ipynb_path} to {output_path}")
    print(f"  - Added {len(markdown_cells)} markdown cells")
    print(f"  - Preserved notebook structure")


def main():
    """Main entry point for the conversion script."""
    if len(sys.argv) < 3:
        print(
            "Usage: python scripts/convert_jupyter_to_marimo.py <input.ipynb> <output.py> [--overwrite]"
        )
        print("\nExample:")
        print(
            "  python scripts/convert_jupyter_to_marimo.py examples/metrics_examples.ipynb examples/metrics_examples_marimo_nb.py"
        )
        sys.exit(1)

    ipynb_path = Path(sys.argv[1])
    output_path = Path(sys.argv[2])
    overwrite = "--overwrite" in sys.argv or "-f" in sys.argv

    try:
        convert_jupyter_to_marimo(ipynb_path, output_path, overwrite=overwrite)
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
