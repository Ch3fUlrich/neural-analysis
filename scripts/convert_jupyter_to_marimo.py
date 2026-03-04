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
    python scripts/convert_jupyter_to_marimo.py input.ipynb [output.py] [--overwrite]
    python scripts/convert_jupyter_to_marimo.py examples/metrics_examples.ipynb
    # Output will be automatically generated as: examples/metrics_examples_marimo_nb.py

After conversion, export the notebook to HTML with outputs:
    uv run marimo export html examples/notebook_marimo_nb.py -o examples/__marimo__/notebook_marimo_nb.html

Note: All exported HTML notebooks with outputs are saved to examples/__marimo__/
      This directory is automatically created by marimo when exporting.
"""

import json
import re
import subprocess
import sys
from pathlib import Path


def extract_markdown_cells(ipynb_path: Path) -> list[tuple[int, str]]:
    """
    Extract all markdown cells from Jupyter notebook.

    Args:
        ipynb_path: Path to the Jupyter notebook (.ipynb file)

    Returns:
        List of tuples (cell_index, markdown_content)
    """
    with open(ipynb_path, encoding="utf-8") as f:
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
    content: str, markdown_cells: list[tuple[int, str]]
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


def remove_duplicate_mo_imports(content: str) -> str:
    """
    Remove all duplicate mo import cells, keeping only the header one.

    Args:
        content: The marimo notebook content

    Returns:
        Updated content with duplicate mo imports removed
    """
    lines = content.split("\n")
    result_lines = []
    i = 0

    while i < len(lines):
        # Check if this is the header mo import cell (keep this one)
        if (
            i + 5 < len(lines)
            and lines[i].strip() == "@app.cell(hide_code=True)"
            and lines[i + 1].strip() == "def __():"
            and "import marimo as mo" in "\n".join(lines[i : i + 10])
        ):
            # This is the header cell - keep it
            # Find the end of this cell (next @app.cell or end of function)
            j = i
            while j < len(lines) and not (
                j > i + 3 and lines[j].strip().startswith("@app.cell")
            ):
                result_lines.append(lines[j])
                j += 1
                if j < len(lines) and lines[j - 1].strip() == "return mo":
                    break
            i = j
        # Check if this is a duplicate mo import cell (remove it)
        elif (
            i + 3 < len(lines)
            and lines[i].strip().startswith("@app.cell")
            and (
                lines[i + 1].strip().startswith("def _():")
                or (i + 2 < len(lines) and lines[i + 2].strip().startswith("def _():"))
            )
            and "import marimo as mo" in "\n".join(lines[i : i + 10])
        ):
            # This is a duplicate - skip it
            # Find the end of this cell
            j = i
            while j < len(lines):
                if j > i + 3 and lines[j].strip().startswith("@app.cell"):
                    break
                if "return" in lines[j] and ("mo" in lines[j] or "mo," in lines[j]):
                    j += 1
                    break
                j += 1
            i = j
        else:
            result_lines.append(lines[i])
            i += 1

    return "\n".join(result_lines)


def remove_hidden_cells_except_header(content: str) -> str:
    """
    Remove all hide_code=True cells except the header mo import cell.

    Args:
        content: The marimo notebook content

    Returns:
        Updated content with hidden cells removed (except header)
    """
    lines = content.split("\n")
    result_lines = []
    i = 0

    while i < len(lines):
        line = lines[i].strip()
        # Check if this is a hide_code=True cell
        if line == "@app.cell(hide_code=True)":
            # Check if it's the header cell (def __())
            if i + 1 < len(lines) and lines[i + 1].strip() == "def __():":
                # This is the header - keep it
                j = i
                while j < len(lines):
                    result_lines.append(lines[j])
                    if j > i + 3 and lines[j].strip().startswith("@app.cell"):
                        break
                    if "return mo" in lines[j]:
                        j += 1
                        break
                    j += 1
                i = j
            else:
                # This is a non-header hide_code cell - remove hide_code=True
                result_lines.append("@app.cell")
                i += 1
        else:
            result_lines.append(lines[i])
            i += 1

    return "\n".join(result_lines)


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
        subprocess.run(
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
    except FileNotFoundError as e:
        raise RuntimeError(
            "marimo command not found. Install marimo: pip install marimo"
        ) from e

    # Step 3: Read the converted file
    print("  Reading converted notebook...")
    with open(output_path, encoding="utf-8") as f:
        content = f.read()

    # Step 4: Replace the beginning with the standard structure
    print("  Setting up standard notebook header...")
    content = ensure_standard_header(content)

    # Step 5: Remove duplicate mo imports (keep only header)
    print("  Removing duplicate mo imports...")
    content = remove_duplicate_mo_imports(content)

    # Step 6: Remove hidden cells except header
    print("  Removing hidden cells (except header)...")
    content = remove_hidden_cells_except_header(content)

    # Step 7: Replace empty cells with markdown (after mo import is ensured)
    print("  Replacing empty cells with markdown...")
    content = replace_empty_cells_with_markdown(content, markdown_cells)

    # Step 8: Remove unnecessary blocks
    print("  Cleaning up unnecessary code blocks...")
    content = remove_unnecessary_blocks(content)

    # Step 7: Write the final file
    print("  Writing final notebook...")
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(content)

    print(f"✓ Successfully converted {ipynb_path} to {output_path}")
    print(f"  - Added {len(markdown_cells)} markdown cells")
    print("  - Preserved notebook structure")


def main() -> None:
    """Main entry point for the conversion script."""
    if len(sys.argv) < 2:
        print(
            "Usage: python scripts/convert_jupyter_to_marimo.py <input.ipynb> [output.py] [--overwrite]"
        )
        print("\nExample:")
        print(
            "  python scripts/convert_jupyter_to_marimo.py examples/metrics_examples.ipynb"
        )
        print(
            "  python scripts/convert_jupyter_to_marimo.py examples/metrics_examples.ipynb examples/metrics_examples_marimo_nb.py"
        )
        sys.exit(1)

    ipynb_path = Path(sys.argv[1])

    # Auto-generate output filename if not provided
    if len(sys.argv) >= 3 and not sys.argv[2].startswith("--"):
        output_path = Path(sys.argv[2])
    else:
        # Generate output filename: {original_name}_marimo_nb.py
        output_path = ipynb_path.parent / f"{ipynb_path.stem}_marimo_nb.py"

    overwrite = "--overwrite" in sys.argv or "-f" in sys.argv

    try:
        convert_jupyter_to_marimo(ipynb_path, output_path, overwrite=overwrite)
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
