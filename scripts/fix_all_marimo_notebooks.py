#!/usr/bin/env python3
"""
Fix all marimo notebooks by removing duplicate mo imports and hidden cells.
"""

import re
from pathlib import Path


def remove_duplicate_mo_imports(content: str) -> str:
    """Remove all duplicate mo import cells, keeping only the header one."""
    lines = content.split('\n')
    result_lines = []
    i = 0
    header_found = False
    
    while i < len(lines):
        # Check if this is the header mo import cell (keep this one)
        if (i + 5 < len(lines) and 
            lines[i].strip() == '@app.cell(hide_code=True)' and
            lines[i+1].strip() == 'def __():' and
            'import marimo as mo' in '\n'.join(lines[i:i+10])):
            # This is the header cell - keep it
            j = i
            while j < len(lines) and not (j > i + 3 and lines[j].strip().startswith('@app.cell')):
                result_lines.append(lines[j])
                j += 1
                if j < len(lines) and lines[j-1].strip() == 'return mo':
                    break
            i = j
            header_found = True
        # Check if this is a duplicate mo import cell (remove it)
        elif (i + 3 < len(lines) and
              lines[i].strip().startswith('@app.cell') and
              (lines[i+1].strip().startswith('def _():') or 
               (i+2 < len(lines) and lines[i+2].strip().startswith('def _():'))) and
              'import marimo as mo' in '\n'.join(lines[i:i+10])):
            # This is a duplicate - skip it
            j = i
            while j < len(lines):
                if j > i + 3 and lines[j].strip().startswith('@app.cell'):
                    break
                if 'return' in lines[j] and ('mo' in lines[j] or 'mo,' in lines[j]):
                    j += 1
                    break
                j += 1
            i = j
        else:
            result_lines.append(lines[i])
            i += 1
    
    return '\n'.join(result_lines)


def remove_hidden_cells_except_header(content: str) -> str:
    """Remove all hide_code=True cells except the header mo import cell."""
    lines = content.split('\n')
    result_lines = []
    i = 0
    
    while i < len(lines):
        line = lines[i].strip()
        # Check if this is a hide_code=True cell
        if line == '@app.cell(hide_code=True)':
            # Check if it's the header cell (def __())
            if i + 1 < len(lines) and lines[i+1].strip() == 'def __():':
                # This is the header - keep it
                j = i
                while j < len(lines):
                    result_lines.append(lines[j])
                    if j > i + 3 and lines[j].strip().startswith('@app.cell'):
                        break
                    if 'return mo' in lines[j]:
                        j += 1
                        break
                    j += 1
                i = j
            else:
                # This is a non-header hide_code cell - remove hide_code=True
                result_lines.append('@app.cell')
                i += 1
        else:
            result_lines.append(lines[i])
            i += 1
    
    return '\n'.join(result_lines)


def fix_notebook(notebook_path: Path) -> bool:
    """Fix a single notebook file."""
    try:
        with open(notebook_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        original_content = content
        
        # Remove duplicate mo imports
        content = remove_duplicate_mo_imports(content)
        
        # Remove hidden cells except header
        content = remove_hidden_cells_except_header(content)
        
        # Only write if content changed
        if content != original_content:
            with open(notebook_path, 'w', encoding='utf-8') as f:
                f.write(content)
            return True
        return False
    except Exception as e:
        print(f"Error fixing {notebook_path}: {e}")
        return False


def main():
    """Fix all marimo notebooks in examples directory."""
    examples_dir = Path(__file__).parent.parent / 'examples'
    notebooks = list(examples_dir.glob('*_marimo_nb.py'))
    
    print(f"Found {len(notebooks)} marimo notebooks to fix...")
    
    fixed_count = 0
    for nb in notebooks:
        print(f"Fixing {nb.name}...")
        if fix_notebook(nb):
            fixed_count += 1
            print(f"  ✓ Fixed {nb.name}")
        else:
            print(f"  - No changes needed for {nb.name}")
    
    print(f"\n✓ Fixed {fixed_count} out of {len(notebooks)} notebooks")


if __name__ == '__main__':
    main()

