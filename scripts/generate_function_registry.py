#!/usr/bin/env python3
"""
Automated Function Registry Generator

This script scans all Python files in the neural_analysis package and generates
a comprehensive function registry that can be used by AI agents and developers
to understand what functions are available and avoid code duplication.

Scans modules: plotting, embeddings, metrics, utils, decoding, synthetic_data

Usage:
    python scripts/generate_function_registry.py
"""

import ast
from pathlib import Path
from typing import Any


class FunctionVisitor(ast.NodeVisitor):
    """AST visitor to extract function definitions and their signatures."""

    def __init__(self, module_name: str) -> None:
        self.module_name = module_name
        self.functions: list[dict[str, Any]] = []

    def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
        """Extract function definition information."""
        # Skip private functions (except __init__)
        if node.name.startswith("_") and node.name != "__init__":
            self.generic_visit(node)
            return

        # Extract docstring
        docstring = ast.get_docstring(node)
        summary = ""
        if docstring:
            # Get first line of docstring as summary
            summary = docstring.strip().split("\n")[0]

        # Extract parameters
        params = []
        for arg in node.args.args:
            if arg.arg != "self" and arg.arg != "cls":
                params.append(arg.arg)

        # Extract return type annotation if present
        return_type = None
        if node.returns:
            return_type = (
                ast.unparse(node.returns) if hasattr(ast, "unparse") else "Any"
            )

        # Store function info
        func_info = {
            "name": node.name,
            "module": self.module_name,
            "parameters": params,
            "return_type": return_type,
            "summary": summary,
            "is_async": isinstance(node, ast.AsyncFunctionDef),
            "line": node.lineno,
        }

        self.functions.append(func_info)
        self.generic_visit(node)


def scan_module(file_path: Path, base_path: Path) -> list[dict[str, Any]]:
    """Scan a Python module and extract function definitions."""
    try:
        with open(file_path, encoding="utf-8") as f:
            source = f.read()

        tree = ast.parse(source)

        # Get relative module name
        rel_path = file_path.relative_to(base_path)
        module_parts = list(rel_path.parts[:-1]) + [rel_path.stem]
        module_name = ".".join(module_parts)

        visitor = FunctionVisitor(module_name)
        visitor.visit(tree)

        return visitor.functions
    except Exception as e:
        print(f"Error processing {file_path}: {e}")
        return []


def generate_registry(neural_analysis_dir: Path) -> dict[str, list[dict[str, Any]]]:
    """Generate function registry for all files in neural_analysis directory."""
    registry = {}

    # Scan all Python files
    for py_file in neural_analysis_dir.rglob("*.py"):
        if "__pycache__" in str(py_file):
            continue

        functions = scan_module(py_file, neural_analysis_dir)
        if functions:
            module_name = functions[0]["module"]
            registry[module_name] = functions

    return registry


def format_markdown_registry(registry: dict[str, list[dict[str, Any]]]) -> str:
    """Format registry as markdown documentation."""
    lines = [
        "# Neural Analysis Function Registry",
        "",
        "Auto-generated function registry for all public functions in "
        "neural_analysis package.",
        "**Last Updated:** Auto-generated",
        "",
        "## Purpose",
        "",
        "This registry helps developers and AI agents:",
        "- Avoid code duplication by finding existing functions",
        "- Understand the package structure and available functionality",
        "- Quickly locate the right function for a task",
        "",
        "---",
        "",
    ]

    # Organize by category
    categories = {
        "Synthetic Data": ["synthetic_data"],
        "Embeddings": ["embeddings"],
        "Decoding": ["decoding"],
        "Metrics - Distance": ["metrics.distance"],
        "Metrics - Distributions": ["metrics.distributions"],
        "Metrics - Similarity": ["metrics.similarity"],
        "Metrics - Outliers": ["metrics.outliers"],
        "Utils - IO": ["utils.io", "utils.io_h5io"],
        "Utils - Preprocessing": ["utils.preprocessing"],
        "Utils - Validation": ["utils.validation"],
        "Utils - Trajectories": ["utils.trajectories"],
        "Plotting - Core System": [
            "plotting.grid_config",
            "plotting.core",
            "plotting.backend",
        ],
        "Plotting - Renderers": ["plotting.renderers"],
        "Plotting - Statistical": ["plotting.statistical_plots"],
        "Plotting - 1D/2D/3D": [
            "plotting.plots_1d",
            "plotting.plots_2d",
            "plotting.plots_3d",
        ],
        "Plotting - Specialized": [
            "plotting.heatmaps",
            "plotting.synthetic_plots",
            "plotting.embeddings",
        ],
    }

    for category, module_prefixes in categories.items():
        lines.append(f"## {category}")
        lines.append("")

        for module_name, functions in sorted(registry.items()):
            # Check if module belongs to this category
            if not any(prefix in module_name for prefix in module_prefixes):
                continue

            lines.append(f"### `{module_name}`")
            lines.append("")

            if not functions:
                lines.append("*No public functions*")
                lines.append("")
                continue

            # Group functions by purpose
            for func in sorted(functions, key=lambda x: x["name"]):
                name = func["name"]
                summary = func["summary"] or "No description"
                params = ", ".join(func["parameters"][:3])
                if len(func["parameters"]) > 3:
                    params += ", ..."

                return_type = func["return_type"] or "Any"

                lines.append(f"#### `{name}({params})`")
                lines.append("")
                lines.append(f"**Returns:** `{return_type}`")
                lines.append("")
                lines.append(f"**Purpose:** {summary}")
                lines.append("")
                lines.append(f"**Location:** `{module_name}.py` (line {func['line']})")
                lines.append("")

        lines.append("---")
        lines.append("")

    return "\n".join(lines)


def main() -> None:
    """Main entry point."""
    # Get project root
    script_dir = Path(__file__).parent
    project_root = script_dir.parent
    neural_analysis_dir = project_root / "src" / "neural_analysis"
    docs_dir = project_root / "docs"

    if not neural_analysis_dir.exists():
        print(f"Error: neural_analysis directory not found: {neural_analysis_dir}")
        return

    print("Scanning neural_analysis package...")
    registry = generate_registry(neural_analysis_dir)

    total_functions = sum(len(funcs) for funcs in registry.values())
    print(f"Found {total_functions} public functions across {len(registry)} modules")

    # Generate markdown documentation
    docs_dir.mkdir(exist_ok=True)
    markdown_path = docs_dir / "function_registry.md"
    markdown_content = format_markdown_registry(registry)

    with open(markdown_path, "w", encoding="utf-8") as f:
        f.write(markdown_content)

    print(f"✓ Markdown registry written to: {markdown_path}")
    print("  (JSON format removed - only markdown is used by AI assistants)")

    # Print summary by category
    print("\nFunction Count by Module:")
    for module_name, functions in sorted(registry.items(), key=lambda x: x[0]):
        print(f"  {module_name}: {len(functions)} functions")


if __name__ == "__main__":
    main()
