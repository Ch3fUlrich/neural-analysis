# Marimo Notebook Guide

This guide explains how to use marimo notebooks with the neural_analysis package. Marimo is a reactive Python notebook environment that provides a modern alternative to Jupyter notebooks.

## Table of Contents

- [Installation](#installation)
- [Quick Start](#quick-start)
- [Running Marimo Notebooks](#running-marimo-notebooks)
- [Using Marimo in Cursor/VSCode](#using-marimo-in-cursorvscode)
- [Workflow Best Practices](#workflow-best-practices)
- [Troubleshooting](#troubleshooting)
- [Quick Reference](#quick-reference)

## Installation

Marimo is already installed via `uv` in this project:

```bash
uv add marimo
```

If you need to install it manually:

```bash
pip install marimo
# or
uv pip install marimo
```

## Quick Start

The easiest way to get started is to open a marimo notebook in interactive mode:

```bash
uv run marimo edit examples/metrics_examples_marimo_nb.py
```

This command will:
- Start a local web server (default: `http://127.0.0.1:2718`)
- Automatically open your browser to the marimo interface
- Allow you to edit and run cells interactively
- Auto-save all changes back to the `.py` file

**Note:** The default port is **2718**. If port 2718 is already in use, marimo will automatically try the next available port.

**WSL/Linux Note:** If you see browser-related errors like `xdg-open: browser not found`, this is normal in headless environments. The notebook is still working correctly - marimo is just trying to auto-open a browser. Use `--headless` mode or manually open the URL shown in the terminal.

## Running Marimo Notebooks

### Interactive Editing (Recommended)

Open a marimo notebook in interactive mode for the best experience:

```bash
uv run marimo edit examples/metrics_examples_marimo_nb.py
```

**What happens:**
- Marimo starts a local web server (default port: 2718)
- Your browser opens automatically to `http://127.0.0.1:2718`
- You can edit cells directly in the browser
- Changes are automatically saved to the `.py` file
- Cell outputs update reactively as you edit

**Port Configuration:**
- **Default port:** 2718
- If 2718 is busy, marimo automatically tries the next available port
- You can specify a custom port with `--port`:
  ```bash
  uv run marimo edit examples/metrics_examples_marimo_nb.py --port 8888
  ```

### Running as a Python Script

You can also execute a marimo notebook as a regular Python script:

```bash
uv run python examples/metrics_examples_marimo_nb.py
```

**Note:** This runs the notebook but doesn't start the interactive web interface. Use this for automated execution or testing.

### Running as a Web App (Headless)

Run the notebook as a web app without automatically opening a browser:

```bash
uv run marimo run examples/metrics_examples_marimo_nb.py --headless
```

The terminal will display the URL to access the notebook (e.g., `http://127.0.0.1:2718`). This is useful when:
- Running on a remote server
- You want to manually control when the browser opens
- You're using a custom port configuration

### Converting Regular Python to Marimo Format

If you have a regular Python file with cell markers (like `# --- Cell 0 (markdown) ---`), convert it to marimo format:

```bash
uv run marimo convert examples/metrics_examples_marimo.py -o examples/metrics_examples_marimo_nb.py
```

## Using Marimo in Cursor/VSCode

Marimo has an official VSCode extension that provides native integration. Here are your options:

### Option 1: VSCode/Cursor Extension (Recommended)

1. **Install the marimo extension:**
   - Open the Extensions view (`Ctrl+Shift+X` or `Cmd+Shift+X`)
   - Search for "marimo"
   - Install the official marimo extension

2. **Open a marimo notebook:**
   - Open any `.py` file that's a marimo notebook
   - Click the marimo logo in the top-right corner to toggle the notebook view
   - You can now edit and run cells directly in the editor

3. **Benefits:**
   - Native integration with your IDE
   - Syntax highlighting and autocomplete
   - Run cells without leaving the editor
   - See outputs inline

### Option 2: Integrated Terminal + Browser

If you prefer the web interface or the extension isn't available:

1. Open the integrated terminal in Cursor/VSCode (`Ctrl+`` or `View > Terminal`)
2. Run:
   ```bash
   uv run marimo edit examples/metrics_examples_marimo_nb.py
   ```
3. The browser opens automatically with the marimo interface
4. Edit in either:
   - **Browser (marimo UI)**: Changes sync to the file automatically
   - **Editor (VSCode/Cursor)**: Use `--watch` flag to auto-reload changes

### Option 3: Watch Mode (Best for IDE Editing)

Use the `--watch` flag so changes in your editor are reflected in marimo:

```bash
uv run marimo edit examples/metrics_examples_marimo_nb.py --watch
```

**Workflow:**
1. Edit the `.py` file in Cursor/VSCode
2. Save the file (`Ctrl+S` or `Cmd+S`)
3. Changes automatically reload in the marimo browser interface
4. See outputs update in real-time

This is ideal when you want to:
- Use your IDE's advanced editing features (autocomplete, refactoring, etc.)
- Still see interactive outputs in the browser
- Have changes sync automatically

### Option 4: Custom Port Configuration

If you need to use a specific port (e.g., for port forwarding or firewall rules):

```bash
uv run marimo edit examples/metrics_examples_marimo_nb.py --port 8888
```

Then open `http://127.0.0.1:8888` in your browser.

**Common use cases:**
- Port forwarding from a remote server
- Multiple notebooks running simultaneously
- Integration with reverse proxies

## Workflow Best Practices

### Recommended Workflow

1. **Edit in Cursor/VSCode**: Use your IDE for code editing
   - Syntax highlighting
   - Autocomplete and IntelliSense
   - Git integration
   - Code navigation and refactoring

2. **Run in Marimo**: Use `marimo edit --watch` to see outputs
   - Interactive visualizations
   - Real-time cell execution
   - Reactive updates

3. **Sync automatically**: The `--watch` flag ensures bidirectional sync
   - Changes in editor → reload in browser
   - Changes in browser → saved to file

### File Structure

- **Source file**: `examples/metrics_examples_marimo.py`
  - Regular Python file with cell markers (`# --- Cell N (type) ---`)
  - Can be run as a script or converted to marimo format

- **Marimo notebook**: `examples/metrics_examples_marimo_nb.py`
  - Proper marimo format with `@app.cell` decorators
  - Can be edited interactively in marimo
  - Can still be run as a regular Python script

### Converting Between Formats

**To marimo format:**
```bash
uv run marimo convert script.py -o notebook.py
```

**From marimo format:**
Marimo notebooks are just Python files, so you can edit them directly. If you need to extract the cell content:
- Remove the `marimo.App()` and `@app.cell` decorators
- Keep the cell content
- Optionally add cell markers for compatibility

## Troubleshooting

### Port Already in Use

If port 2718 (or your specified port) is busy:

**Option 1:** Let marimo auto-select the next available port:
```bash
uv run marimo edit examples/metrics_examples_marimo_nb.py
# Marimo will automatically try 2719, 2720, etc.
```

**Option 2:** Specify a different port:
```bash
uv run marimo edit examples/metrics_examples_marimo_nb.py --port 8888
```

**Option 3:** Find and kill the process using the port:
```bash
# Linux/Mac
lsof -ti:2718 | xargs kill -9

# Windows
netstat -ano | findstr :2718
taskkill /PID <PID> /F
```

### Browser Doesn't Open Automatically (WSL/Linux)

If you see errors like `xdg-open: browser not found` or the browser doesn't open:

1. **Use headless mode** and manually open the URL:
   ```bash
   uv run marimo edit examples/metrics_examples_marimo_nb.py --headless
   ```
   The terminal will display the URL (e.g., `http://127.0.0.1:2718`)

2. **Copy the URL** from the terminal output and paste it into your browser

3. **In WSL**: You may need to set up X11 forwarding or use Windows browser:
   - The notebook server is running correctly (the browser errors are just warnings)
   - Copy the URL from terminal (e.g., `http://127.0.0.1:2718`)
   - Open it in your Windows browser (if using WSL2) or use port forwarding

4. **Check your browser settings** - some browsers block auto-opening

### Changes Not Syncing

If changes in your editor aren't reflected in marimo:

1. **Use the `--watch` flag:**
   ```bash
   uv run marimo edit examples/metrics_examples_marimo_nb.py --watch
   ```

2. **Save your file** - marimo only reloads on file save

3. **Check file permissions** - ensure marimo can read/write the file

4. **Restart marimo** - sometimes a restart fixes sync issues

### Import Errors

If you encounter import errors:

1. **Use `uv run`** to ensure the correct environment:
   ```bash
   uv run marimo edit examples/metrics_examples_marimo_nb.py
   ```

2. **Check your working directory** - run from the project root:
   ```bash
   cd /home/donatolab/code/neural-analysis
   uv run marimo edit examples/metrics_examples_marimo_nb.py
   ```

3. **Verify installation** - ensure neural_analysis is installed:
   ```bash
   uv run python -c "import neural_analysis; print(neural_analysis.__version__)"
   ```

### Notebook Structure Issues

If the notebook shows as one big code block instead of separate cells:

1. **Check the file format** - ensure it's a proper marimo notebook:
   ```bash
   head -5 examples/metrics_examples_marimo_nb.py
   # Should show: import marimo, __generated_with, app = marimo.App()
   ```

2. **Reconvert the file:**
   ```bash
   uv run marimo convert examples/metrics_examples_marimo.py -o examples/metrics_examples_marimo_nb.py
   ```

3. **Check for syntax errors** - marimo requires valid Python syntax

## Saving Outputs and Plots

Unlike Jupyter notebooks, marimo notebooks do **not** save outputs directly in the notebook file (`.py`). However, you can export notebooks to formats that include outputs:

### Exporting to HTML (with outputs)

Export your notebook to HTML, which includes all outputs. **All example notebooks with outputs are saved to `examples/__marimo__/`**:

```bash
uv run marimo export html examples/metrics_examples_marimo_nb.py -o examples/__marimo__/metrics_examples_marimo_nb.html
```

**Finding Example Notebooks with Outputs:**

All exported HTML notebooks with executed outputs are located in `examples/__marimo__/`. This directory contains:
- Fully executed HTML notebooks with all outputs, plots, and visualizations
- Session data for interactive editing
- Automatic snapshots (if enabled)

**To view**: Open any `.html` file in `examples/__marimo__/` in your web browser. These are standalone HTML files that don't require marimo to view.

**Available examples**:
- `examples/__marimo__/metrics_examples_marimo_nb.html`
- `examples/__marimo__/structure_index_examples_marimo_nb.html`
- `examples/__marimo__/neural_analysis_example_marimo_nb.html`
- `examples/__marimo__/synthetic_datasets_example_marimo_nb.html`
- And more...

The `__marimo__/` directory is automatically created by marimo when exporting notebooks.

### Automatic Export with Watch Mode

To automatically regenerate the HTML file whenever the notebook changes:

```bash
uv run marimo export html examples/metrics_examples_marimo_nb.py -o output.html --watch
```

This will:
- Monitor the notebook file for changes
- Automatically regenerate the HTML output
- Overwrite the output file each time

**Note:** Install `watchdog` for efficient file watching:
```bash
pip install watchdog
# or
uv pip install watchdog
```

### Exporting to Jupyter Format (with outputs)

You can also export to Jupyter notebook format (`.ipynb`) with outputs:

```bash
uv run marimo export ipynb examples/metrics_examples_marimo_nb.py -o output.ipynb --include-outputs
```

### Automatic Snapshots

Marimo can automatically save snapshots of your notebook:
- Open the notebook in marimo editor
- Use the menu to enable automatic snapshots
- Snapshots are saved to `__marimo__/` directory in the notebook folder

**Important:** Unlike Jupyter, marimo notebooks are **reactive** - outputs are computed on-demand when cells are executed, not stored in the file. To persist outputs, use the export commands above.

### Finding Example Notebooks with Outputs

All example notebooks with executed outputs are located in `examples/__marimo__/`. This directory contains:

- **HTML exports**: Fully executed notebooks with all outputs, plots, and visualizations
- **Session data**: Interactive editing state (`.json` files)
- **Snapshots**: Automatic snapshots if enabled

**To view**: Simply open any `.html` file in `examples/__marimo__/` in your web browser. These are standalone HTML files that work without marimo installed.

**To regenerate**: Export a marimo notebook to HTML:
```bash
uv run marimo export html examples/notebook.py -o examples/__marimo__/notebook.html
```

The `__marimo__/` directory is automatically created by marimo when exporting notebooks.

## Quick Reference

### Essential Commands

```bash
# Edit interactively (opens browser, default port 2718)
uv run marimo edit notebook.py

# Edit with watch mode (syncs with editor)
uv run marimo edit notebook.py --watch

# Edit on custom port
uv run marimo edit notebook.py --port 8888

# Run as script (no web interface)
uv run python notebook.py

# Run as app (headless, no auto-open browser)
uv run marimo run notebook.py --headless

# Convert to marimo format
uv run marimo convert script.py -o notebook.py
```

### Port Information

- **Default port:** 2718
- **Auto-selection:** If 2718 is busy, marimo tries 2719, 2720, etc.
- **Custom port:** Use `--port` flag to specify
- **Check port:** Look at terminal output for the actual URL

### Key Differences from Jupyter

| Feature | Jupyter | Marimo |
|---------|---------|--------|
| Default port | 8888 | 2718 |
| File format | `.ipynb` (JSON) | `.py` (Python) |
| Cell execution | Manual | Reactive |
| State management | Global kernel | Per-cell dependencies |
| Version control | Difficult (JSON) | Easy (Python) |

## Additional Resources

- [Marimo Documentation](https://docs.marimo.io/)
- [Marimo VSCode Extension](https://marketplace.visualstudio.com/items?itemName=marimo-team.marimo)
- [Marimo CLI Reference](https://docs.marimo.io/cli/)

