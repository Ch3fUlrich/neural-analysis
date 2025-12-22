# Synthetic Datasets Notebook Guide

The synthetic datasets notebook demonstrates how to generate toy data, run core analyses, and visualise results with the standard plotting and storage stack.

---

## 1. What You Can Do in 60 Seconds

- Generate a synthetic dataset (e.g. point clouds, trajectories).
- Compute distribution comparisons or structure indices.
- Save results via the storage helpers.
- Visualise outputs using PlotGrid helpers.

The notebook is the quickest way to see the full pipeline end‑to‑end.

---

## 2. Task Map

| Task                                | Action in notebook                   | Output                           |
|-------------------------------------|--------------------------------------|----------------------------------|
| Generate synthetic dataset           | Run the “Generate data” cell         | In‑memory arrays / DataFrame     |
| Compute distribution comparisons     | Run “Distribution comparisons” cell  | HDF5 file with comparisons       |
| Compute structure index              | Run “Structure index” cell           | HDF5 file with SI results        |
| Visualise distributions              | Run “Plot distributions” cell        | PlotGrid figure(s)               |
| Benchmark storage profiles           | Run “Storage benchmark” cell         | Timing table / summary           |

Use the notebook sections as templates for your own analyses.

---

## 3. Integration Points

- **Plotting** – cells call PlotGrid helpers (`plot_bar`, `plot_violin`, etc.); see `docs/plotgrid.md` for details.
- **Storage** – results are saved/loaded via `save_result_to_hdf5_dataset` and `StorageManager`; see `docs/storage.md`.
- **Testing ideas** – the notebook is a convenient place to prototype new metrics or visualisations before turning them into library functions + tests.

---

## 4. Recommended Workflow

1. Duplicate the notebook under a new name for your experiment.
2. Replace the data‑generation cell with your own loader or generator.
3. Reuse the analysis and plotting cells as‑is where possible.
4. Once stable, move core logic into `src/neural_analysis/...` and add tests.
