# Usage

## Prerequisites

- Python **3.11+**
- Dataset from Zenodo: **10.5281/zenodo.17559063**
- Place `.tracks/.h5` files in `data/Hugel_2025/`

## Step 1 — Extract trajectories (`get_traces.py`)

### CLI reference

- `--data-dir` (Path): input directory with `.tracks/.tracks.h5` files.
- `--export-dir` (Path): output directory for per-particle CSVs and `fret_matrix.csv`.
- `--frame-interval` (float): frame spacing in seconds.
- `--fret-min` / `--fret-max` (float): FRET filtering bounds.
- `--min-traj-length` (int): minimum per-particle frames kept.
- `--use-interpolation`: enable interpolation onto a common grid.
- `--no-inspect-plots`: disable representative inspection plots.
- `--save-plots`: save inspection figures to disk.
- `--plots-dir` (Path): directory for saved inspection plots.
- `--keep-intermediate`: keep per-particle CSVs after building the matrix.

### Inputs / outputs

- **Input**: raw tracking HDF5 files (`*.tracks*.h5`).
- **Outputs**:
  - per-particle CSV traces (`*_particle_XXXXX.csv`)
  - `fret_matrix.csv` (time × particle matrix)

### Example

```bash
python get_traces.py --data-dir data/Hugel_2025 --export-dir data/timeseries --save-plots
```

## Step 2 — Fit model (`pipeline.py`)

### CLI reference

- `--outdir`: directory for all fit outputs and figures.
- `--multistarts`: number of independent initializations per fit.
- `--bootstraps`: number of bootstrap replicates.
- `--cores`: worker count for parallelizable sections.

### Example

```bash
python pipeline.py --outdir results --multistarts 5 --bootstraps 10 --cores 7
```

!!! danger "🔥 Long runtimes"
    Bootstrap × multistart combinations can take hours on large datasets. For quick exploration, start with `--bootstraps 5 --multistarts 2`.

!!! tip "Parallelism"
    A practical default is `--cores $(($(nproc)-1))` to leave one CPU free for system responsiveness.

## Output files in `results/`

| File | Description |
|---|---|
| `best_fit_params.csv` | Best-fit parameter table from global fit. |
| `summary_conditions.csv` | Per-condition fitted parameter summary. |
| `summary_constructs.csv` | Construct-level pooled summary. |
| `bootstrap_summary_conditions.csv` | Condition-level bootstrap mean/CI table. |
| `bootstrap_summary_constructs.csv` | Construct-level bootstrap mean/CI table. |
| `sobol_condition.csv` | Sobol sensitivity indices by condition. |
| `sobol_construct.csv` | Sobol sensitivity indices by construct. |
| `*.png` | Fit overlays, residual, bootstrap, and sensitivity plots. |
