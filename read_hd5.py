import pandas as pd
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import warnings

from scipy.interpolate import CubicSpline

warnings.filterwarnings("ignore", category=UserWarning, module="pandas")

# === Configuration ===
data_dir = Path("data/Hugel_2025")   # adjust path
key = "/tracks/Data"                 # default key
frame_interval = 0.05                # seconds per frame
fret_max = 2                       # max FRET efficiency to consider
fret_min = 0                      # min FRET efficiency to consider

def interpolate_trace(time_grid: np.ndarray,
                      t_trace: np.ndarray,
                      E_trace: np.ndarray) -> np.ndarray:
    """
    Interpolate a single FRET trace onto a common time grid using cubic splines
    where possible. Handles NaNs and short traces robustly.

    Returns an array of shape time_grid with NaNs outside the observed range
    or when not enough data points are available.
    """
    # Ensure arrays
    t_trace = np.asarray(t_trace, float)
    E_trace = np.asarray(E_trace, float)

    # Mask out non-finite values
    mask = np.isfinite(t_trace) & np.isfinite(E_trace)
    t_clean = t_trace[mask]
    E_clean = E_trace[mask]

    # Not enough points to interpolate
    if t_clean.size == 0:
        return np.full_like(time_grid, np.nan, dtype=float)
    if t_clean.size == 1:
        # Single point: constant over its range, NaN elsewhere
        y = np.full_like(time_grid, np.nan, dtype=float)
        idx = np.argmin(np.abs(time_grid - t_clean[0]))
        y[idx] = E_clean[0]
        return y
    # Remove duplicate time stamps, if any
    t_unique, idx_unique = np.unique(t_clean, return_index=True)
    E_unique = E_clean[idx_unique]

    if t_unique.size < 2:
        return np.full_like(time_grid, np.nan, dtype=float)

    # For 2 points, a spline is pointless; use linear interpolation
    if t_unique.size == 2:
        y = np.interp(time_grid, t_unique, E_unique,
                      left=np.nan, right=np.nan)
        return y

    # For >=3 points, try cubic spline
    try:
        cs = CubicSpline(t_unique, E_unique, extrapolate=False)
        y = cs(time_grid)
    except Exception:
        y = np.interp(time_grid, t_unique, E_unique,
                      left=np.nan, right=np.nan)

    outside = (time_grid < t_unique.min()) | (time_grid > t_unique.max())
    y[outside] = np.nan
    y[(y < fret_min) | (y > fret_max)] = np.nan
    return y

# === Inspect and plot data ===
for path in sorted(data_dir.glob("*.tracks*")):
    print("=" * 80)

    # Extract metadata from filename
    fname = path.stem  # e.g. filtered-241107-Hsp90_409_601-v014.tracks
    parts = fname.split("-")
    if len(parts) >= 3:
        exp_id = parts[1]
        construct = parts[2].split(".")[0]
    else:
        exp_id = "unknown"
        construct = "unknown"

    print(f"File: {path.name}")
    print(f"Experiment: {construct}, Date/ID: {exp_id}")

    # Step 1 — list keys
    try:
        store = pd.HDFStore(path, mode="r")
        keys = store.keys()
        store.close()
        print(f"Keys in file: {keys}")
    except Exception as e:
        print(f"Could not open file: {e}")
        continue

    # Step 2 — read dataset
    k = key if key in keys else keys[0]
    try:
        df = pd.read_hdf(path, key=k)
    except Exception as e:
        print(f"Error reading {k}: {e}")
        continue

    # Step 3 — flatten multiindex columns
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = ["_".join(filter(None, col)).strip() for col in df.columns]

    # Step 4 — basic info
    print(f"Loaded with shape: {df.shape}")
    print(f"Columns ({len(df.columns)}):")
    print("   " + ", ".join(df.columns[:10]) + ("..." if len(df.columns) > 10 else ""))
    print()

    # Step 5 — show sample rows
    print(df.head(5))
    print()

    # Step 6 — numeric summary
    num_cols = df.select_dtypes(include=[np.number]).columns
    summary = df[num_cols].describe().T[["mean", "std", "min", "max"]]
    print("Numeric summary (mean ± std, min–max):")
    print(summary.head(10))
    print()

    # Step 7 — add time column
    if "donor_frame" in df.columns:
        df["time_s"] = df["donor_frame"] * frame_interval
        print(f"Added time_s (first 5): {df['time_s'].head().tolist()}")
    print()

    # Step 8 — detect FRET column
    fret_candidates = ["fret_eff", "fret_eff_app", "fret_efficiency"]
    fret_col = next((c for c in fret_candidates if c in df.columns), None)

    if fret_col is None:
        print(f"No FRET column found (checked: {', '.join(fret_candidates)}). Skipping plot.\n")
        continue

    # Ensure we have a time axis
    if "time_s" not in df.columns:
        if "donor_frame" in df.columns:
            df["time_s"] = df["donor_frame"] * frame_interval
        else:
            print("No donor_frame/time info found. Skipping plot.\n")
            continue

    # Choose a representative particle: longest trajectory
    part_col = "fret_particle" if "fret_particle" in df.columns else None

    if part_col is not None:
        counts = df.groupby(part_col).size()
        longest_pid = counts.sort_values(ascending=False).index[0]
        traj = df[df[part_col] == longest_pid].sort_values("time_s")
        label = f"{construct} {exp_id} – particle {int(longest_pid)}"
    else:
        traj = df.sort_values("time_s")
        label = f"{construct} {exp_id} – all data (no particle id)"

    # Plot representative trace
    plt.figure()
    plt.plot(traj["time_s"], traj[fret_col], marker="o", linestyle="-", markersize=3)
    plt.xlabel("time (s)")
    plt.ylabel(f"{fret_col}")
    plt.title(label)
    plt.tight_layout()
    plt.show()


# === Export per-particle time series ===
export_dir = Path("data/timeseries")
export_dir.mkdir(exist_ok=True)

for path in sorted(data_dir.glob("*.tracks*.h5")):
    fname = path.stem
    parts = fname.split("-")
    if len(parts) >= 3:
        exp_id = parts[1]
        construct = parts[2].split(".")[0]
    else:
        exp_id = "unknown"
        construct = "unknown"

    print(f"\nExporting per-particle time series from: {path.name}")
    print(f"Experiment: {construct}, Date/ID: {exp_id}")

    try:
        df = pd.read_hdf(path, key="/tracks/Data")
    except Exception as e:
        print(f"Could not read {path.name}: {e}")
        continue

    # flatten columns
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = ["_".join(filter(None, c)).strip() for c in df.columns]

    # ensure required columns
    needed_cols = {"donor_frame", "fret_particle"}
    if not needed_cols.issubset(df.columns):
        print("Missing required columns, skipping.")
        continue

    df["time_s"] = df["donor_frame"] * frame_interval

    fret_candidates = ["fret_eff", "fret_eff_app"]
    fret_col = next((c for c in fret_candidates if c in df.columns), None)
    if fret_col is None:
        print("No FRET efficiency column found, skipping file.")
        continue

    if "filter_manual" in df.columns:
        df = df[df["filter_manual"] == 1]

    # keep only finite, physically reasonable FRET values
    df = df[np.isfinite(df[fret_col])]
    df = df[(df[fret_col] > fret_min) & (df[fret_col] < fret_max)]

    # drop short trajectories AFTER filtering
    lengths = df.groupby("fret_particle")["donor_frame"].nunique()
    keep = lengths[lengths >= 20].index
    df = df[df["fret_particle"].isin(keep)]

    count = 0
    for pid, traj in df.groupby("fret_particle"):
        traj = traj.sort_values("donor_frame")
        out = traj[["time_s", fret_col]].rename(columns={fret_col: "FRET"})
        out_name = f"{path.stem}_particle_{int(pid):05d}.csv"
        out.to_csv(export_dir / out_name, index=False)
        count += 1

    print(f"Exported {count} per-particle traces to {export_dir}/")


# === Combine all exported trajectories into a single matrix ===
combined_out = export_dir / "combined_fret_matrix.csv"
print("\nBuilding combined FRET matrix (uniform 0–max_t grid)...")

csv_files = sorted(export_dir.glob("*.csv"))
if not csv_files:
    print("No per-particle CSV files found, skipping matrix creation.")
else:
    max_t = 0.0

    # Find max time across all traces
    for f in csv_files:
        df_tmp = pd.read_csv(f)
        if "time_s" in df_tmp.columns and len(df_tmp) > 0:
            max_t = max(max_t, df_tmp["time_s"].max())

    time_grid = np.arange(0.0, max_t + frame_interval / 2, frame_interval)
    # Collect all interpolated traces first
    columns = {"time_s": time_grid}

    for i, f in enumerate(csv_files):
        df = pd.read_csv(f)
        if len(df) == 0:
            continue

        t_trace = df["time_s"].values
        E_trace = df["FRET"].values
        interp = interpolate_trace(time_grid, t_trace, E_trace)

        stem = f.stem
        parts = stem.split("-")
        if len(parts) >= 3:
            exp_id = parts[1]
            construct = parts[2].split(".")[0]
        else:
            exp_id = "unknown"
            construct = "unknown"

        pid = stem.split("particle_")[-1]
        col_name = f"{construct}_{exp_id}_p{pid}"

        columns[col_name] = interp

        if (i + 1) % 100 == 0:
            print(f"Processed {i + 1} traces ...")

    # Build one DataFrame in a single allocation
    combined = pd.DataFrame(columns)
    combined.to_csv(combined_out, index=False)
    print(f"Combined matrix saved → {combined_out}")
    print(f"Time points: {len(time_grid)}, trajectories: {combined.shape[1] - 1}")

