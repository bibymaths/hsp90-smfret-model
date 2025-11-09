import pandas as pd
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

# === Configuration ===
data_dir = Path("data/Hugel_2025")   # adjust path
key = "/tracks/Data"                 # default key
frame_interval = 0.05                # s per frame (adjust later)

# === Loop through all .tracks/.h5 files ===
for path in sorted(data_dir.glob("*.tracks*")):
    print("="*100)
    print(f"📂 File: {path.name}")

    # Step 1 — list keys
    try:
        store = pd.HDFStore(path, mode="r")
        keys = store.keys()
        store.close()
        print(f"   Keys in file: {keys}")
    except Exception as e:
        print(f"   ❌ Could not open file: {e}")
        continue

    # Step 2 — try loading the most likely dataset
    k = key if key in keys else keys[0]
    try:
        df = pd.read_hdf(path, key=k)
    except Exception as e:
        print(f"   ❌ Error reading {k}: {e}")
        continue

    # Step 3 — flatten multiindex columns
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = ['_'.join(filter(None, col)).strip() for col in df.columns]

    # Step 4 — basic info
    print(f"   ✅ Loaded with shape: {df.shape}")
    print(f"   Columns ({len(df.columns)}):")
    print("   " + ", ".join(df.columns[:10]) + ("..." if len(df.columns) > 10 else ""))
    print()

    # Step 5 — show first few rows
    print(df.head(5))
    print()

    # Step 6 — quick summary for numeric columns
    num_cols = df.select_dtypes(include=[np.number]).columns
    summary = df[num_cols].describe().T[["mean", "std", "min", "max"]]
    print("   🔢 Quick numeric summary (mean ± std, min–max):")
    print(summary.head(10))
    print()

    # Step 7 — optional derived column preview
    if "donor_frame" in df.columns:
        df["time_s"] = df["donor_frame"] * frame_interval
        print(f"   Added time_s (first 5): {df['time_s'].head().tolist()}")
    print()

    # Step 8 — simple plot of first numeric column vs time if available
    # === Plot one representative FRET trace per file ===
    # Try to detect a FRET column
    fret_candidates = ["fret_eff", "fret_eff_app", "fret_efficiency"]
    fret_col = next((c for c in fret_candidates if c in df.columns), None)

    if fret_col is None:
        print("   ⚠ No FRET column found (looked for: "
              f"{', '.join(fret_candidates)}). Skipping plot.\n")
        continue

    # Ensure we have a time axis
    if "time_s" not in df.columns:
        if "donor_frame" in df.columns:
            df["time_s"] = df["donor_frame"] * frame_interval
        else:
            print("   ⚠ No donor_frame/time info found. Skipping plot.\n")
            continue

    # Choose a representative particle: longest trajectory
    part_col = "fret_particle" if "fret_particle" in df.columns else None

    if part_col is not None:
        counts = df.groupby(part_col).size()
        longest_pid = counts.sort_values(ascending=False).index[0]
        traj = df[df[part_col] == longest_pid].sort_values("time_s")
        label = f"particle {longest_pid}"
    else:
        # Fall back: just use the whole dataset
        traj = df.sort_values("time_s")
        label = "all data (no particle id)"

    # Make the plot
    plt.figure()
    plt.plot(traj["time_s"], traj[fret_col], marker="o", linestyle="-", markersize=3)
    plt.xlabel("time (s)")
    plt.ylabel(f"{fret_col}")
    plt.title(f"{path.name} – {label}")
    plt.tight_layout()
    plt.show()
