#!/usr/bin/env python3
from pathlib import Path
import pandas as pd
import re

DATA_DIR = Path("data/Hugel_2025")   # change if needed
KEY = "/tracks/Data"                # default; falls back to first key

# quick test on one file
path = "data/Hugel_2025/filtered-241107-Hsp90_409_601-v014.tracks.h5"
df = pd.read_hdf(path, "/tracks/Data")
if isinstance(df.columns, pd.MultiIndex):
    df.columns = ["_".join([c for c in col if c]).strip() for col in df.columns]

for c in df.columns:
    if any(k in str(c).lower() for k in ["dd","da","aa","donor","acceptor","int","count","photon"]):
        print(c)

# patterns that typically appear in ALEX datasets
PATTERNS = [
    r"\bexc\b", r"exc_type", r"laser", r"alternat", r"alex",
    r"\bDD\b", r"\bDA\b", r"\bAA\b", r"I_DD", r"I_DA", r"I_AA",
    r"donor.*exc", r"acceptor.*exc",
    r"donor.*int", r"acceptor.*int", r"intensity", r"photon", r"count"
]
rx = re.compile("|".join(PATTERNS), flags=re.IGNORECASE)

for path in sorted(DATA_DIR.glob("*.tracks*.h5")):
    print("=" * 90)
    print("FILE:", path.name)

    # list keys
    try:
        store = pd.HDFStore(path, mode="r")
        keys = store.keys()
        store.close()
        print("Keys:", keys)
    except Exception as e:
        print("Could not open:", e)
        continue

    k = KEY if KEY in keys else keys[0]
    try:
        df = pd.read_hdf(path, key=k)
    except Exception as e:
        print("Could not read key:", k, "error:", e)
        continue

    # flatten columns
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = ["_".join([c for c in col if c]).strip() for col in df.columns]

    cols = list(map(str, df.columns))
    hits = [c for c in cols if rx.search(c)]

    print("Using key:", k)
    print("n_rows:", len(df), "n_cols:", len(cols))

    # show the most informative stuff
    print("\n--- ALEX-related column hits ---")
    if hits:
        for c in hits:
            print(" ", c)
    else:
        print(" (none)")

    # check for excitation-type / alternation column values if present
    exc_candidates = [c for c in cols if re.search(r"exc|laser|alternat|alex", c, re.I)]
    for c in exc_candidates[:10]:
        try:
            vals = pd.Series(df[c]).dropna().astype(str).unique()[:20]
            print(f"\nValues sample for '{c}':", vals)
        except Exception:
            pass
