#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Visualize per-trajectory HMM fits (hmmlearn outputs).

Expected input directory structure (from fit_hmm_per_trace.py):
  results_hmm/
    hmm_summary.csv
    statepaths/<traj>.statepath.csv
    dwells/<traj>.dwells.csv

Outputs:
  results_hmm/viz/
    *.png

Run:
  python viz_hmm_results.py --hmm-dir results_hmm --n-examples 12
"""

import argparse
import logging
from pathlib import Path
from typing import Tuple, Optional

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger("viz_hmm")


# -----------------------
# Helpers
# -----------------------
def parse_key_from_filename(fname: str) -> Tuple[str, str, str]:
    """
    Best-effort parse from your naming patterns.
    Handles e.g.:
      <anything>-<exp_id>-<construct>..._particle_00001.csv
      OR
      <construct>_<exp_id>_p00001.statepath.csv

    Returns: (construct, exp_id, particle_id)
    """
    stem = Path(fname).stem

    # Case A: statepath names from your combined matrix approach:
    #   construct_expId_p00001.statepath
    if "_p" in stem and "_" in stem:
        parts = stem.split("_")
        if len(parts) >= 3 and parts[-1].startswith("p"):
            particle = parts[-1]
            exp_id = parts[-2]
            construct = "_".join(parts[:-2])
            return construct, exp_id, particle

    # Case B: export names:
    #   something-241107-Hsp90_409_601..._particle_00001
    if "particle_" in stem and "-" in stem:
        left, particle = stem.rsplit("particle_", 1)
        particle = f"p{particle}"
        dash_parts = left.split("-")
        if len(dash_parts) >= 3:
            exp_id = dash_parts[1]
            construct = dash_parts[2].split(".")[0]
            return construct, exp_id, particle

    return "unknown", "unknown", "unknown"


def ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def load_summary(hmm_dir: Path) -> pd.DataFrame:
    p = hmm_dir / "hmm_summary.csv"
    if not p.exists():
        raise FileNotFoundError(f"Missing {p}")
    df = pd.read_csv(p)

    # annotate construct/exp_id
    constructs, exp_ids, particles = [], [], []
    for traj in df["trajectory"].astype(str).tolist():
        c, e, pid = parse_key_from_filename(traj)
        constructs.append(c)
        exp_ids.append(e)
        particles.append(pid)
    df["construct"] = constructs
    df["exp_id"] = exp_ids
    df["particle"] = particles
    df["condition"] = df["construct"].astype(str) + "_" + df["exp_id"].astype(str)

    return df


def plot_scatter(df: pd.DataFrame, x: str, y: str, out: Path, title: str) -> None:
    dd = df[[x, y]].dropna()
    if dd.empty:
        logger.info(f"[skip] {title} (no data)")
        return
    plt.figure(figsize=(7, 5))
    plt.scatter(dd[x].to_numpy(), dd[y].to_numpy(), s=12, alpha=0.7)
    plt.xlabel(x)
    plt.ylabel(y)
    plt.title(title)
    plt.tight_layout()
    plt.savefig(out, dpi=300)
    plt.close()


def plot_hist(vals: np.ndarray, out: Path, title: str, xlabel: str, bins: int = 40) -> None:
    vals = vals[np.isfinite(vals)]
    if vals.size == 0:
        logger.info(f"[skip] {title} (no values)")
        return
    plt.figure(figsize=(7, 5))
    plt.hist(vals, bins=bins, alpha=0.9)
    plt.xlabel(xlabel)
    plt.ylabel("count")
    plt.title(title)
    plt.tight_layout()
    plt.savefig(out, dpi=300)
    plt.close()


def compute_state_occupancy_from_statepath(sp: pd.DataFrame, n_states: int) -> np.ndarray:
    s = sp["state"].to_numpy(dtype=int)
    occ = np.zeros(n_states, dtype=float)
    for k in range(n_states):
        occ[k] = np.mean(s == k) if len(s) else np.nan
    return occ


def read_statepath(hmm_dir: Path, traj_name: str) -> Optional[pd.DataFrame]:
    # our fit script wrote: statepaths/<traj_stem>.statepath.csv
    stem = Path(traj_name).stem
    p = hmm_dir / "statepaths" / f"{stem}.statepath.csv"
    if not p.exists():
        return None
    return pd.read_csv(p)


def read_dwells(hmm_dir: Path, traj_name: str) -> Optional[pd.DataFrame]:
    stem = Path(traj_name).stem
    p = hmm_dir / "dwells" / f"{stem}.dwells.csv"
    if not p.exists():
        return None
    return pd.read_csv(p)


def plot_examples_with_states(hmm_dir: Path, df: pd.DataFrame, outdir: Path, n_examples: int, seed: int) -> None:
    df2 = df.dropna(subset=["trajectory", "n_states"]).copy()
    if df2.empty:
        return

    rng = np.random.default_rng(seed)
    pick = df2.sample(n=min(n_examples, len(df2)), random_state=seed)

    for _, row in pick.iterrows():
        traj = str(row["trajectory"])
        sp = read_statepath(hmm_dir, traj)
        if sp is None or sp.empty:
            continue

        t = sp["time_s"].to_numpy(dtype=float)
        E = sp["E"].to_numpy(dtype=float)
        s = sp["state"].to_numpy(dtype=int)

        plt.figure(figsize=(10, 4))
        plt.plot(t, E, lw=1)

        # overlay state as step line (scaled)
        s_scaled = (s - s.min()) / max(1, (s.max() - s.min()))
        s_scaled = 0.05 + 0.25 * s_scaled
        plt.plot(t, s_scaled, lw=2)

        c, e, pid = parse_key_from_filename(traj)
        plt.title(f"{traj}\nconstruct={c}, exp_id={e}, {pid}")
        plt.xlabel("time (s)")
        plt.ylabel("E (and state overlay scaled)")
        plt.tight_layout()
        plt.savefig(outdir / f"example_{Path(traj).stem}.png", dpi=300)
        plt.close()


# -----------------------
# Biology-facing summaries
# -----------------------
def build_occupancy_table(hmm_dir: Path, df: pd.DataFrame) -> pd.DataFrame:
    records = []
    for _, row in df.iterrows():
        traj = str(row["trajectory"])
        K = int(row["n_states"])
        sp = read_statepath(hmm_dir, traj)
        if sp is None or sp.empty:
            continue
        occ = compute_state_occupancy_from_statepath(sp, K)
        rec = {
            "trajectory": traj,
            "construct": row["construct"],
            "exp_id": row["exp_id"],
            "condition": row["condition"],
            "n_states": K,
        }
        for k in range(K):
            rec[f"occ_{k}"] = float(occ[k])
        records.append(rec)

    return pd.DataFrame(records)


def gather_dwell_table(hmm_dir: Path, df: pd.DataFrame) -> pd.DataFrame:
    all_dw = []
    for _, row in df.iterrows():
        traj = str(row["trajectory"])
        dw = read_dwells(hmm_dir, traj)
        if dw is None or dw.empty:
            continue
        dw = dw.copy()
        dw["trajectory"] = traj
        dw["construct"] = row["construct"]
        dw["exp_id"] = row["exp_id"]
        dw["condition"] = row["condition"]
        dw["n_states"] = row["n_states"]
        all_dw.append(dw)
    if not all_dw:
        return pd.DataFrame()
    return pd.concat(all_dw, ignore_index=True)


def plot_dwell_distributions(dw: pd.DataFrame, outdir: Path) -> None:
    if dw.empty:
        logger.info("[skip] dwell plots (no dwell table)")
        return

    # focus on 2-state for biology (open/closed)
    dw2 = dw[dw["n_states"] == 2].copy()
    if dw2.empty:
        logger.info("[skip] dwell plots (no 2-state trajectories)")
        return

    for st in sorted(dw2["state"].dropna().unique()):
        vals = dw2.loc[dw2["state"] == st, "dwell_s"].to_numpy(dtype=float)
        plot_hist(vals, outdir / f"dwell_state_{int(st)}.png",
                  title=f"Dwell time distribution (2-state), state={int(st)}",
                  xlabel="dwell time (s)", bins=40)


def plot_state_means(df: pd.DataFrame, outdir: Path) -> None:
    # plot mu_0, mu_1 distributions for 2-state fits
    df2 = df[df["n_states"] == 2].copy()
    if df2.empty:
        logger.info("[skip] state mean plots (no 2-state fits)")
        return

    plot_hist(df2["mu_0"].to_numpy(dtype=float), outdir / "mu0_hist.png",
              title="State mean distribution (2-state): mu_0 (lower FRET)",
              xlabel="mu_0 (FRET)")

    plot_hist(df2["mu_1"].to_numpy(dtype=float), outdir / "mu1_hist.png",
              title="State mean distribution (2-state): mu_1 (higher FRET)",
              xlabel="mu_1 (FRET)")

    # scatter mu0 vs mu1
    dd = df2[["mu_0", "mu_1"]].dropna()
    if not dd.empty:
        plt.figure(figsize=(6, 6))
        plt.scatter(dd["mu_0"].to_numpy(), dd["mu_1"].to_numpy(), s=14, alpha=0.7)
        plt.xlabel("mu_0 (lower FRET)")
        plt.ylabel("mu_1 (higher FRET)")
        plt.title("2-state fitted means: mu_0 vs mu_1")
        plt.tight_layout()
        plt.savefig(outdir / "mu0_vs_mu1.png", dpi=300)
        plt.close()


def plot_condition_level_insights(df: pd.DataFrame, occ: pd.DataFrame, outdir: Path) -> None:
    """
    Condition-level (construct+exp_id) summaries:
      - fraction in high-FRET state (occ_1 for 2-state)
      - mean mu_0 / mu_1
      - mean self-transition probabilities (T_0_0, T_1_1)
    """
    df2 = df[df["n_states"] == 2].copy()
    if df2.empty:
        logger.info("[skip] condition plots (no 2-state fits)")
        return

    # merge occupancy if available
    if not occ.empty:
        occ2 = occ[occ["n_states"] == 2].copy()
        dfm = df2.merge(occ2[["trajectory", "occ_0", "occ_1"]], on="trajectory", how="left")
    else:
        dfm = df2.copy()

    # aggregate by condition
    g = dfm.groupby("condition", dropna=False).agg(
        n=("trajectory", "count"),
        mu0=("mu_0", "mean"),
        mu1=("mu_1", "mean"),
        T00=("T_0_0", "mean"),
        T11=("T_1_1", "mean"),
        occ1=("occ_1", "mean"),
    ).reset_index()

    g = g.sort_values("condition")

    # plot occ1 by condition (proxy for closed-state occupancy if mu1 is higher-FRET)
    if g["occ1"].notna().any():
        plt.figure(figsize=(10, 4))
        plt.plot(g["condition"], g["occ1"], "o-")
        plt.xticks(rotation=90)
        plt.ylabel("mean occupancy of state 1 (high FRET)")
        plt.title("Condition-level high-FRET occupancy (2-state HMM)")
        plt.tight_layout()
        plt.savefig(outdir / "condition_occ1.png", dpi=300)
        plt.close()

    # plot T00 and T11 by condition (stickiness)
    plt.figure(figsize=(10, 4))
    plt.plot(g["condition"], g["T00"], "o-", label="T00")
    plt.plot(g["condition"], g["T11"], "o-", label="T11")
    plt.xticks(rotation=90)
    plt.ylabel("mean self-transition probability")
    plt.title("Condition-level self-transition (2-state HMM)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(outdir / "condition_self_transition.png", dpi=300)
    plt.close()

    # plot mu0/mu1 by condition
    plt.figure(figsize=(10, 4))
    plt.plot(g["condition"], g["mu0"], "o-", label="mu0 (low FRET)")
    plt.plot(g["condition"], g["mu1"], "o-", label="mu1 (high FRET)")
    plt.xticks(rotation=90)
    plt.ylabel("mean state mean (FRET)")
    plt.title("Condition-level fitted state means (2-state HMM)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(outdir / "condition_state_means.png", dpi=300)
    plt.close()

    g.to_csv(outdir / "condition_summary_2state.csv", index=False)


# -----------------------
# Main
# -----------------------
def main():
    ap = argparse.ArgumentParser(description="Visualize HMM fit performance + biological insights.")
    ap.add_argument("--hmm-dir", type=Path, default=Path("results_hmm"))
    ap.add_argument("--outdir", type=Path, default=None, help="Output dir (default: <hmm-dir>/viz)")
    ap.add_argument("--n-examples", type=int, default=12)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    hmm_dir = args.hmm_dir
    outdir = args.outdir if args.outdir is not None else (hmm_dir / "viz")
    ensure_dir(outdir)

    df = load_summary(hmm_dir)
    logger.info(f"Loaded {len(df)} fitted trajectories from {hmm_dir/'hmm_summary.csv'}")

    # -----------------------
    # Model performance plots
    # -----------------------
    plot_scatter(df, "n_points", "bic", outdir / "bic_vs_npoints.png", "BIC vs trajectory length")
    plot_scatter(df, "n_points", "logL", outdir / "logL_vs_npoints.png", "Log-likelihood vs trajectory length")

    # distribution of number of states
    plot_hist(df["n_states"].to_numpy(dtype=float), outdir / "n_states_hist.png",
              title="Selected number of states per trajectory", xlabel="n_states", bins=10)

    # Means (state emissions)
    plot_state_means(df, outdir)

    # -----------------------
    # Example overlays
    # -----------------------
    ex_dir = outdir / "examples"
    ensure_dir(ex_dir)
    plot_examples_with_states(hmm_dir, df, ex_dir, n_examples=args.n_examples, seed=args.seed)

    # -----------------------
    # Biological insight plots
    # -----------------------
    occ = build_occupancy_table(hmm_dir, df)
    if not occ.empty:
        occ.to_csv(outdir / "occupancy_table.csv", index=False)

        # occupancy hist for 2-state high-FRET state
        occ2 = occ[occ["n_states"] == 2].copy()
        if not occ2.empty and "occ_1" in occ2.columns:
            plot_hist(occ2["occ_1"].to_numpy(dtype=float), outdir / "occ1_hist.png",
                      title="Occupancy of high-FRET state (2-state HMM)", xlabel="occ_1", bins=40)

    dw = gather_dwell_table(hmm_dir, df)
    if not dw.empty:
        dw.to_csv(outdir / "dwells_all.csv", index=False)
    plot_dwell_distributions(dw, outdir)

    # Condition-level summaries (construct+exp_id)
    plot_condition_level_insights(df, occ, outdir)

    logger.info(f"Saved all plots/tables to: {outdir.resolve()}")


if __name__ == "__main__":
    main()
