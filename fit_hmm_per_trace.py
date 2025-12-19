#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Fit per-trajectory HMMs (hmmlearn) to smFRET trajectories exported as CSV:
columns: time_s, E, S
"""

import argparse
import logging
from pathlib import Path
from typing import Dict, Tuple, List

import numpy as np
import pandas as pd

from hmmlearn.hmm import GaussianHMM, GMMHMM

# -----------------------
# Logging
# -----------------------
logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger("hmm_fit")


# -----------------------
# Utilities
# -----------------------
def safe_read_csv(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    if "time_s" not in df.columns or "E" not in df.columns:
        raise ValueError(f"{path.name}: expected columns time_s, E")
    df = df.sort_values("time_s").reset_index(drop=True)
    return df


def preprocess_trace(df: pd.DataFrame,
                     e_min: float,
                     e_max: float,
                     s_min: float | None,
                     s_max: float | None) -> Tuple[np.ndarray, np.ndarray]:
    """
    Return (t, E) after filtering invalid points.
    """
    t = df["time_s"].to_numpy(dtype=float)
    E = df["E"].to_numpy(dtype=float)

    mask = np.isfinite(t) & np.isfinite(E) & (E >= e_min) & (E <= e_max)

    if (s_min is not None or s_max is not None) and "S" in df.columns:
        S = df["S"].to_numpy(dtype=float)
        sm = np.isfinite(S)
        if s_min is not None:
            sm &= (S >= s_min)
        if s_max is not None:
            sm &= (S <= s_max)
        mask &= sm

    t = t[mask]
    E = E[mask]
    return t, E


def init_means_from_quantiles(x: np.ndarray, n_states: int) -> np.ndarray:
    """
    Robust initializer: place means at quantiles of E.
    """
    qs = np.linspace(0.15, 0.85, n_states)
    return np.quantile(x, qs).reshape(-1, 1)


def fit_hmm_1d(E: np.ndarray,
               n_states: int,
               random_state: int,
               n_iter: int,
               tol: float,
               use_gmm: bool = True,
               n_mix: int = 2):
    X = E.reshape(-1, 1)
    K = n_states

    # ----- common initial trans/start (sticky) -----
    startprob = np.full(K, 1.0 / K)
    stay = 0.98
    trans = np.full((K, K), (1.0 - stay) / (K - 1))
    np.fill_diagonal(trans, stay)

    var = np.var(E)
    var = float(var) if np.isfinite(var) and var > 1e-6 else 1e-3

    if not use_gmm:
        model = GaussianHMM(
            n_components=K,
            covariance_type="diag",
            min_covar=1e-6,
            n_iter=n_iter,
            tol=tol,
            random_state=random_state,
            verbose=False,
            init_params="",     # IMPORTANT: don't overwrite what we set
            params="stmc",
        )
        model.startprob_ = startprob
        model.transmat_ = trans
        model.means_ = init_means_from_quantiles(E, K)         # (K,1)
        model.covars_ = np.full((K, 1), var)                   # (K,1)
        model.fit(X)
        return model

    # ----- GMMHMM proper shapes -----
    M = n_mix
    model = GMMHMM(
        n_components=K,
        n_mix=M,
        covariance_type="diag",
        min_covar=1e-6,
        n_iter=n_iter,
        tol=tol,
        random_state=random_state,
        verbose=False,
        init_params="",     # IMPORTANT: do NOT let hmmlearn re-init
        params="stmcw",
    )

    model.startprob_ = startprob
    model.transmat_ = trans

    # state anchors from quantiles (K,1)
    mu_state = init_means_from_quantiles(E, K).reshape(K)

    # build (K,M,1) means: split each state into M nearby mixture means
    # small spread around the state mean so EM can separate tails
    offsets = np.linspace(-0.03, 0.03, M)  # tweakable
    means = np.zeros((K, M, 1), dtype=float)
    for k in range(K):
        means[k, :, 0] = np.clip(mu_state[k] + offsets, 0.0, 1.0)
    model.means_ = means

    # mixture weights uniform
    model.weights_ = np.full((K, M), 1.0 / M)

    # covars (K,M,1) start from global variance
    model.covars_ = np.full((K, M, 1), var)

    model.fit(X)
    return model

def bic_score(model, X: np.ndarray) -> float:
    K = model.n_components
    logL = model.score(X)
    N = X.shape[0]

    if hasattr(model, "weights_"):  # GMMHMM
        M = model.n_mix
        k = (K - 1) + K * (K - 1) + K * (M - 1) + 2 * K * M
    else:  # GaussianHMM
        k = (K - 1) + K * (K - 1) + K + K

    return float(-2.0 * logL + k * np.log(max(N, 1)))

def state_means_1d(model) -> np.ndarray:
    if hasattr(model, "weights_"):  # GMMHMM
        mu = model.means_[..., 0]        # (K,M)
        w = model.weights_               # (K,M)
        return (w * mu).sum(axis=1)      # (K,)
    else:  # GaussianHMM
        return model.means_.reshape(-1)  # (K,)

def order_states_by_mean(model) -> Dict[int, int]:
    mu_state = state_means_1d(model)
    order = np.argsort(mu_state)
    return {int(old): int(new) for new, old in enumerate(order)}

def remap_state_sequence(states: np.ndarray, mapping: Dict[int, int]) -> np.ndarray:
    return np.array([mapping[int(s)] for s in states], dtype=int)


def compute_dwells(t: np.ndarray, states: np.ndarray) -> pd.DataFrame:
    """
    Compute dwell times per contiguous run of states.
    Dwell time is (t[end] - t[start]) + dt, where dt is median step.
    """
    if len(t) < 2:
        return pd.DataFrame(columns=["state", "t_start", "t_end", "dwell_s", "n_points"])

    dt = np.nanmedian(np.diff(t))
    dt = float(dt) if np.isfinite(dt) and dt > 0 else 0.0

    runs = []
    s0 = states[0]
    i0 = 0
    for i in range(1, len(states)):
        if states[i] != s0:
            i1 = i - 1
            dwell = (t[i1] - t[i0]) + dt
            runs.append((int(s0), float(t[i0]), float(t[i1]), float(dwell), int(i1 - i0 + 1)))
            s0 = states[i]
            i0 = i
    # last run
    i1 = len(states) - 1
    dwell = (t[i1] - t[i0]) + dt
    runs.append((int(s0), float(t[i0]), float(t[i1]), float(dwell), int(i1 - i0 + 1)))

    return pd.DataFrame(runs, columns=["state", "t_start", "t_end", "dwell_s", "n_points"])


# -----------------------
# Main
# -----------------------
def main():
    ap = argparse.ArgumentParser(description="Fit per-trajectory HMMs with hmmlearn (Gaussian emissions on E).")
    ap.add_argument("--indir", type=Path, default=Path("data/timeseries"),
                    help="Directory containing per-particle CSVs (time_s, E, S).")
    ap.add_argument("--outdir", type=Path, default=Path("results_hmm"),
                    help="Output directory.")
    ap.add_argument("--min-len", type=int, default=80,
                    help="Minimum number of points per trajectory to fit an HMM.")
    ap.add_argument("--states", type=str, default="2,3",
                    help="Comma-separated candidate state counts, e.g. '2' or '2,3'.")
    ap.add_argument("--use-bic", action="store_true",
                    help="Select best state count by BIC (otherwise uses first in --states).")
    ap.add_argument("--use-gmm", action="store_true",
                    help="Use GMMHMM (mixture of Gaussians per state) instead of GaussianHMM.")
    ap.add_argument("--e-min", type=float, default=0.0)
    ap.add_argument("--e-max", type=float, default=1.0)
    ap.add_argument("--s-min", type=float, default=None)
    ap.add_argument("--s-max", type=float, default=None)
    ap.add_argument("--n-iter", type=int, default=300)
    ap.add_argument("--tol", type=float, default=1e-4)
    ap.add_argument("--seed", type=int, default=72)

    args = ap.parse_args()

    outdir = args.outdir
    outdir.mkdir(parents=True, exist_ok=True)
    (outdir / "statepaths").mkdir(parents=True, exist_ok=True)
    (outdir / "dwells").mkdir(parents=True, exist_ok=True)

    # collect CSVs (exclude combined matrix)
    csvs = sorted([p for p in args.indir.glob("*.csv") if p.name != "fret_matrix.csv"])
    if not csvs:
        logger.error(f"No per-trajectory CSVs found in {args.indir}")
        return

    state_candidates = [int(x.strip()) for x in args.states.split(",") if x.strip()]
    if not state_candidates:
        raise ValueError("No valid --states provided")

    summary_rows: List[dict] = []

    for f in csvs:
        try:
            df = safe_read_csv(f)
        except Exception as e:
            logger.info(f"[skip] {f.name}: read error: {e}")
            continue

        t, E = preprocess_trace(df, args.e_min, args.e_max, args.s_min, args.s_max)
        if len(E) < args.min_len:
            logger.info(f"[skip] {f.name}: too short after filtering (n={len(E)})")
            continue

        X = E.reshape(-1, 1)

        best = None
        best_bic = np.inf
        best_k = None

        for k in state_candidates:
            try:
                model = fit_hmm_1d(E, n_states=k, random_state=args.seed, n_iter=args.n_iter, tol=args.tol, use_gmm=args.use_gmm)
                bic = bic_score(model, X)
            except Exception as e:
                logger.info(f"[fail] {f.name}: K={k} fit error: {e}")
                continue

            if (not args.use_bic and best is None) or (args.use_bic and bic < best_bic):
                best = model
                best_bic = bic
                best_k = k

        if best is None or best_k is None:
            logger.info(f"[skip] {f.name}: all candidate fits failed")
            continue

        # Viterbi decode
        states = best.predict(X)

        # reorder states by increasing mean E to make "state 0=open-ish, state last=closed-ish"
        mapping = order_states_by_mean(best)
        states_ord = remap_state_sequence(states, mapping)

        mu_state = state_means_1d(best)
        inv = np.argsort(mu_state)

        # report a single "effective" mean/var per hidden state
        means_ord = mu_state[inv]

        if hasattr(best, "weights_"):  # GMMHMM: effective variance = E[var] + Var[E]
            mu = best.means_[..., 0]  # (K,M)
            w = best.weights_  # (K,M)
            var_m = best.covars_[..., 0]  # (K,M)

            Ex2 = (w * (var_m + mu ** 2)).sum(axis=1)  # E[X^2]
            Ex = (w * mu).sum(axis=1)  # E[X]
            var_state = Ex2 - Ex ** 2
            covs_ord = var_state[inv]
        else:
            covs_ord = best.covars_.reshape(-1)[inv]

        # NOTE: transmat reordering: P(new_i -> new_j) = P(old_inv[i] -> old_inv[j])
        if hasattr(best, "weights_"):  # GMMHMM
            mu = best.means_[..., 0]  # (K, M)
            w = best.weights_  # (K, M)
            mu_state = (w * mu).sum(axis=1)  # (K,)
            inv = np.argsort(mu_state)
        else:  # GaussianHMM
            inv = np.argsort(best.means_.reshape(-1))

        trans_ord = best.transmat_[inv][:, inv]
        start_ord = best.startprob_[inv]

        # save statepath
        sp = pd.DataFrame({"time_s": t, "E": E, "state": states_ord})
        sp.to_csv(outdir / "statepaths" / f"{f.stem}.statepath.csv", index=False)

        # dwell stats
        dw = compute_dwells(t, states_ord)
        dw.to_csv(outdir / "dwells" / f"{f.stem}.dwells.csv", index=False)

        # summary row
        row = {
            "trajectory": f.name,
            "n_points": int(len(E)),
            "n_states": int(best_k),
            "logL": float(best.score(X)),
            "bic": float(best_bic),
        }

        # store ordered means/vars
        for i in range(best_k):
            row[f"mu_{i}"] = float(means_ord[i])
            row[f"var_{i}"] = float(covs_ord[i])

        # store flattened startprob + transmat (ordered)
        for i in range(best_k):
            row[f"pi_{i}"] = float(start_ord[i])
        for i in range(best_k):
            for j in range(best_k):
                row[f"T_{i}_{j}"] = float(trans_ord[i, j])

        # dwell summary per state
        if not dw.empty:
            for i in range(best_k):
                dwi = dw.loc[dw["state"] == i, "dwell_s"].to_numpy(dtype=float)
                row[f"dwell_mean_{i}"] = float(np.nanmean(dwi)) if dwi.size else np.nan
                row[f"dwell_med_{i}"] = float(np.nanmedian(dwi)) if dwi.size else np.nan
                row[f"dwell_n_{i}"] = int(dwi.size)
        else:
            for i in range(best_k):
                row[f"dwell_mean_{i}"] = np.nan
                row[f"dwell_med_{i}"] = np.nan
                row[f"dwell_n_{i}"] = 0

        summary_rows.append(row)
        logger.info(f"[ok] {f.name}: K={best_k}, BIC={best_bic:.2f}, means={means_ord}")

    if not summary_rows:
        logger.error("No trajectories were successfully fit.")
        return

    summary = pd.DataFrame(summary_rows)
    summary = summary.sort_values(["n_states", "bic", "trajectory"]).reset_index(drop=True)
    summary.to_csv(outdir / "hmm_summary.csv", index=False)
    logger.info(f"\nSaved: {outdir / 'hmm_summary.csv'}")
    logger.info(f"Saved statepaths -> {outdir / 'statepaths'}")
    logger.info(f"Saved dwells     -> {outdir / 'dwells'}")


if __name__ == "__main__":
    main()
