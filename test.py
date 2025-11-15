#!/usr/bin/env python3
"""
Hsp90 3-State Conformational Dynamics Analysis Script
=====================================================

Description:
This script performs kinetic analysis of Hsp90 single-molecule FRET data.
It fits a 3-state conformational model (Open <-> Intermediate <-> Closed)
that includes irreversible photobleaching and a static (non-dynamic) subpopulation.

Key features:
- Conformational states: Open (O), Intermediate (I), Closed (C).
- Irreversible bleaching from all active states to a dark state (B).
- Explicit modelling of a static fraction (f_static = 1 - f_dyn).
- High-performance ODE solving using Numba JIT compilation.
- Parallelized batch fitting across different experimental conditions.
- Parallelized bootstrapping for robust parameter uncertainty estimation.

Usage:
Ensure 'data/timeseries/combined_fret_matrix.csv' exists, then run:
$ python hsp90_analysis.csv

Author: Abhinav Mishra (mishraabhinav36@gmail.com)
Date: November 2025
License: BSD 3-Clause
"""

# ----------------------------------------------------------------------
# License (BSD 3-Clause)
# ----------------------------------------------------------------------
# Copyright (c) 2025, Abhinav Mishra
# All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# 1. Redistributions of source code must retain the above copyright notice, this
#    list of conditions and the following disclaimer.
#
# 2. Redistributions in binary form must reproduce the above copyright notice,
#    this list of conditions and the following disclaimer in the documentation
#    and/or other materials provided with the distribution.
#
# 3. Neither the name of the copyright holder nor the names of its
#    contributors may be used to endorse or promote products derived from
#    this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
# ----------------------------------------------------------------------

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from dataclasses import dataclass
from pathlib import Path
from typing import Tuple, List, Optional, Dict, Any
from scipy.integrate import solve_ivp
from scipy.optimize import curve_fit
from numba import njit
from joblib import Parallel, delayed


# ----------------------------------------------------------------------
# Data Structures for Model
# ----------------------------------------------------------------------

@dataclass
class Hsp90Params3State:
    """
    Data class holding parameters for the 3-state Hsp90 model with bleaching.

    States:
        O (Open) <-> I (Intermediate) <-> C (Closed)
        All states can bleach irreversibly to B (Bleached/Dark) with rate k_B.

    Attributes
    ----------
    k_OI : float
        Rate constant for Open -> Intermediate (1/s).
    k_IO : float
        Rate constant for Intermediate -> Open (1/s).
    k_IC : float
        Rate constant for Intermediate -> Closed (1/s).
    k_CI : float
        Rate constant for Closed -> Intermediate (1/s).
    k_B : float
        Global photobleaching rate constant (1/s).
    E_open : float
        Mean FRET efficiency of the Open state.
    E_inter : float
        Mean FRET efficiency of the Intermediate state.
    E_closed : float
        Mean FRET efficiency of the Closed state.
    P_O0 : float
        Initial probability of being in the Open state at t=0.
    P_C0 : float
        Initial probability of being in the Closed state at t=0.
        (Note: P_I0 is derived as 1.0 - P_O0 - P_C0).
    """
    k_OI: float
    k_IO: float
    k_IC: float
    k_CI: float
    k_B: float
    E_open: float
    E_inter: float
    E_closed: float
    P_O0: float
    P_C0: float


@dataclass
class Hsp90Fit3State:
    """
    Container for the complete fit results, including the dynamic model
    parameters and the static subpopulation fraction.

    Attributes
    ----------
    params : Hsp90Params3State
        The fitted kinetic and FRET parameters for the dynamic population.
    f_dyn : float
        Fraction of molecules that are dynamic (0.0 to 1.0).
    E_static : float
        Apparent FRET efficiency of the static (non-dynamic) population.
    """
    params: Hsp90Params3State
    f_dyn: float
    E_static: float


# ----------------------------------------------------------------------
# Core Physics / ODEs (JIT Optimized)
# ----------------------------------------------------------------------

@njit("float64[:](float64, float64[:], float64[:])", cache=True, fastmath=True)
def rhs_hsp90_numba(t: float, y: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    JIT-compiled ODE right-hand side for the 3-state system + bleaching.

    dP_O/dt = -k_OI*P_O + k_IO*P_I - k_B*P_O
    dP_I/dt = k_OI*P_O - (k_IO + k_IC + k_B)*P_I + k_CI*P_C
    dP_C/dt = k_IC*P_I - k_CI*P_C - k_B*P_C

    Parameters
    ----------
    t : float
        Current time (required by ODE solver signature, unused here if autonomous).
    y : np.ndarray
        Current state vector [P_O, P_I, P_C].
    params : np.ndarray
        Kinetic parameters [k_OI, k_IO, k_IC, k_CI, k_B].

    Returns
    -------
    np.ndarray
        Derivatives [dP_O/dt, dP_I/dt, dP_C/dt].
    """
    k_OI, k_IO, k_IC, k_CI, k_B = params[0], params[1], params[2], params[3], params[4]
    P_O, P_I, P_C = y[0], y[1], y[2]

    dP_O = -k_OI * P_O + k_IO * P_I - k_B * P_O
    dP_I = k_OI * P_O - (k_IO + k_IC + k_B) * P_I + k_CI * P_C
    dP_C = k_IC * P_I - k_CI * P_C - k_B * P_C

    return np.array([dP_O, dP_I, dP_C], dtype=np.float64)


def model_fret_3state(t_eval: np.ndarray, p: Hsp90Params3State) -> np.ndarray:
    """
    Calculate the time-dependent FRET efficiency for the dynamic population only.

    Solves the ODEs and computes: E_dyn(t) = E_O*P_O(t) + E_I*P_I(t) + E_C*P_C(t).

    Parameters
    ----------
    t_eval : np.ndarray
        Time points at which to evaluate the model.
    p : Hsp90Params3State
        Model parameters.

    Returns
    -------
    np.ndarray
        Predicted FRET efficiency trajectory for the dynamic fraction.
    """
    P_O0 = p.P_O0
    P_C0 = p.P_C0
    P_I0 = 1.0 - P_O0 - P_C0

    # Ensure initial probabilities are physically valid (crudely normalize if needed)
    if P_I0 < 0.0:
        total = max(P_O0 + P_C0, 1e-9)
        P_O0 = (P_O0 / total) * 0.999
        P_C0 = (P_C0 / total) * 0.001
        P_I0 = 1.0 - P_O0 - P_C0

    # Prepare inputs for Numba/ODE solver (must be explicitly float64)
    y0 = np.array([P_O0, P_I0, P_C0], dtype=np.float64)
    k_params = np.array([p.k_OI, p.k_IO, p.k_IC, p.k_CI, p.k_B], dtype=np.float64)

    sol = solve_ivp(
        fun=rhs_hsp90_numba,
        t_span=(t_eval.min(), t_eval.max()),
        y0=y0,
        t_eval=t_eval,
        args=(k_params,),  # Passed as a tuple containing one array
        vectorized=False,
        method='RK45',  # RK45 is generally robust for these stiff-ish systems
        # atol=1e-7, rtol=1e-7  # Uncomment for higher precision if needed
    )

    if not sol.success:
        return np.full_like(t_eval, np.nan, dtype=float)

    # Calculate observed FRET from state probabilities
    E_t = (p.E_open * sol.y[0] +
           p.E_inter * sol.y[1] +
           p.E_closed * sol.y[2])

    return E_t


def model_total_fret(t_eval: np.ndarray, fit: Hsp90Fit3State) -> np.ndarray:
    """
    Calculate the total observed ensemble FRET efficiency.

    Combines the dynamic population with the static subpopulation:
    E_total(t) = f_dyn * E_dyn(t) + (1 - f_dyn) * E_static

    Parameters
    ----------
    t_eval : np.ndarray
        Time points.
    fit : Hsp90Fit3State
        Complete fit parameters including static fraction.

    Returns
    -------
    np.ndarray
        Total predicted ensemble FRET.
    """
    E_dyn = model_fret_3state(t_eval, fit.params)
    return fit.f_dyn * E_dyn + (1.0 - fit.f_dyn) * fit.E_static


# ----------------------------------------------------------------------
# Data Handling Utilities
# ----------------------------------------------------------------------

def load_combined_matrix(path: Path) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    """
    Load the combined FRET matrix from a CSV file.

    Parameters
    ----------
    path : Path
        Path to the 'combined_fret_matrix.csv' file.

    Returns
    -------
    t : np.ndarray
        (T,) array of time points.
    E_mat : np.ndarray
        (T, N_traj) matrix of FRET trajectories.
    traj_cols : List[str]
        List of column names corresponding to the trajectories in E_mat.
    """
    df = pd.read_csv(path)
    if "time_s" not in df.columns:
        raise ValueError("Expected a 'time_s' column in the combined matrix.")

    t = df["time_s"].values
    traj_cols = [c for c in df.columns if c != "time_s"]
    E_mat = df[traj_cols].to_numpy()

    # Keep only rows where at least one trajectory has data
    row_valid = np.isfinite(E_mat).any(axis=1)
    return t[row_valid], E_mat[row_valid, :], traj_cols


def parse_column_metadata(col_names: List[str]) -> pd.DataFrame:
    """
    Parse trajectory column names into structured metadata.

    Assumes format: <construct>_<exp_id>_p<particle_id>
    e.g., "Hsp90_409_601_241107_p00001"

    Parameters
    ----------
    col_names : List[str]
        List of trajectory column strings.

    Returns
    -------
    pd.DataFrame
        Metadata with columns: [col, construct, exp_id, particle, condition].
    """
    records = []
    for c in col_names:
        if c == "time_s": continue
        parts = c.split("_")
        if len(parts) < 3:
            # Fallback for non-standard names
            records.append((c, c, "unknown", "unknown", f"{c}_unknown"))
        else:
            particle = parts[-1]
            exp_id = parts[-2]
            construct = "_".join(parts[:-2])
            condition = f"{construct}_{exp_id}"
            records.append((c, construct, exp_id, particle, condition))

    return pd.DataFrame(records, columns=["col", "construct", "exp_id", "particle", "condition"])


def subset_matrix_by_columns(
        t: np.ndarray, E_mat: np.ndarray, all_cols: List[str], cols_to_keep: List[str]
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Extract a subset of trajectories from the main matrix.

    Parameters
    ----------
    t : np.ndarray
        Global time vector.
    E_mat : np.ndarray
        Global data matrix.
    all_cols : List[str]
        List of all column names in E_mat.
    cols_to_keep : List[str]
        List of column names to extract.

    Returns
    -------
    t_sub : np.ndarray
        Subset time vector (rows with no data in subset are removed).
    E_sub : np.ndarray
        Subset data matrix.
    """
    name_to_idx = {name: i for i, name in enumerate(all_cols)}
    indices = [name_to_idx[c] for c in cols_to_keep if c in name_to_idx]

    if not indices:
        raise ValueError("No matching columns found for subset.")

    E_sub_full = E_mat[:, indices]
    # Remove time points where this specific subset has absolutely no data
    valid_rows = np.isfinite(E_sub_full).any(axis=1)

    return t[valid_rows], E_sub_full[valid_rows, :]


def compute_ensemble_metrics(t: np.ndarray, E_mat: np.ndarray, fit: Hsp90Fit3State) -> Dict[str, Any]:
    """
    Calculate RMSE and R-squared for the fit against the ensemble mean of the data.
    """
    E_mean = np.nanmean(E_mat, axis=1)
    E_pred = model_total_fret(t, fit)

    # Only evaluate where we have both data and prediction
    mask = np.isfinite(E_mean) & np.isfinite(E_pred)
    if not np.any(mask):
        return {"rmse": np.nan, "r2": np.nan, "n_time": 0, "n_traj": E_mat.shape[1]}

    E_obs = E_mean[mask]
    E_mod = E_pred[mask]

    residuals = E_obs - E_mod
    rmse = np.sqrt(np.mean(residuals ** 2))

    ss_res = np.sum(residuals ** 2)
    ss_tot = np.sum((E_obs - np.mean(E_obs)) ** 2)
    r2 = 1.0 - (ss_res / ss_tot) if ss_tot > 1e-12 else np.nan

    return {
        "rmse": float(rmse), "r2": float(r2),
        "n_time": int(np.sum(mask)), "n_traj": int(E_mat.shape[1])
    }


# ----------------------------------------------------------------------
# Fitting Routines (Global & Parallel Workers)
# ----------------------------------------------------------------------

def fit_global_3state(
        t: np.ndarray, E_mat: np.ndarray, p0: Optional[np.ndarray] = None
) -> Hsp90Fit3State:
    """
    Fit the 3-state model to the ensemble mean of the provided data matrix.

    Parameters
    ----------
    t : np.ndarray
        Time vector.
    E_mat : np.ndarray
        FRET data matrix (trajectories in columns).
    p0 : np.ndarray, optional
        Initial guess for parameters.
        Order: [k_OI, k_IO, k_IC, k_CI, k_B, E_o, E_i, E_c, P_O0, P_C0, f_dyn, E_stat]

    Returns
    -------
    Hsp90Fit3State
        Fitted model object.
    """
    # Calculate ensemble mean for fitting
    E_mean = np.nanmean(E_mat, axis=1)
    mask = np.isfinite(E_mean)
    t_fit = t[mask]
    E_fit = E_mean[mask]

    if t_fit.size == 0:
        raise RuntimeError("No valid ensemble mean data for fitting.")

    # Default initial guess if none provided
    if p0 is None:
        p0 = np.array([
            0.1, 0.1, 0.1, 0.1,  # k_OI, k_IO, k_IC, k_CI
            0.005,  # k_B
            0.15, 0.55, 0.85,  # E_open, E_inter, E_closed
            0.8, 0.1,  # P_O0, P_C0 (implies P_I0 ~ 0.1)
            0.7, 0.2  # f_dyn, E_static
        ])

    # Parameter bounds (all must be positive; probabilities/E must be <= 1.0)
    lower_bounds = [0.0] * 12
    upper_bounds = [10.0] * 5 + [1.0] * 7  # Rates up to 10/s, others up to 1.0

    def _fit_wrapper(t_in, *args):
        # Unpack args used by curve_fit back into model parameters
        params = Hsp90Params3State(
            k_OI=args[0], k_IO=args[1], k_IC=args[2], k_CI=args[3], k_B=args[4],
            E_open=args[5], E_inter=args[6], E_closed=args[7],
            P_O0=args[8], P_C0=args[9]
        )
        f_d, E_s = args[10], args[11]
        E_dyn = model_fret_3state(t_in, params)
        return f_d * E_dyn + (1.0 - f_d) * E_s

    popt, _ = curve_fit(
        _fit_wrapper, t_fit, E_fit, p0=p0, bounds=(lower_bounds, upper_bounds), maxfev=10000
    )

    return Hsp90Fit3State(
        params=Hsp90Params3State(*popt[:10]),
        f_dyn=popt[10],
        E_static=popt[11]
    )


def _fit_single_condition_worker(
        key: str, t: np.ndarray, E_mat: np.ndarray, col_names: List[str],
        meta: pd.DataFrame, group_by: str
) -> Optional[Tuple[str, Hsp90Fit3State, Dict[str, Any]]]:
    """
    Worker function for parallel condition fitting.
    """
    cols = meta.loc[meta[group_by] == key, "col"].tolist()
    if not cols: return None

    try:
        t_sub, E_sub = subset_matrix_by_columns(t, E_mat, col_names, cols)
        fit = fit_global_3state(t_sub, E_sub)
        mets = compute_ensemble_metrics(t_sub, E_sub, fit)
        p = fit.params

        record = {
            "group_key": key, "n_traj": mets["n_traj"], "rmse": mets["rmse"], "r2": mets["r2"],
            "k_OI": p.k_OI, "k_IO": p.k_IO, "k_IC": p.k_IC, "k_CI": p.k_CI, "k_B": p.k_B,
            "E_open": p.E_open, "E_inter": p.E_inter, "E_closed": p.E_closed,
            "f_dyn": fit.f_dyn, "E_static": fit.E_static
        }
        return (key, fit, record)
    except Exception as e:
        print(f"FAILED: {key} - {e}")
        return None


def fit_all_conditions(
        t: np.ndarray, E_mat: np.ndarray, col_names: List[str],
        group_by: str = "condition", do_plots: bool = False
) -> Tuple[pd.DataFrame, Dict[str, Hsp90Fit3State]]:
    """
    Fit the model to all conditions in parallel.

    Parameters
    ----------
    t, E_mat, col_names : Data inputs.
    group_by : str
        Metadata column to group by (e.g., "condition", "construct").
    do_plots : bool
        If True, generates time-series plots for each condition sequentially after fitting.

    Returns
    -------
    summary_df : pd.DataFrame
        Table of fitted parameters for all conditions.
    fit_dict : dict
        Dictionary mapping group keys to their Hsp90Fit3State objects.
    """
    meta = parse_column_metadata(col_names)
    keys = sorted(meta[group_by].unique())
    print(f"Fitting {len(keys)} conditions in parallel (group_by='{group_by}')...")

    results = Parallel(n_jobs=-1, verbose=5)(
        delayed(_fit_single_condition_worker)(k, t, E_mat, col_names, meta, group_by)
        for k in keys
    )

    fit_dict = {}
    records = []
    for res in results:
        if res:
            key, fit, rec = res
            fit_dict[key] = fit
            records.append(rec)

    summary_df = pd.DataFrame(records) if records else pd.DataFrame()

    if do_plots and fit_dict:
        print("Generating plots...")
        for key, fit in fit_dict.items():
            cols = meta.loc[meta[group_by] == key, "col"].tolist()
            t_sub, E_sub = subset_matrix_by_columns(t, E_mat, col_names, cols)
            plot_hsp90_fit_time(t_sub, E_sub, fit)  # Assumes this function exists below

    return summary_df, fit_dict


def _bootstrap_worker(t_sub: np.ndarray, E_sub: np.ndarray, seed: int) -> Optional[Dict[str, float]]:
    """
    Worker function for a single bootstrap iteration.
    """
    rng = np.random.default_rng(seed)
    # Resample trajectories with replacement
    idx = rng.integers(0, E_sub.shape[1], size=E_sub.shape[1])
    E_boot = E_sub[:, idx]
    try:
        fit = fit_global_3state(t_sub, E_boot)
        p = fit.params
        return {
            "k_OI": p.k_OI, "k_IO": p.k_IO, "k_IC": p.k_IC, "k_CI": p.k_CI, "k_B": p.k_B,
            "E_open": p.E_open, "E_inter": p.E_inter, "E_closed": p.E_closed,
            "f_dyn": fit.f_dyn, "E_static": fit.E_static
        }
    except:
        return None


def bootstrap_condition_params(
        t: np.ndarray, E_mat: np.ndarray, col_names: List[str], meta: pd.DataFrame,
        group_key: str, group_by: str = "condition", n_boot: int = 100, seed_start: int = 0
) -> pd.DataFrame:
    """
    Perform parallel bootstrapping of trajectories for a specific condition to estimate parameter uncertainty.
    """
    cols = meta.loc[meta[group_by] == group_key, "col"].tolist()
    if not cols: raise ValueError(f"No data for {group_key}")

    t_sub, E_sub = subset_matrix_by_columns(t, E_mat, col_names, cols)
    print(f"Bootstrapping {group_key} ({n_boot} iterations)...")

    results = Parallel(n_jobs=-1, verbose=5)(
        delayed(_bootstrap_worker)(t_sub, E_sub, seed_start + i) for i in range(n_boot)
    )
    return pd.DataFrame([r for r in results if r is not None])


# ----------------------------------------------------------------------
# Plotting & Visualization
# ----------------------------------------------------------------------

def plot_ensemble_fit(t: np.ndarray, E_mat: np.ndarray, fit: Hsp90Fit3State) -> None:
    """Plot observed ensemble mean vs model prediction correlation."""
    E_mean = np.nanmean(E_mat, axis=1)
    E_pred = model_total_fret(t, fit)
    mask = np.isfinite(E_mean) & np.isfinite(E_pred)

    plt.figure(figsize=(5, 5))
    plt.scatter(E_pred[mask], E_mean[mask], alpha=0.5, s=10)
    lims = [np.min(E_pred), np.max(E_pred)]
    plt.plot(lims, lims, 'k--', alpha=0.75)
    plt.xlabel("Model FRET")
    plt.ylabel("Data Mean FRET")
    plt.title("Ensemble Fit Correlation")
    plt.tight_layout()
    plt.show()


def plot_hsp90_fit_time(
        t: np.ndarray, E_mat: np.ndarray, fit: Hsp90Fit3State, n_traces_overlay: int = 100, random_seed: int = 42
) -> None:
    """Plot FRET time-series: individual traces (faded), ensemble mean, and model fit."""
    E_mean = np.nanmean(E_mat, axis=1)
    E_pred = model_total_fret(t, fit)

    plt.figure(figsize=(10, 6))
    # Overlay generic individual traces if available
    if n_traces_overlay > 0 and E_mat.shape[1] > 0:
        rng = np.random.default_rng(random_seed)
        idx = rng.choice(E_mat.shape[1], size=min(n_traces_overlay, E_mat.shape[1]), replace=False)
        plt.plot(t, E_mat[:, idx], color='gray', alpha=0.05)

    plt.plot(t, E_mean, 'b-', lw=2, label='Ensemble Mean Data', alpha=0.8)
    plt.plot(t, E_pred, 'r--', lw=2.5, label='3-State Model Fit')
    plt.xlabel('Time (s)')
    plt.ylabel('FRET Efficiency')
    plt.legend()
    plt.title(f"Kinetic Fit (N={E_mat.shape[1]} trajectories)")
    plt.tight_layout()
    plt.show()


def plot_bootstrap_compare(df_a: pd.DataFrame, df_b: pd.DataFrame, param: str, label_a: str, label_b: str) -> None:
    """Compare bootstrap distributions for a parameter between two conditions."""
    plt.figure()
    plt.hist(df_a[param], bins=20, alpha=0.5, density=True, label=label_a)
    plt.hist(df_b[param], bins=20, alpha=0.5, density=True, label=label_b)
    plt.xlabel(param)
    plt.ylabel("Density")
    plt.legend()
    plt.title(f"Bootstrap Comparison: {param}")
    plt.tight_layout()
    plt.show()


# ----------------------------------------------------------------------
# Main Execution Block
# ----------------------------------------------------------------------

def main():
    data_path = Path("data/timeseries/fret_matrix.csv")
    if not data_path.exists():
        print(f"Data file not found at {data_path}")
        return

    # 1. Load Data
    t, E_mat, col_names = load_combined_matrix(data_path)
    print(f"Loaded data: {E_mat.shape} (Time x Trajectories)")

    # 2. Global Fit (All data pooled)
    print("\n--- Running Global Fit ---")
    global_fit = fit_global_3state(t, E_mat)
    print("Global Params:", global_fit.params)
    plot_ensemble_fit(t, E_mat, global_fit)
    plot_hsp90_fit_time(t, E_mat, global_fit)

    # 3. Parallel Fit by Condition
    print("\n--- Running Parallel Condition Fits ---")
    df_cond, _ = fit_all_conditions(t, E_mat, col_names, group_by="condition", do_plots=True)
    print("\nCondition Summary:\n", df_cond)

    # 4. Example Bootstrap Comparison (Update condition names as needed for your data)
    meta = parse_column_metadata(col_names)
    conditions = meta['condition'].unique()
    if len(conditions) >= 2:
        cond_a, cond_b = conditions[0], conditions[1]
        print(f"\n--- Bootstrapping Comparison: {cond_a} vs {cond_b} ---")
        boot_a = bootstrap_condition_params(t, E_mat, col_names, meta, cond_a, n_boot=20)
        boot_b = bootstrap_condition_params(t, E_mat, col_names, meta, cond_b, n_boot=20)

        plot_bootstrap_compare(boot_a, boot_b, "k_OI", cond_a, cond_b)
        plot_bootstrap_compare(boot_a, boot_b, "f_dyn", cond_a, cond_b)


if __name__ == "__main__":
    main()