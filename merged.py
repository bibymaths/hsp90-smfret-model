#!/usr/bin/env python3
"""
Hsp90 3-State Conformational Dynamics Analysis Script (Merged)
==============================================================

Description
-----------
Unified version of `pipeline.py` and `test.py` for kinetic analysis of
Hsp90 single-molecule FRET data.

Core model
----------
- Conformational states: Open (O), Intermediate (I), Closed (C).
- Irreversible bleaching from all fluorescent states to a dark state (B).
- Explicit dynamic fraction f_dyn and static fraction (1 - f_dyn).

Key capabilities
----------------
- Load combined FRET matrix from CSV (time x trajectories).
- Global fit (all trajectories pooled into ensemble mean).
- Parallel per-condition / per-construct fits with Joblib.
- Ensemble goodness-of-fit metrics: RMSE, R², n_time, n_traj.
- Rich plotting:
  - Ensemble correlation plot.
  - Time traces with mean ± SD and model overlay.
  - Parameter vs condition plots.
  - Bootstrap histogram comparisons between conditions.
- Bootstrap-based parameter uncertainty estimation per condition.

Expected combined matrix format
-------------------------------
- CSV file: first column is `time_s` (time in seconds).
- Remaining columns are trajectories, named with a pattern like:
  `construct_expID_condition_pXXXXX`, e.g.:

    Hsp90_409_601_241107_A_p00001

  where metadata fields can be parsed as:
  - construct
  - exp_id (or date)
  - condition (e.g. buffer, ligand)
  - pXXXXX (particle index; not used for grouping)

Adjust `parse_column_metadata` if your naming differs.
"""

# ----------------------------------------------------------------------
# BSD 3-Clause License
# ----------------------------------------------------------------------
# Copyright (c) 2025, Abhinav Mishra
# All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# 1. Redistributions of source code must retain the above copyright notice,
#    this list of conditions and the following disclaimer.
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
# Imports
# ----------------------------------------------------------------------
from __future__ import annotations

import numpy as np
import pandas as pd
from dataclasses import dataclass
from pathlib import Path
from typing import Tuple, List, Dict, Any, Optional

import matplotlib.pyplot as plt
from numba import njit
from joblib import Parallel, delayed
from scipy.integrate import solve_ivp
from scipy.optimize import curve_fit

# ----------------------------------------------------------------------
# Data classes
# ----------------------------------------------------------------------


@dataclass
class Hsp90Params3State:
    """
    Parameters for the 3-state Hsp90 model with bleaching.

    States:
        O (Open), I (Intermediate), C (Closed), B (Bleached).

    Transitions:
        O <-> I <-> C, and O/I/C -> B (irreversible).

    Attributes
    ----------
    k_OI, k_IO, k_IC, k_CI : float
        Conformational rate constants [1/s].
    k_B : float
        Bleaching rate (from each fluorescent state) [1/s].

    E_open, E_inter, E_closed : float
        FRET efficiencies of the Open, Intermediate, and Closed states.

    P_O0, P_C0 : float
        Initial probabilities of the Open and Closed states.
        Initial Intermediate probability is computed as:
            P_I0 = 1 - P_O0 - P_C0
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
    Container for fitted parameters and static fraction.

    Attributes
    ----------
    params : Hsp90Params3State
        Kinetic and FRET parameters.
    f_dyn : float
        Fraction of molecules following dynamic 3-state kinetics.
    E_static : float
        FRET level of static (non-dynamic) subpopulation.
    """
    params: Hsp90Params3State
    f_dyn: float
    E_static: float


# ----------------------------------------------------------------------
# ODE right-hand side: Python & Numba versions
# ----------------------------------------------------------------------


def rhs_hsp90_3state(t: float, y: np.ndarray, p: Hsp90Params3State) -> np.ndarray:
    """
    Pure-Python ODE RHS for Hsp90 3-state + bleaching.

    Parameters
    ----------
    t : float
        Time (unused for autonomous system).
    y : np.ndarray
        State vector [P_O, P_I, P_C].
    p : Hsp90Params3State
        Model parameters.

    Returns
    -------
    np.ndarray
        Derivatives [dP_O/dt, dP_I/dt, dP_C/dt].
    """
    P_O, P_I, P_C = y

    dP_O = -p.k_OI * P_O + p.k_IO * P_I - p.k_B * P_O
    dP_I = p.k_OI * P_O - (p.k_IO + p.k_IC + p.k_B) * P_I + p.k_CI * P_C
    dP_C = p.k_IC * P_I - p.k_CI * P_C - p.k_B * P_C

    return np.array([dP_O, dP_I, dP_C], dtype=np.float64)


@njit("float64[:](float64, float64[:], float64[:])", cache=True, fastmath=False, nogil=False)
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


# ----------------------------------------------------------------------
# Forward model: dynamic FRET & total FRET
# ----------------------------------------------------------------------


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
        Dynamic FRET trajectory E_dyn(t) evaluated at t_eval.
    """
    t_eval = np.asarray(t_eval, dtype=float)
    if t_eval.ndim != 1:
        raise ValueError("t_eval must be a 1D array of time points.")

    # Ensure probabilities are valid
    P_O0 = max(p.P_O0, 0.0)
    P_C0 = max(p.P_C0, 0.0)
    P_I0 = 1.0 - P_O0 - P_C0

    # Simple renormalization if rounding pushes sum out of [0,1]
    if P_I0 < 0.0:
        total = P_O0 + P_C0
        if total <= 0.0:
            P_O0, P_C0, P_I0 = 0.8, 0.1, 0.1
        else:
            P_O0 = P_O0 / total * 0.999
            P_C0 = P_C0 / total * 0.001
            P_I0 = 1.0 - P_O0 - P_C0

    y0 = np.array([P_O0, P_I0, P_C0], dtype=float)
    k_params = np.array([p.k_OI, p.k_IO, p.k_IC, p.k_CI, p.k_B], dtype=float)

    sol = solve_ivp(
        fun=rhs_hsp90_numba,
        t_span=(t_eval.min(), t_eval.max()),
        y0=y0,
        t_eval=t_eval,
        vectorized=False,
        args=(k_params,),
        # Uncomment if you need stricter tolerances:
        # atol=1e-7, rtol=1e-7
    )

    if not sol.success:
        return np.full_like(t_eval, np.nan, dtype=float)

    E_t = (p.E_open * sol.y[0] +
           p.E_inter * sol.y[1] +
           p.E_closed * sol.y[2])
    return E_t


def model_total_fret(t_eval: np.ndarray, fit: Hsp90Fit3State) -> np.ndarray:
    """
    Total observed FRET including dynamic and static fractions.

    E_total(t) = f_dyn * E_dyn(t) + (1 - f_dyn) * E_static
    """
    E_dyn = model_fret_3state(t_eval, fit.params)
    return fit.f_dyn * E_dyn + (1.0 - fit.f_dyn) * fit.E_static


# ----------------------------------------------------------------------
# I/O utilities
# ----------------------------------------------------------------------


def load_combined_matrix(path: Path) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    """
    Load combined FRET matrix from CSV.

    Parameters
    ----------
    path : Path
        Path to CSV file. First column is interpreted as time (seconds),
        remaining columns as trajectories.

    Returns
    -------
    t : np.ndarray
        Time vector (1D).
    E_mat : np.ndarray
        FRET matrix of shape (T, N) with possible NaNs.
    col_names : List[str]
        Column names for trajectories.
    """
    df = pd.read_csv(path)
    if df.shape[1] < 2:
        raise ValueError("Combined FRET matrix must have at least 2 columns (time + one trajectory).")

    t = df.iloc[:, 0].to_numpy(dtype=float)
    E_mat = df.iloc[:, 1:].to_numpy(dtype=float)
    col_names = df.columns[1:].tolist()
    return t, E_mat, col_names


def parse_column_metadata(col_names: List[str]) -> pd.DataFrame:
    """
    Parse metadata from trajectory column names.

    Expected pattern (but flexible):
        "<construct>_<exp_id>_<condition>_pXXXXX"

    Anything beyond the first three underscores is treated as the "particle" tag.

    Returns
    -------
    meta : DataFrame
        Columns: ["col", "construct", "exp_id", "condition", "particle"]
    """
    records = []
    for c in col_names:
        base = c
        if base.endswith(".csv"):
            base = base[:-4]

        parts = base.split("_")
        if len(parts) >= 4:
            construct = "_".join(parts[0:3])
            exp_id = parts[3]
            if len(parts) >= 5:
                condition = parts[4]
                particle = "_".join(parts[5:]) if len(parts) > 5 else ""
            else:
                condition = "unknown"
                particle = ""
        elif len(parts) >= 3:
            construct = parts[0]
            exp_id = parts[1]
            condition = parts[2]
            particle = "_".join(parts[3:]) if len(parts) > 3 else ""
        elif len(parts) >= 2:
            construct = parts[0]
            exp_id = parts[1]
            condition = "unknown"
            particle = "_".join(parts[2:]) if len(parts) > 2 else ""
        else:
            construct = "unknown"
            exp_id = "unknown"
            condition = "unknown"
            particle = ""

        records.append(
            dict(
                col=c,
                construct=construct,
                exp_id=exp_id,
                condition=condition,
                particle=particle,
            )
        )

    meta = pd.DataFrame(records)
    return meta


def subset_matrix_by_columns(
    t: np.ndarray,
    E_mat: np.ndarray,
    col_names: List[str],
    cols_subset: List[str],
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Extract sub-matrix for a subset of columns.

    Parameters
    ----------
    t : np.ndarray
        Global time vector (shared by all trajectories).
    E_mat : np.ndarray
        FRET matrix (T x N).
    col_names : List[str]
        Column names, length N.
    cols_subset : List[str]
        Subset of column names to keep.

    Returns
    -------
    t_sub : np.ndarray
        Same as `t`, provided for convenience.
    E_sub : np.ndarray
        Sub-matrix with just the selected trajectories.
    """
    name_to_idx = {name: j for j, name in enumerate(col_names)}
    idx = [name_to_idx[c] for c in cols_subset if c in name_to_idx]
    if not idx:
        raise ValueError("No matching columns for subset.")

    E_sub = E_mat[:, idx]
    return t, E_sub


# ----------------------------------------------------------------------
# Metrics
# ----------------------------------------------------------------------


def compute_ensemble_metrics(
    t: np.ndarray,
    E_mat: np.ndarray,
    fit: Hsp90Fit3State
) -> Dict[str, Any]:
    """
    Compute ensemble RMSE and R² for a given condition and fitted model.

    Uses ensemble mean vs model prediction. Only time points with at least
    one finite trajectory are considered.
    """
    # Only keep times where at least one trajectory is finite
    row_valid = np.isfinite(E_mat).any(axis=1)
    t_plot = t[row_valid]
    E_plot = E_mat[row_valid, :]

    if t_plot.size == 0:
        return {"rmse": np.nan, "r2": np.nan, "n_time": 0, "n_traj": 0}

    E_mean = np.nanmean(E_plot, axis=1)
    E_model = model_total_fret(t_plot, fit)

    mask = np.isfinite(E_mean) & np.isfinite(E_model)
    if not np.any(mask):
        return {"rmse": np.nan, "r2": np.nan, "n_time": 0, "n_traj": E_mat.shape[1]}

    E_obs = E_mean[mask]
    E_mod = E_model[mask]

    residuals = E_obs - E_mod
    rmse = np.sqrt(np.mean(residuals ** 2))
    ss_res = np.sum(residuals ** 2)
    ss_tot = np.sum((E_obs - E_obs.mean()) ** 2)
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 1e-12 else np.nan

    return {
        "rmse": float(rmse),
        "r2": float(r2),
        "n_time": int(E_obs.size),
        "n_traj": int(E_mat.shape[1]),
    }


# ----------------------------------------------------------------------
# Global fitting with curve_fit
# ----------------------------------------------------------------------


def _fit_wrapper(
    t_in: np.ndarray,
    k_oi: float, k_io: float, k_ic: float, k_ci: float, k_b: float,
    e_o: float, e_i: float, e_c: float,
    p_o0: float, p_c0: float,
    f_dyn: float, e_static: float
) -> np.ndarray:
    """
    Helper for curve_fit: maps parameter vector to model prediction.
    """
    params = Hsp90Params3State(
        k_OI=k_oi, k_IO=k_io, k_IC=k_ic, k_CI=k_ci, k_B=k_b,
        E_open=e_o, E_inter=e_i, E_closed=e_c,
        P_O0=p_o0, P_C0=p_c0
    )
    fit = Hsp90Fit3State(params=params, f_dyn=f_dyn, E_static=e_static)
    return model_total_fret(t_in, fit)


def fit_global_3state(
    t: np.ndarray,
    E_mat: np.ndarray,
    p0: Optional[np.ndarray] = None
) -> Hsp90Fit3State:
    """
    Fit the 3-state + bleaching + static model to the ensemble mean.

    Parameters
    ----------
    t : np.ndarray
        Time vector.
    E_mat : np.ndarray
        FRET data matrix (trajectories in columns).
    p0 : Optional[np.ndarray]
        Initial guess for parameters.
        Order:
            [k_OI, k_IO, k_IC, k_CI, k_B,
             E_open, E_inter, E_closed,
             P_O0, P_C0,
             f_dyn, E_static]

    Returns
    -------
    Hsp90Fit3State
        Fitted parameter container.
    """
    # Build ensemble mean and mask valid entries
    row_valid = np.isfinite(E_mat).any(axis=1)
    t_fit = t[row_valid]
    E_mean = np.nanmean(E_mat[row_valid, :], axis=1)

    mask = np.isfinite(E_mean)
    t_fit = t_fit[mask]
    E_fit = E_mean[mask]

    if t_fit.size < 3:
        raise RuntimeError("Not enough valid time points for global fit.")

    # Default initial guess (merged: conservative from pipeline)
    if p0 is None:
        p0 = np.array([
            0.01, 0.01, 0.01, 0.01,  # k_OI, k_IO, k_IC, k_CI
            0.001,                   # k_B
            0.1, 0.3, 0.6,           # E_open, E_inter, E_closed
            0.7, 0.2,                # P_O0, P_C0
            0.7, 0.18                # f_dyn, E_static
        ], dtype=float)

    # Bounds (shared across both versions)
    lower_bounds = np.array([
        0.0, 0.0, 0.0, 0.0,  # k's >= 0
        0.0,                 # k_B >= 0
        0.0, 0.0, 0.0,       # FRET levels >= 0
        0.0, 0.0,            # P_O0, P_C0 >= 0
        0.0, 0.0             # f_dyn, E_static >= 0
    ], dtype=float)

    upper_bounds = np.array([
        10.0, 10.0, 10.0, 10.0,  # max rates
        1.0,                     # max bleaching
        1.0, 1.0, 1.0,           # FRET levels <= 1
        1.0, 1.0,                # P_O0, P_C0 <= 1
        1.0, 1.0                 # f_dyn, E_static <= 1
    ], dtype=float)

    popt, _ = curve_fit(
        _fit_wrapper,
        t_fit,
        E_fit,
        p0=p0,
        bounds=(lower_bounds, upper_bounds),
        maxfev=20000,
    )

    params = Hsp90Params3State(
        k_OI=float(popt[0]),
        k_IO=float(popt[1]),
        k_IC=float(popt[2]),
        k_CI=float(popt[3]),
        k_B=float(popt[4]),
        E_open=float(popt[5]),
        E_inter=float(popt[6]),
        E_closed=float(popt[7]),
        P_O0=float(popt[8]),
        P_C0=float(popt[9]),
    )
    return Hsp90Fit3State(
        params=params,
        f_dyn=float(popt[10]),
        E_static=float(popt[11]),
    )


# ----------------------------------------------------------------------
# Parallel per-condition fits
# ----------------------------------------------------------------------


def _fit_single_condition_worker(
    key: str,
    t: np.ndarray,
    E_mat: np.ndarray,
    col_names: List[str],
    meta: pd.DataFrame,
    group_by: str
) -> Optional[Tuple[str, Hsp90Fit3State, Dict[str, Any]]]:
    """
    Internal worker for parallel fitting of a single condition/group.
    """
    cols_subset = meta.loc[meta[group_by] == key, "col"].tolist()
    if not cols_subset:
        return None

    try:
        t_sub, E_sub = subset_matrix_by_columns(t, E_mat, col_names, cols_subset)
        fit = fit_global_3state(t_sub, E_sub)
        metrics = compute_ensemble_metrics(t_sub, E_sub, fit)
        p = fit.params

        rec = dict(
            group_key=key,
            n_traj=metrics["n_traj"],
            n_time=metrics["n_time"],
            rmse=metrics["rmse"],
            r2=metrics["r2"],
            k_OI=p.k_OI,
            k_IO=p.k_IO,
            k_IC=p.k_IC,
            k_CI=p.k_CI,
            k_B=p.k_B,
            E_open=p.E_open,
            E_inter=p.E_inter,
            E_closed=p.E_closed,
            P_O0=p.P_O0,
            P_C0=p.P_C0,
            f_dyn=fit.f_dyn,
            E_static=fit.E_static,
        )
        return key, fit, rec

    except Exception as e:
        print(f"  Fit failed for group '{key}': {e}")
        return None


def fit_all_conditions(
    t: np.ndarray,
    E_mat: np.ndarray,
    col_names: List[str],
    group_by: str = "condition",
    do_plots: bool = False,
    max_overlay_traces: int = 100,
) -> Tuple[pd.DataFrame, Dict[str, Hsp90Fit3State]]:
    """
    Fit the 3-state+bleaching+static model separately for each group.

    Parameters
    ----------
    t : np.ndarray
        Global time grid.
    E_mat : np.ndarray
        FRET matrix (T x N).
    col_names : List[str]
        Column names for trajectories.
    group_by : str
        Metadata column to group by ("condition", "construct", "exp_id", ...).
    do_plots : bool
        If True, generate time-series plots after fitting.
    max_overlay_traces : int
        Maximum number of individual trajectories to overlay in time plots.

    Returns
    -------
    summary_df : pd.DataFrame
        One row per group with fitted parameters and metrics.
    fit_dict : Dict[str, Hsp90Fit3State]
        Mapping from group key to fitted Hsp90Fit3State.
    """
    meta = parse_column_metadata(col_names)
    keys = sorted(meta[group_by].unique())
    print(f"Fitting {len(keys)} groups in parallel (group_by = '{group_by}')...")

    results = Parallel(n_jobs=-1, verbose=5)(
        delayed(_fit_single_condition_worker)(
            key, t, E_mat, col_names, meta, group_by
        )
        for key in keys
    )

    fit_dict: Dict[str, Hsp90Fit3State] = {}
    records: List[Dict[str, Any]] = []

    for res in results:
        if res is None:
            continue
        key, fit, rec = res
        fit_dict[key] = fit
        records.append(rec)

    summary_df = pd.DataFrame(records) if records else pd.DataFrame()

    if do_plots and fit_dict:
        print("Generating per-group time plots...")
        for key, fit in fit_dict.items():
            cols = meta.loc[meta[group_by] == key, "col"].tolist()
            t_sub, E_sub = subset_matrix_by_columns(t, E_mat, col_names, cols)
            plot_hsp90_fit_time(
                t_sub,
                E_sub,
                fit,
                n_traces_overlay=max_overlay_traces,
                random_seed=0,
            )

    return summary_df, fit_dict


# ----------------------------------------------------------------------
# Bootstrap utilities
# ----------------------------------------------------------------------


def _bootstrap_worker(
    t_sub: np.ndarray,
    E_sub: np.ndarray,
    seed: int
) -> Optional[Dict[str, float]]:
    """
    Internal worker for a single bootstrap resample and fit.
    """
    try:
        rng = np.random.default_rng(seed)

        # Resample trajectories with replacement
        idx_boot = rng.integers(0, E_sub.shape[1], size=E_sub.shape[1])
        E_boot = E_sub[:, idx_boot]

        fit_b = fit_global_3state(t_sub, E_boot)
        p = fit_b.params

        rec: Dict[str, float] = dict(
            k_OI=p.k_OI,
            k_IO=p.k_IO,
            k_IC=p.k_IC,
            k_CI=p.k_CI,
            k_B=p.k_B,
            E_open=p.E_open,
            E_inter=p.E_inter,
            E_closed=p.E_closed,
            P_O0=p.P_O0,
            P_C0=p.P_C0,
            f_dyn=fit_b.f_dyn,
            E_static=fit_b.E_static,
        )
        return rec
    except Exception:
        return None


def bootstrap_condition_params(
    t: np.ndarray,
    E_mat: np.ndarray,
    col_names: List[str],
    meta: pd.DataFrame,
    group_key: str,
    group_by: str = "condition",
    n_boot: int = 100,
    random_seed: int = 0,
) -> pd.DataFrame:
    """
    Bootstrap parameter estimates for a single condition by resampling trajectories.

    Returns
    -------
    pd.DataFrame
        One row per bootstrap replicate, columns:
        k_OI, k_IO, k_IC, k_CI, k_B,
        E_open, E_inter, E_closed,
        P_O0, P_C0,
        f_dyn, E_static
    """
    cols = meta.loc[meta[group_by] == group_key, "col"].tolist()
    if not cols:
        raise ValueError(f"No data for {group_key} in group_by={group_by}.")

    t_sub, E_sub = subset_matrix_by_columns(t, E_mat, col_names, cols)
    print(f"Bootstrapping {group_key} with {n_boot} replicates...")

    seeds = [random_seed + b for b in range(n_boot)]
    results = Parallel(n_jobs=-1, verbose=5)(
        delayed(_bootstrap_worker)(t_sub, E_sub, seed)
        for seed in seeds
    )

    records = [r for r in results if r is not None]
    if not records:
        print(f"Warning: all {n_boot} bootstrap fits failed for {group_key}.")
        return pd.DataFrame()

    return pd.DataFrame(records)


def summarize_bootstrap(boot_df: pd.DataFrame, param: str) -> Tuple[float, float, float]:
    """
    Return mean and 95% CI (2.5, 97.5 percentile) for a parameter from bootstrap DF.
    """
    vals = boot_df[param].dropna().to_numpy()
    if vals.size == 0:
        return np.nan, np.nan, np.nan
    mean = float(np.mean(vals))
    lo, hi = np.percentile(vals, [2.5, 97.5])
    return mean, float(lo), float(hi)


# ----------------------------------------------------------------------
# Plotting
# ----------------------------------------------------------------------


def plot_ensemble_fit(t: np.ndarray, E_mat: np.ndarray, fit: Hsp90Fit3State) -> None:
    """
    Goodness-of-fit plot based on ensemble-averaged FRET (data vs model).
    """
    row_valid = np.isfinite(E_mat).any(axis=1)
    t_plot = t[row_valid]
    E_plot = E_mat[row_valid, :]

    if t_plot.size == 0:
        raise RuntimeError("No valid time points for ensemble fit plot.")

    E_mean = np.nanmean(E_plot, axis=1)
    E_model = model_total_fret(t_plot, fit)

    mask = np.isfinite(E_mean) & np.isfinite(E_model)
    if not np.any(mask):
        raise RuntimeError("No valid (mean, model) pairs for goodness-of-fit plot.")

    E_obs = E_mean[mask]
    E_mod = E_model[mask]

    residuals = E_obs - E_mod
    rmse = np.sqrt(np.mean(residuals ** 2))
    ss_res = np.sum(residuals ** 2)
    ss_tot = np.sum((E_obs - E_obs.mean()) ** 2)
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else np.nan

    fig, ax = plt.subplots(figsize=(5, 5))
    ax.scatter(E_mod, E_obs, alpha=0.5, s=10, label="Data mean")
    lims = [min(E_mod.min(), E_obs.min()), max(E_mod.max(), E_obs.max())]
    ax.plot(lims, lims, "k--", alpha=0.7, label="y = x")
    ax.set_xlabel("Model FRET")
    ax.set_ylabel("Data mean FRET")
    ax.set_title(f"Ensemble fit: RMSE={rmse:.3f}, R²={r2:.3f}")
    ax.legend()
    ax.set_xlim(lims)
    ax.set_ylim(lims)
    fig.tight_layout()
    plt.show()


def plot_hsp90_fit_time(
    t: np.ndarray,
    E_mat: np.ndarray,
    fit: Hsp90Fit3State,
    n_traces_overlay: int = 200,
    random_seed: int = 0,
) -> None:
    """
    Plot Hsp90 3-state + bleaching + static fraction vs data across time.

    - Faded individual traces (up to n_traces_overlay).
    - Mean ± SD band.
    - Model prediction line.
    """
    row_valid = np.isfinite(E_mat).any(axis=1)
    t_plot = t[row_valid]
    E_plot = E_mat[row_valid, :]

    if t_plot.size == 0:
        raise RuntimeError("No valid time points for time-series plot.")

    E_mean = np.nanmean(E_plot, axis=1)
    E_std = np.nanstd(E_plot, axis=1)
    E_model = model_total_fret(t_plot, fit)

    # Choose subset of trajectories for overlay
    rng = np.random.default_rng(random_seed)
    if E_plot.shape[1] > n_traces_overlay:
        idx = rng.choice(E_plot.shape[1], size=n_traces_overlay, replace=False)
        E_subset = E_plot[:, idx]
    else:
        E_subset = E_plot

    fig, ax = plt.subplots(figsize=(8, 4))

    if E_subset is not None:
        for j in range(E_subset.shape[1]):
            ax.plot(t_plot, E_subset[:, j], color="gray", alpha=0.05, lw=0.5)

    ax.plot(t_plot, E_mean, color="tab:blue", lw=2, label="Data mean")
    ax.fill_between(t_plot, E_mean - E_std, E_mean + E_std,
                    color="tab:blue", alpha=0.2, label="Mean ± SD")

    ax.plot(t_plot, E_model, color="tab:red", lw=2, label="Model")

    ax.set_xlabel("Time (s)")
    ax.set_ylabel("FRET")
    ax.set_title("Hsp90 3-state model vs data")
    ax.legend()
    fig.tight_layout()
    plt.show()


def plot_param_vs_condition(summary_df: pd.DataFrame, param: str) -> None:
    """
    Plot parameter value vs condition using summary_df from fit_all_conditions.
    """
    if summary_df.empty:
        print("Summary DF is empty; skipping param plot.")
        return

    fig, ax = plt.subplots(figsize=(6, 3))
    ax.plot(summary_df["group_key"], summary_df[param], marker="o")
    ax.set_xlabel("group_key")
    ax.set_ylabel(param)
    ax.set_title(f"{param} vs group_key")
    plt.xticks(rotation=45, ha="right")
    fig.tight_layout()
    plt.show()


def plot_bootstrap_compare(
    df_a: pd.DataFrame,
    df_b: pd.DataFrame,
    param: str,
    label_a: str,
    label_b: str
) -> None:
    """
    Compare bootstrap distributions for a parameter between two conditions.
    """
    vals_a = df_a[param].dropna().to_numpy()
    vals_b = df_b[param].dropna().to_numpy()

    plt.figure()
    plt.hist(vals_a, bins=30, alpha=0.5, density=True, label=label_a)
    plt.hist(vals_b, bins=30, alpha=0.5, density=True, label=label_b)
    plt.xlabel(param)
    plt.ylabel("Density")
    plt.title(f"Bootstrap Comparison: {param}")
    plt.legend()
    plt.tight_layout()
    plt.show()


# ----------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------


def main() -> None:
    # Adjust path if needed:
    data_path = Path("data/timeseries/fret_matrix.csv")
    if not data_path.exists():
        print(f"Data file not found at {data_path}")
        return

    # 1. Load data
    t, E_mat, col_names = load_combined_matrix(data_path)
    print(f"Loaded data: {E_mat.shape[0]} time points, {E_mat.shape[1]} trajectories")

    # 2. Global fit
    print("\n--- Global Fit (all trajectories pooled) ---")
    global_fit = fit_global_3state(t, E_mat)
    print("Global fit parameters:")
    print(global_fit)

    plot_ensemble_fit(t, E_mat, global_fit)
    plot_hsp90_fit_time(t, E_mat, global_fit)

    # 3. Per-condition fits
    print("\n--- Per-condition fits ---")
    summary_cond, fits_cond = fit_all_conditions(
        t, E_mat, col_names,
        group_by="exp_id",
        do_plots=False,
        max_overlay_traces=150,
    )
    if not summary_cond.empty:
        print(summary_cond)

        # Example: inspect param trends
        for param in ["k_OI", "k_IC", "f_dyn", "E_closed"]:
            if param in summary_cond.columns:
                plot_param_vs_condition(summary_cond, param)

    # 4. Per-construct fits (optional)
    print("\n--- Per-construct fits (optional) ---")
    summary_construct, _ = fit_all_conditions(
        t, E_mat, col_names,
        group_by="construct",
        do_plots=False,
        max_overlay_traces=100,
    )
    if not summary_construct.empty:
        print(summary_construct)

    # 5. Example bootstrap comparison (first two conditions)
    meta = parse_column_metadata(col_names)
    conditions = meta["condition"].unique()
    if len(conditions) >= 2:
        cond_a, cond_b = conditions[0], conditions[1]
        print(f"\n--- Bootstrap comparison: {cond_a} vs {cond_b} ---")

        boot_a = bootstrap_condition_params(t, E_mat, col_names, meta, cond_a, n_boot=50)
        boot_b = bootstrap_condition_params(t, E_mat, col_names, meta, cond_b, n_boot=50)

        for param in ["k_OI", "k_IC", "f_dyn", "E_closed"]:
            if param not in boot_a.columns or param not in boot_b.columns:
                continue
            mA, loA, hiA = summarize_bootstrap(boot_a, param)
            mB, loB, hiB = summarize_bootstrap(boot_b, param)
            print(param)
            print(f"  {cond_a}: mean={mA:.4f}, 95% CI [{loA:.4f}, {hiA:.4f}]")
            print(f"  {cond_b}: mean={mB:.4f}, 95% CI [{loB:.4f}, {hiB:.4f}]")
            print()
            plot_bootstrap_compare(boot_a, boot_b, param, cond_a, cond_b)


if __name__ == "__main__":
    main()
