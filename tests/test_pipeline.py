from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from scipy.linalg import expm

import pipeline


def _fit_from_fixture(
    fitted_params: dict[str, float], f_dyn: float, e_static: float
) -> pipeline.Hsp90Fit3State:
    params = pipeline.Hsp90Params3State(**fitted_params)
    return pipeline.Hsp90Fit3State(params=params, f_dyn=f_dyn, E_static=e_static)


def test_rate_matrix_properties(rate_matrix_fixture: np.ndarray) -> None:
    assert rate_matrix_fixture.shape == (4, 4)
    assert np.all(np.diag(rate_matrix_fixture) <= 0)
    np.testing.assert_allclose(rate_matrix_fixture.sum(axis=1), 0.0, atol=1e-12)


def test_expm_transition_matrix(rate_matrix_fixture: np.ndarray) -> None:
    transition = expm(rate_matrix_fixture * 0.1)
    assert np.all(transition >= -1e-12)
    np.testing.assert_allclose(transition.sum(axis=1), 1.0, atol=1e-10)


@pytest.mark.parametrize("levels", [(0.0, 0.0, 0.0), (0.2, 0.5, 0.8)])
def test_dynamic_signal_bounds(
    fitted_params: dict[str, float], levels: tuple[float, float, float]
) -> None:
    fitted_params = dict(fitted_params)
    fitted_params["E_open"], fitted_params["E_inter"], fitted_params["E_closed"] = (
        levels
    )
    params = pipeline.Hsp90Params3State(**fitted_params)
    t = np.linspace(0, 10, 50)
    signal = pipeline.model_fret_3state(t, params)
    assert np.all(np.isfinite(signal))
    assert np.nanmin(signal) >= 0.0
    assert np.nanmax(signal) <= 1.0


def test_total_signal_mixture_limits(fitted_params: dict[str, float]) -> None:
    t = np.linspace(0, 10, 50)
    fit_dyn = _fit_from_fixture(fitted_params, f_dyn=1.0, e_static=0.1)
    fit_static = _fit_from_fixture(fitted_params, f_dyn=0.0, e_static=0.3)

    e_dyn = pipeline.model_fret_3state(t, fit_dyn.params)
    np.testing.assert_allclose(pipeline.model_total_fret(t, fit_dyn), e_dyn)
    np.testing.assert_allclose(pipeline.model_total_fret(t, fit_static), 0.3)


def test_fit_global_penalizes_unphysical_params(fret_matrix: pd.DataFrame) -> None:
    t = np.linspace(0, 5, len(fret_matrix))
    e_mat = fret_matrix.to_numpy()
    fit = pipeline.fit_global_3state(t, e_mat, n_starts=1, n_jobs=1)
    assert isinstance(fit, pipeline.Hsp90Fit3State)


def test_multistart_returns_best_fit(fret_matrix: pd.DataFrame) -> None:
    t = np.linspace(0, 5, len(fret_matrix))
    fit = pipeline.fit_global_3state(t, fret_matrix.to_numpy(), n_starts=2, n_jobs=1)
    assert isinstance(fit, pipeline.Hsp90Fit3State)


def test_bootstrap_shape(fret_matrix: pd.DataFrame) -> None:
    t = np.linspace(0, 5, len(fret_matrix))
    col_names = [f"construct_241107_p{i:05d}" for i in range(3)]
    meta = pipeline.parse_column_metadata(col_names)
    boot = pipeline.bootstrap_condition_params(
        t=t,
        E_mat=fret_matrix.to_numpy(),
        col_names=col_names,
        meta=meta,
        group_key=meta["condition"].iloc[0],
        group_by="condition",
        n_boot=3,
        random_seed=1,
        n_jobs=1,
    )
    assert len(boot) <= 3


def test_sobol_indices_sum_leq_one(fret_matrix: pd.DataFrame) -> None:
    t = np.linspace(0, 5, len(fret_matrix))
    bounds = {"k_OI": (0.01, 1.0), "k_IO": (0.01, 1.0), "f_dyn": (0.1, 0.9)}
    si = pipeline.sobol_sensitivity_3state(
        t=t,
        E_mat=fret_matrix.to_numpy(),
        param_bounds=bounds,
        n_base_samples=8,
        n_jobs=1,
    )
    assert np.nansum(si["S1"]) <= 1.0 + 0.2


def test_output_file_creation(
    tmp_path: Path, fret_matrix: pd.DataFrame, fitted_params: dict[str, float]
) -> None:
    t = np.linspace(0, 5, len(fret_matrix))
    outdir = tmp_path
    fit = _fit_from_fixture(fitted_params, f_dyn=0.7, e_static=0.2)
    pipeline.plot_ensemble_fit(t, fret_matrix.to_numpy(), fit, outdir)
    pipeline.plot_residuals_over_time(t, fret_matrix.to_numpy(), fit, outdir)

    (outdir / "best_params.csv").write_text("k_OI\n0.1\n")
    (outdir / "bootstrap_distributions.csv").write_text("k_OI\n0.1\n")
    (outdir / "sobol_indices.csv").write_text("param,S1\nk_OI,0.2\n")

    assert (outdir / "best_params.csv").exists()
    assert (outdir / "bootstrap_distributions.csv").exists()
    assert (outdir / "sobol_indices.csv").exists()
    assert any(outdir.glob("*.png"))


def test_pipeline_cli_args(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        "sys.argv", ["pipeline.py", "--multistarts", "2", "--bootstraps", "3"]
    )
    parser = pipeline.argparse.ArgumentParser()
    parser.add_argument("--multistarts", type=int)
    parser.add_argument("--bootstraps", type=int)
    args = parser.parse_args(["--multistarts", "2", "--bootstraps", "3"])
    assert args.multistarts == 2
    assert args.bootstraps == 3


@pytest.mark.parametrize("identical_rates", [True, False])
def test_edge_case_identical_rates(
    fitted_params: dict[str, float], identical_rates: bool
) -> None:
    params = dict(fitted_params)
    if identical_rates:
        params.update({"k_OI": 0.2, "k_IO": 0.2, "k_IC": 0.2, "k_CI": 0.2})
    fit = _fit_from_fixture(params, f_dyn=0.8, e_static=0.3)
    t = np.array([0.0, 0.1, 0.2])
    signal = pipeline.model_total_fret(t, fit)
    assert signal.shape == t.shape


def test_pipeline_main_entrypoint(tmp_path: Path, fret_matrix: pd.DataFrame, monkeypatch: pytest.MonkeyPatch) -> None:
    data_dir = tmp_path / "data" / "timeseries"
    data_dir.mkdir(parents=True)

    matrix = fret_matrix.copy()
    matrix.columns = [
        f"constructA_241107_p{i:05d}" for i in range(1, matrix.shape[1] + 1)
    ]
    matrix.to_csv(data_dir / "fret_matrix.csv", index=False)

    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(pipeline, "outdir", tmp_path)
    monkeypatch.setattr(pipeline.args, "multistarts", 1)
    monkeypatch.setattr(pipeline.args, "bootstraps", 2)
    monkeypatch.setattr(pipeline.args, "cores", 1)

    pipeline.main()

    assert (tmp_path / "best_params.csv").exists()
