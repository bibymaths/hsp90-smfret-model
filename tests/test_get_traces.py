from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

import get_traces


def test_h5_read_and_particle_export(tmp_data_dir: Path, tmp_path: Path) -> None:
    export_dir = tmp_path / "timeseries"
    get_traces.export_per_particle_time_series(
        data_dir=tmp_data_dir,
        export_dir=export_dir,
        frame_interval=0.07,
        fret_min=0.0,
        fret_max=1.0,
        min_traj_length=10,
    )
    csvs = list(export_dir.glob("*.csv"))
    assert csvs
    sample = pd.read_csv(csvs[0])
    assert {"time_s", "FRET"}.issubset(sample.columns)


def test_combined_matrix_shape(tmp_data_dir: Path, tmp_path: Path) -> None:
    export_dir = tmp_path / "timeseries"
    get_traces.export_per_particle_time_series(
        data_dir=tmp_data_dir,
        export_dir=export_dir,
        frame_interval=0.07,
        fret_min=0.0,
        fret_max=1.0,
    )
    combined = get_traces.build_combined_fret_matrix(
        export_dir=export_dir,
        frame_interval=0.07,
        fret_min=0.0,
        fret_max=1.0,
        use_interpolation=False,
    )
    assert combined is not None
    matrix = pd.read_csv(combined)
    assert matrix.shape[0] > 0
    assert matrix.shape[1] >= 2


def test_malformed_h5_logs_and_skips(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    data_dir = tmp_path / "bad"
    data_dir.mkdir()
    (data_dir / "bad-241107-Hsp90_409_601-v014.tracks.h5").write_text("not hdf")
    export_dir = tmp_path / "out"

    with caplog.at_level("ERROR"):
        get_traces.export_per_particle_time_series(
            data_dir=data_dir,
            export_dir=export_dir,
            frame_interval=0.07,
            fret_min=0.0,
            fret_max=1.0,
        )
    assert "Could not read" in caplog.text


def test_interpolate_trace_cleans_nans_and_bounds() -> None:
    time_grid = np.arange(0, 1.0, 0.1)
    t_trace = np.array([0.0, 0.1, 0.2, 0.3])
    e_trace = np.array([0.2, np.nan, 1.4, 0.6])
    out = get_traces.interpolate_trace(time_grid, t_trace, e_trace, interpolate=False)
    assert out[0] == pytest.approx(0.2)
    assert np.isnan(out[1])
    assert out[2] == pytest.approx(1.4)
    assert out[3] == pytest.approx(0.6)


def test_interpolate_trace_empty_and_single_point() -> None:
    time_grid = np.array([0.0, 0.1, 0.2])
    out_empty = get_traces.interpolate_trace(
        time_grid=time_grid,
        t_trace=np.array([]),
        E_trace=np.array([]),
        interpolate=True,
    )
    assert np.isnan(out_empty).all()

    out_single = get_traces.interpolate_trace(
        time_grid=time_grid,
        t_trace=np.array([0.1]),
        E_trace=np.array([0.7]),
        interpolate=True,
    )
    assert np.isnan(out_single[0])
    assert out_single[1] == pytest.approx(0.7)
    assert np.isnan(out_single[2])


def test_interpolate_trace_linear_and_out_of_bounds_nan() -> None:
    time_grid = np.array([0.0, 0.1, 0.2, 0.3])
    out = get_traces.interpolate_trace(
        time_grid=time_grid,
        t_trace=np.array([0.1, 0.2]),
        E_trace=np.array([0.25, 0.75]),
        interpolate=True,
    )
    assert np.isnan(out[0])
    assert out[1] == pytest.approx(0.25)
    assert out[2] == pytest.approx(0.75)
    assert np.isnan(out[3])


def test_parse_args(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        "sys.argv",
        [
            "get_traces.py",
            "--data-dir",
            "data/Hugel_2025",
            "--export-dir",
            "data/timeseries",
        ],
    )
    args = get_traces.parse_args()
    assert args.data_dir.name == "Hugel_2025"
    assert args.export_dir.name == "timeseries"


def test_main_entrypoint(
    tmp_data_dir: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(
        "sys.argv",
        [
            "get_traces.py",
            "--data-dir",
            str(tmp_data_dir),
            "--export-dir",
            str(tmp_path),
            "--save-plots",
            "--keep-intermediate",
        ],
    )
    get_traces.main()
    assert (tmp_path / "fret_matrix.csv").exists()
    assert any((tmp_data_dir / "plots").glob("*.png"))


def test_cleanup_intermediate_csv(tmp_path: Path) -> None:
    export_dir = tmp_path / "timeseries"
    export_dir.mkdir()
    (export_dir / "trace_1.csv").write_text("time_s,FRET\n0.0,0.2\n")
    (export_dir / "trace_2.csv").write_text("time_s,FRET\n0.1,0.3\n")
    combined = export_dir / "fret_matrix.csv"
    combined.write_text("time_s,traj\n0.0,0.2\n")

    get_traces.cleanup_intermediate_csv(export_dir, combined_name=combined.name)

    assert combined.exists()
    assert not (export_dir / "trace_1.csv").exists()
    assert not (export_dir / "trace_2.csv").exists()
