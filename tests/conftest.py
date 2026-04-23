from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest


@pytest.fixture
def tmp_data_dir(tmp_path: Path) -> Path:
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    rng = np.random.default_rng(42)

    for idx in range(3):
        n = 50
        df = pd.DataFrame(
            {
                "donor_frame": np.arange(n),
                "fret_particle": np.full(n, idx + 1),
                "fret_eff": np.clip(rng.normal(0.45 + idx * 0.1, 0.1, n), 0.01, 0.99),
                "fret_exc_type": ["d"] * n,
                "filter_manual": np.ones(n),
            }
        )
        out = data_dir / f"filtered-24110{idx}-Hsp90_409_601-v014.tracks.h5"
        df.to_hdf(out, key="/tracks/Data", mode="w")

    return data_dir


@pytest.fixture
def fret_matrix() -> pd.DataFrame:
    rng = np.random.default_rng(7)
    data = np.clip(rng.normal(0.5, 0.15, (50, 3)), 0.0, 1.0)
    return pd.DataFrame(data, columns=["traj_1", "traj_2", "traj_3"])


@pytest.fixture
def fitted_params() -> dict[str, float]:
    return {
        "k_OI": 0.2,
        "k_IO": 0.15,
        "k_IC": 0.3,
        "k_CI": 0.25,
        "k_BO": 0.02,
        "k_BI": 0.03,
        "k_BC": 0.04,
        "E_open": 0.2,
        "E_inter": 0.5,
        "E_closed": 0.8,
        "P_O0": 0.5,
        "P_C0": 0.2,
    }


@pytest.fixture
def rate_matrix_fixture() -> np.ndarray:
    """Valid 4x4 generator matrix: O↔I↔C, with bleaching from O/I/C."""
    q_mat = np.array(
        [
            [0.0, 0.20, 0.00, 0.02],
            [0.15, 0.00, 0.30, 0.03],
            [0.00, 0.25, 0.00, 0.04],
            [0.00, 0.00, 0.00, 0.00],
        ],
        dtype=float,
    )
    np.fill_diagonal(q_mat, -q_mat.sum(axis=1))
    return q_mat
