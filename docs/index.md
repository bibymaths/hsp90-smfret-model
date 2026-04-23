# smFRET Kinetic Model

Three-state OIC kinetic model for single-molecule FRET data with bleaching-aware ensemble fitting.

[![Code License: BSD-3-Clause](https://img.shields.io/badge/License-BSD--3--Clause-blue.svg)](../LICENSE)
[![Figures & Plots License: CC BY 4.0](https://img.shields.io/badge/License-CC--BY--4.0-lightgrey.svg)](https://creativecommons.org/licenses/by/4.0/)
[![Python >=3.11](https://img.shields.io/badge/Python-%3E%3D3.11-blue.svg)](https://www.python.org/)
[![Lint](https://github.com/abhinavmishra/hsp90-smfret-model/actions/workflows/lint.yml/badge.svg)](https://github.com/abhinavmishra/hsp90-smfret-model/actions/workflows/lint.yml)
[![Tests](https://github.com/abhinavmishra/hsp90-smfret-model/actions/workflows/tests.yml/badge.svg)](https://github.com/abhinavmishra/hsp90-smfret-model/actions/workflows/tests.yml)
[![Docs](https://github.com/abhinavmishra/hsp90-smfret-model/actions/workflows/docs.yml/badge.svg)](https://github.com/abhinavmishra/hsp90-smfret-model/actions/workflows/docs.yml)
[![codecov](https://codecov.io/gh/abhinavmishra/hsp90-smfret-model/branch/main/graph/badge.svg)](https://codecov.io/gh/abhinavmishra/hsp90-smfret-model)

<div style="display:flex;gap:24px;align-items:center;">
  <img src="images/logo.png" alt="Project logo" width="42%" />
  <img src="images/flow.png" alt="OIC state model" width="42%" />
</div>

## Quick start

```bash
pip install -e ".[dev,docs]"
python get_traces.py --data-dir data/Hugel_2025 --export-dir data/timeseries
python pipeline.py --outdir results --multistarts 5 --bootstraps 10 --cores 4
```
