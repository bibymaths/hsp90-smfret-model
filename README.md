## License

[![License: BSD-3-Clause](https://img.shields.io/badge/License-BSD--3--Clause-blue.svg)](LICENSE)
[![License: CC BY 4.0](https://img.shields.io/badge/License-CC--BY--4.0-lightgrey.svg)](https://creativecommons.org/licenses/by/4.0/)

This repository is dual-licensed:

- **Software code** → BSD-3-Clause  
- **Figures, plots, visualizations** → CC-BY 4.0  

See the [LICENSE](./LICENSE) file for complete details.

# FRET simulation and fitting toolbox for single-molecule biophysics

### Understanding FRET datasets from Hugel et al., Biophysical Journal, 2025:

## Dataset Summary 

This directory contains **processed single-molecule FRET tracking datasets** from  
**Anandamurugan A. et al., “Single-molecule FRET and tracking of transfected biomolecules in living cells,” _Biophysical Journal_, 2025.**

Each `.tracks.h5` file is the filtered output of the Schütz Group `fret-analysis` pipeline (`/tracks/Data` table).  
All datasets have a uniform schema (33 columns) describing per-frame donor/acceptor intensities, positions, background, and corrected FRET efficiencies.

---

## 🧩 Experimental Overview

| Category | Entity | Biological Context | Purpose |
|-----------|---------|--------------------|----------|
| **Calibration** | End-stretched DNA (esDNA) with donor/acceptor dyes | *In vitro* fixed sample and live-cell transfection controls | Establish FRET correction factors, measure photobleaching and tracking precision |
| **Protein Dynamics** | Yeast **Hsp90 (residues 409–601)** labelled at both termini | Live-cell single-molecule FRET reporter | Quantify conformational transitions between open (low FRET) and closed (high FRET) states |

---

## ⚙️ Data Generation Pipeline (per Supplementary Info ([Link](https://www.jove.com/files/ftp_upload/63124/si.pdf))

1. **01. Tracking.ipynb** – Localize emitters, register donor/acceptor channels, link detections into trajectories.  
2. **02. Analysis.ipynb** – Flat-field and beam-profile correction, donor/acceptor crosstalk correction, bleaching-state detection, trajectory filtering by stoichiometry.  
3. **03. Plot.ipynb** – Generate efficiency–stoichiometry (E–S) plots and export filtered trajectories as `.tracks.h5`.

Each trajectory (`fret_particle`) corresponds to one fluorescently labelled molecule followed over time.  
Frame interval: **0.05 s** (typical duration 10 – 60 s per trajectory).

---

## 📊 Exploratory Data Summary

| File | Entity / System | Frames × Cols | Mean Track Len (frames) | FRET_eff Range | Typical Behaviour | Interpretation |
|------|-----------------|---------------|--------------------------|----------------|-------------------|----------------|
| `filtered-221020-fixed_esDNA-v014.tracks.h5` | Fixed esDNA (in vitro) | 123 445 × 33 | ≈ 450 | 0 – 0.25 | Flat low FRET ≈ 0.15 | Static calibration ruler |
| `filtered-241203-esDNA-R1-v014.tracks.h5` | esDNA R1 (live-cell) | 407 091 × 33 | ≈ 270 | −0.2 – 0.3 | Gradual decay | Photobleaching & intracellular diffusion |
| `filtered-241204-esDNA-R1-v014.tracks.h5` | esDNA R1 (replicate) | 562 684 × 33 | ≈ 240 | −0.4 – 0.3 | Flat, low FRET | Background-limited control |
| `filtered-241107-Hsp90_409_601-v014.tracks.h5` | Hsp90 409-601 rep 1 | 398 124 × 33 | ≈ 225 | 0 – 1.0 | Stable high FRET (~0.7–0.8) | Closed conformation |
| `filtered-241108-Hsp90_409_601-v014.tracks.h5` | Hsp90 409-601 rep 2 | 463 971 × 33 | ≈ 220 | 0 – 0.5 | Decay (~0.4 → 0) | Opening transition / bleaching |
| `filtered-250430-Hsp90_409_601-v014.tracks.h5` | Hsp90 409-601 rep 3 | 159 966 × 33 | ≈ 300 | −0.4 – 0.4 | Weak/flat (~0.1) | Mostly open population |
| `filtered-250508-Hsp90_409_601-v014.tracks.h5` | Hsp90 409-601 rep 4 | 522 401 × 33 | ≈ 310 | −0.2 – 0.2 | Constant low FRET | Open/bleached trajectories |

---

## 🧠 Interpretation

- **esDNA datasets** act as *non-dynamic references* to calibrate optical alignment, correction coefficients, and photobleaching behaviour.  
- **Hsp90 datasets** report real conformational dynamics of the molecular chaperone:
  - High FRET (≈ 0.7 – 0.8) → closed compact state  
  - Low FRET (≈ 0.1 – 0.3) → open state or post-bleach residual  
- Per-trajectory time series are accessible under `/tracks/Data` → columns `fret_eff`, `fret_stoi`, `donor_frame`, `fret_particle`.  

---

## 🧾 Notes

I can group by `fret_particle` to export per-molecule `time_s × fret_eff` traces for kinetic modeling.

---

## 📚 References

1. Lambert, TJ (2019) FPbase: a community-editable fluorescent protein database. Nature Methods. 16, 277–278. doi: 10.1038/s41592-019-0352-8. [GitHub](https://github.com/tlambert03/FPbase) 
2. **Hugel, T.** (2025). Datasets for "Single-molecule FRET and tracking of transfected biomolecules in living cells". In Biophysical Journal (Vol. 125, Number March 3, pp. 1–9). [Zenodo](https://doi.org/10.5281/zenodo.17559063) 
3. Anandamurugan, A., Eidloth, A., Frank, V., Wortmann, P., Schrangl, L., Lan, C., Schütz, G. J., & Hugel, T. (2025). Single-molecule FRET and tracking of transfected biomolecules in living cells. Biophysical Journal, 124(9). [DOI](https://doi.org/10.1016/j.bpj.2025.09.024)