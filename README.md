# **Hsp90 Single-Molecule FRET Analysis**
 
[![License: BSD-3-Clause](https://img.shields.io/badge/License-BSD--3--Clause-blue.svg)](LICENSE)
[![License: CC BY 4.0](https://img.shields.io/badge/License-CC--BY--4.0-lightgrey.svg)](https://creativecommons.org/licenses/by/4.0/)
 
[![Python >=3.10](https://img.shields.io/badge/Python-%3E%3D3.10-blue.svg)](https://www.python.org/)
[![Scientific Computing](https://img.shields.io/badge/Scientific%20Computing-NumPy%20%7C%20SciPy%20%7C%20pandas-informational.svg)](https://scipy.org/)

This repository provides a minimal pipeline to process **single-molecule FRET (.tracks/.h5)** data and fit a **three-state kinetic model** (Open ↔ Intermediate ↔ Closed) with state-dependent bleaching.

The code implements the computational analysis used to interpret the datasets from:

**Anandamurugan et al., Biophysical Journal (2025)**
**Schrangl & Schütz et al., JoVE (2021)**

---

## **Workflow**

### 1. Extract & clean trajectories

```bash
python get_traces.py --data-dir data/Hugel_2025 --export-dir data/timeseries
```

This produces:

* Per-particle CSV traces (intermediate files)
* `fret_matrix.csv` (time × trajectories)

---

### 2. Fit kinetic model & run bootstrap/sensitivity

```bash
python pipeline.py \
  --outdir results \
  --multistarts 5 \
  --bootstraps 50 \
  --cores 8
```

Outputs:

* Best-fit parameters
* Goodness-of-fit plots
* Bootstrap distributions
* Sobol indices

---

## **Model**

The fitted model contains:

* 7 kinetic rates
* 3 ordered FRET levels
* 2 initial state probabilities
* Dynamic fraction + static FRET level

---

### **State Diagram** 

![Model Diagram](images/img.png) 

#### **Model Description**

The full observed ensemble signal is a **mixture**:

$$
E_\text{total}(t) = f_{\text{dyn}} , E_{\text{dyn}}(t) * (1 - f_{\text{dyn}}), E_{\text{static}}
$$

Dynamic component evolves through:

```math
\frac{d}{dt}
\begin{pmatrix}
P_O \\ P_I \\ P_C \\ P_B
\end{pmatrix}
=
\begin{pmatrix}
-(k_{OI}+k_{BO}) & k_{IO} & 0 & 0 \\
k_{OI} & -(k_{IO}+k_{IC}+k_{BI}) & k_{CI} & 0 \\
0 & k_{IC} & -(k_{CI}+k_{BC}) & 0 \\
k_{BO} & k_{BI} & k_{BC} & 0
\end{pmatrix}
\begin{pmatrix}
P_O \\ P_I \\ P_C \\ P_B
\end{pmatrix}
```

where \( P_O, P_I, P_C, P_B \) are the probabilities of being in Open, Intermediate, Closed, and Bleached states, respectively.


## **Directory Structure**

```
get_traces.py     # Extract/clean trajectories
pipeline.py       # Global fitting + bootstrap + sensitivity
data/             # Input .tracks/.h5 files
data/timeseries/  # Per-particle CSV + combined matrix
results/          # Fits, plots, bootstrap, Sobol
```

---

## **Licensing**

* **Code** → BSD-3-Clause
* **Figures & plots** → CC-BY-4.0

See `LICENSE` for details.

---
