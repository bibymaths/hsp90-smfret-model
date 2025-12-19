# Hsp90 Single-Molecule FRET Kinetic Modeling and Analysis Pipeline

[![License: BSD-3-Clause](https://img.shields.io/badge/License-BSD--3--Clause-blue.svg)](LICENSE)
[![License: CC BY 4.0](https://img.shields.io/badge/License-CC--BY--4.0-lightgrey.svg)](https://creativecommons.org/licenses/by/4.0/)
[![Python >=3.10](https://img.shields.io/badge/Python-%3E%3D3.10-blue.svg)](https://www.python.org/)
[![Scientific Computing](https://img.shields.io/badge/Scientific%20Computing-NumPy%20%7C%20SciPy%20%7C%20pandas-informational.svg)](https://scipy.org/)

This repository provides a minimal pipeline to process **single-molecule FRET (.tracks/.h5)** data<sup>[1]</sup> and fit a  
**three-state kinetic model** (Open ↔ Intermediate ↔ Closed) with state-dependent bleaching.

--- 

## **Data** 

The data is too big for LFS storage. Download the dataset from Zenodo<sup>[1]</sup> and place uzipped files in `data/Hugel_2025/`. 

```bash 
aria2c \
  -x 16 \
  -s 16 \
  -k 1M \
  -c \
  -o FRET_TTB_Raw_Data.zip \
  "https://zenodo.org/records/17559063/files/FRET_TTB_Raw_Data.zip?download=1"
```
 
or 

```bash
wget \
  -c \
  --tries=0 \
  --read-timeout=20 \
  --timeout=20 \
  --continue \
  -O FRET_TTB_Raw_Data.zip \
  "https://zenodo.org/records/17559063/files/FRET_TTB_Raw_Data.zip?download=1"
```

or 

```bash
curl \
  -L \
  -C - \
  --retry 10 \
  --retry-delay 5 \
  --retry-all-errors \
  -o FRET_TTB_Raw_Data.zip \
  "https://zenodo.org/records/17559063/files/FRET_TTB_Raw_Data.zip?download=1"
```

Use this command to get the dataset from local to remote server:

```bash
rsync -a --info=progress2 --partial --inplace \
  --compress-choice=zstd --compress-level=3 \
  data/ user@host:/home/userpath/hsp90-smfret-model/data/
```


---
## **Workflow**
 
Download the Zenodo dataset<sup>[1]</sup> and place all .h5 tracking files into `data/Hugel_2025/` before running the pipeline. 

### 1. Extract & clean trajectories

```bash
python get_traces.py \
    --alpha 0.17 \
    --delta 0.12 \
    --gamma 0.8 \
    --frame-interval 0.18 \
    --save-plots
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

where \( P_O, P_I, P_C, P_B \) are the probabilities of being in Open, Intermediate, Closed, and Bleached states,
respectively.

## **Directory Structure**

```
get_traces.py     # Extract/clean trajectories
pipeline.py       # Global fitting + bootstrap + sensitivity
data/             # Input .tracks/.h5 files
data/timeseries/  # Per-particle CSV + combined matrix
results/          # Fits, plots, bootstrap, Sobol
```

---

## **References**

<a id="1"></a>
[1] Hugel, T. (2025). Datasets for “Single-molecule FRET and tracking of transfected biomolecules in living
cells” [Data set]. Biophysical Journal, 125(March 3), 1–9. Zenodo. https://doi.org/10.5281/zenodo.17559063

[2] **Schrangl, L.**, Göhring, J., Kellner, F., Huppa, J. B., & Schütz, G. J. (2021). *Automated Two-dimensional
Spatiotemporal Analysis of Mobile Single-molecule FRET Probes.* **Journal of Visualized Experiments**, 177,
e63124. [https://doi.org/10.3791/63124](https://doi.org/10.3791/63124)

[3] **Anandamurugan, A.**, Eidloth, A., Frank, V., Wortmann, P., Schrangl, L., Lan, C., Schütz, G. J., & Hugel, T. (
2025). *Single-molecule FRET and tracking of transfected biomolecules in living cells.* **Biophysical Journal**,
S0006-3495(25)00604-6. Advance online
publication. [https://doi.org/10.1016/j.bpj.2025.09.024](https://doi.org/10.1016/j.bpj.2025.09.024)

---

## **Licensing**

* **Code** → BSD-3-Clause
* **Figures & plots** → CC-BY-4.0

See `LICENSE` for details.

---


