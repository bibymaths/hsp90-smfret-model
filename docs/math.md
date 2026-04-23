# Mathematical model

## Four-state master equation (OIC + bleached)

The kinetic model augments the conformational O-I-C manifold with an absorbing bleached state **B**:

$$
\frac{d\mathbf{P}(t)}{dt} = Q\,\mathbf{P}(t),
\quad
\mathbf{P}(t)=\begin{bmatrix}P_O(t)\\P_I(t)\\P_C(t)\\P_B(t)\end{bmatrix}.
$$

Rate matrix:

$$
Q=
\begin{bmatrix}
-(k_{OI}+k_{BO}) & k_{IO} & 0 & 0 \\
k_{OI} & -(k_{IO}+k_{IC}+k_{BI}) & k_{CI} & 0 \\
0 & k_{IC} & -(k_{CI}+k_{BC}) & 0 \\
k_{BO} & k_{BI} & k_{BC} & 0
\end{bmatrix}.
$$

The seven kinetic rates are:

- `k_OI`: Open → Intermediate
- `k_IO`: Intermediate → Open
- `k_IC`: Intermediate → Closed
- `k_CI`: Closed → Intermediate
- `k_BO`: Open → Bleached
- `k_BI`: Intermediate → Bleached
- `k_BC`: Closed → Bleached

## Emission model in FRET space

Conformational FRET levels are:

- `E_O`, `E_I`, `E_C` for open/intermediate/closed states.

Initial conditions are parameterized by two probabilities:

- `π_O` and `π_I` (with `π_C = 1 - π_O - π_I`, `π_B = 0`).

Dynamic signal:

$$
\mathbf{P}(t) = e^{Qt}\,\pi,
\quad
E_{\text{dyn}}(t) = \begin{bmatrix}E_O & E_I & E_C\end{bmatrix}
\begin{bmatrix}P_O(t)\\P_I(t)\\P_C(t)\end{bmatrix}.
$$

Total ensemble signal:

$$
E_{\text{total}}(t)=f_{\text{dyn}}\,E_{\text{dyn}}(t)+(1-f_{\text{dyn}})\,E_{\text{static}}.
$$

## Likelihood / goodness-of-fit

Given observed mean trajectory `\hat E(t_i)` with optional uncertainty weights `w_i`, fitting minimizes a weighted residual objective,

$$
\mathcal{L}(\theta) \propto \sum_i w_i\left(\hat E(t_i)-E_{\text{total}}(t_i;\theta)\right)^2,
$$

and reports diagnostic metrics (RMSE, residual structure, and bootstrap intervals) for parameter reliability.

!!! warning "Numerical stability"
    Matrix exponentials can be sensitive when rates are near-degenerate or eigenvalues are near zero. Constrain rates to physically plausible bounds and validate solutions with residual diagnostics.

!!! tip "Bootstrap interpretation"
    Interpret bootstrap intervals on the 7 rates as empirical uncertainty under trajectory resampling. Narrow intervals indicate robust identifiability; broad or skewed intervals suggest parameter coupling or weakly informative data.
