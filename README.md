# PolyODENet

Inverse chemical kinetics modeling using ODENet.

The initial effort will focus on deriving chemical 
rate equations from concentration time-series data 
based on the law of mass action, i.e. systems of 
first-order ODEs with only polynomial terms on the 
right-hand side.

The following tools/principles are used.

1. Neural ODE by Ricky T.Q. Chen et al. (https://github.com/rtqichen/torchdiffeq)

2. Symbolic regression

3. Sparse regression

4. Knowledge of kinetic differential equations
