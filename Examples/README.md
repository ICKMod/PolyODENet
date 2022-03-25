# Example 1: 2 consecutive reactions with 1 intermediate

This example has consecutive reactions with species C being the intermediate.

A <-> C -> B

The dimensionless reaction constants are:
* k_1+ = 2 (first reaction forward)
* k_1- = 1 (first reaction reverse)
* k_2  = 3 (second reaction)

The corresponding ODEs are:
* dA/dt = -2A + 1C
* dB/dt = 3C
* dC/dt = 2A - 4C

The original data are generated using scipy.integrate.odeint.
The initial condition is A0=2, B0=0, C0=0.

In the 'ex1.txt' file, concentration data for C are replaced with -1,
meaning missing profile. Only the initial condition C0=0 is kept.

The 'ex1.guess' file gives almost-optimized ODE coefficients so the 
test can be fast.

The 'ex1.ind' and 'ex1.scale' files control the polynomial model.

