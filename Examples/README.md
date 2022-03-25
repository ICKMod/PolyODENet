## Example 1
###Consecutive reactions with 1 unknown intermediate

This example has consecutive reactions with species C being the intermediate.

A <--> C --> B

The dimensionless reaction constants are:
* k_1+ = 2 (first reaction forward)
* k_1- = 1 (first reaction reverse)
* k_2  = 3 (second reaction)

The corresponding ODEs are:

d[A]/dt = -2[A] + 1[C] 

d[B]/dt = 3[C] 

d[C]/dt = 2[A] - 4[C]

The original data are generated using scipy.integrate.odeint.
The initial condition is A0=2, B0=0, C0=0.

In the 'ex1.txt' file, concentration data for C are replaced with -1,
meaning missing profile. Only the initial condition C0=0 is kept.

The 'ex1.guess' file gives almost-optimized ODE coefficients so the 
test can be fast.

The 'ex1.ind' and 'ex1.scale' files control the polynomial model.

Use the following command for a test run. 

```train_poly -f ex1.txt -m 1000 -N ex1 -igs```

Your 'Rate_equations_of_ex1.txt' should look like

```
PolynomialODE():

d[A]/dt = -1.84[A] + 0.682[C]

d[B]/dt = 2.73[C]

d[C]/dt = 1.84[A] - 3.32[C]
```