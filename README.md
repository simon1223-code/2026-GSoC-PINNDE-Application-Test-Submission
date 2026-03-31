# 2026-GSoC-PINNDE-Application-Test-Submission
This project implements a Physics-Informed Neural Network (PINN) in PyTorch to solve a parameterized underdamped harmonic oscillator. 

## Problem Description

Consider a simplified damped harmonic oscillator with the angular frequency $\omega=1$,

\frac{d^2x}{dz^2}+2\xi \frac{dx}{dz}+x =0

where:
- z ∈ [0, 20] is the domain
- ξ ∈ [0.1, 0.4] is the damping ratio

Boundary conditions:
- x(0) = 0.7
- dx/dz(0) = 1.2

with $z\in[0,20]$, $\xi\in[0.1, 0.4]$, and two initial conditions $x(0)=0.7$, $\frac{dx}{dz}(0)=1.2$. This oscillator is underdamped because $\xi< 1$.

We solve the normalized second-order ODE:

d²x/dz² + 40ξ dx/dz + 400x = 0


