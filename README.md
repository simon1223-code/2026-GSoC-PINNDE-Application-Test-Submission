# 2026-GSoC-PINNDE-Application-Test-Submission
This project implements a Physics-Informed Neural Network (PINN) in PyTorch to solve a parameterized underdamped harmonic oscillator. 

## Problem Description

We solve the normalized second-order ODE:

d²x/dz² + 40ξ dx/dz + 400x = 0

where:
- z ∈ [0, 1] is the normalized domain
- ξ ∈ [0.1, 0.4] is the damping ratio

Boundary conditions:
- x(0) = 0.7
- dx/dz(0) = 20 × 1.2
