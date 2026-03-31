# 2026-GSoC-PINNDE-Application-Test-Submission
This project implements a Physics-Informed Neural Network (PINN) in PyTorch to solve a parameterized underdamped harmonic oscillator. 

## Problem Description

Consider a simplified damped harmonic oscillator with the angular frequency $\omega=1$,

$$\frac{d^2x}{dz^2}+2\xi \frac{dx}{dz}+x =0$$

where:
- $z \in [0, 20]$ is the domain
- $\xi \in [0.1, 0.4]$ is the damping ratio

Boundary conditions:
- $x(0)$ = 0.7
- $\frac{dx}{dz}(0)$ = 1.2

This oscillator is underdamped because $\xi< 1$.

## Normalization of the Domain of $z$

We normalize the range of $z$ into $[0,1]$. Replace $z$ with $\frac{z}{20}$,
$$\frac{d^2x}{dz^2}+2\xi \frac{dx}{dz}+x =0 \implies \frac{d^2x}{dz^2}+40\xi \frac{dx}{dz}+400x =0$$

where:
- $z \in [0, 1]$ is the normalized domain
- $\xi \in [0.1, 0.4]$ is the damping ratio

Boundary conditions:
- $x(0)$ = 0.7
- $\frac{dx}{dz}(0)$ = $1.2 \times 20$ (due to rescaling)

## Method

We use a Physics-Informed Neural Network (PINN) that minimizes:

$$\mathcal{L} = \mathcal{L}_{\text{interior}} + \text{weight} \cdot \mathcal{L}_{\text{boundary}}$$
The weight factor here specificall amplifies the importance of correctly satisfying the boundary conditions (one Dirichlet type and one Neumann type) at $z=0$. The optimizer then performs gradient descent updates on the network parameters to minimize this total loss $\mathcal{L}$.

## Improvement 

With these hyperparameters designated for the training process (e.g. epoch=50000, colloaction=1000), the result of this model displays noticeable errors, especially when $z=0$ (loose boundary condition enforcement) and when $\xi$ is small (more oscillations). These issues are typical in standard PINNs: the neural network must simultaneously learn both the governing differential equation and the boundary/initial conditions, which can lead to multiple optimization objectives and slow convergence near constrained regions. Two specific improvements are proposed to alleviate these errors.

- Hard Constraint Enforcement: Instead of enforcing initial conditions through an additional penalty term in the loss function, we can explicitly embed them into the solution ansatz. With this modification, the network wil always learn a correction term added to a function that already satisfies the initial conditions exactly. The hard constraint enforces the solution as:
$$x(z, \xi)=x_0 + (20 v_0)z + z^2 \mathcal{N}_{\theta}(z, \xi)$$
where $\mathcal{N}_{\theta}$ is the neural network output. ($20v_0$ here because of rescaling)

- Fourier Feature Embedding: A second limitation of standard PINNs is their difficulty in learning high-frequency or oscillatory solutions, particularly when the damping ratio $\xi$ is small and the system exhibits rapid oscillations. To address this, we apply the Fourier feature mappings to the input variables before passing them into the neural network. Specifically, the inputs $(z, \xi)$ are mapped into a higher-dimensional periodic representation:
$$(z, \xi) ↦ (\sin(2\pi B(z,\xi)), \cos(2\pi B(z,\xi)))$$
where $B$ is a fixed random projection matrix. 





