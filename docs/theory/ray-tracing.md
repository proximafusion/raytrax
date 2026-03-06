---
icon: lucide/spline
---

# Ray Tracing

## Ray Tracing Equations

As introduced in [Theory](../theory.md), the local wave vector $\boldsymbol{k}$ satisfies the dispersion relation $\det \boldsymbol{\mathsf{D}} = 0$.

For ray tracing, we define a Hamiltonian $\mathcal H(\boldsymbol{r}, \boldsymbol{k})$ as the real part of the dispersion relation determinant (or a scalar function proportional to it), assuming that the anti-Hermitian part of the dielectric tensor (representing absorption) is small (weak damping approximation):

$$ \mathcal H(\boldsymbol{r}, \boldsymbol{k}) = \text{Re}(\det \boldsymbol{\mathsf{D}}) = 0 $$

The ray equations can then be derived from Hamilton's equations in terms of a time-like parameter $\tau$ along the ray:

$$\frac{d\boldsymbol{r}}{d\tau} = \frac{\partial \mathcal H}{\partial \boldsymbol{k}}, \quad \frac{d\boldsymbol{k}}{d\tau} = -\frac{\partial \mathcal H}{\partial \boldsymbol{r}}$$

Using the arc length $s$ along the ray, which satisfies

$$\frac{ds}{d\tau} = \left|\frac{d\boldsymbol{r}}{d\tau}\right|$$

we can write

$$\frac{d\boldsymbol{r}}{ds} = \left|\frac{\partial \mathcal H}{\partial \boldsymbol{k}}\right|^{-1} \frac{\partial \mathcal H}{\partial \boldsymbol{k}}, \quad \frac{d\boldsymbol{k}}{ds} = -\left|\frac{\partial \mathcal H}{\partial \boldsymbol{k}}\right|^{-1} \frac{\partial \mathcal H}{\partial \boldsymbol{r}}$$

or, equivalently,

$$\frac{d\boldsymbol{r}}{ds} = \left|\frac{\partial \mathcal H}{\partial \boldsymbol{n}}\right|^{-1} \frac{\partial \mathcal H}{\partial \boldsymbol{n}}, \quad \frac{d\boldsymbol{n}}{ds} = -\left|\frac{\partial \mathcal H}{\partial \boldsymbol{n}}\right|^{-1} \frac{\partial \mathcal H}{\partial \boldsymbol{r}}$$

where $\boldsymbol{n}=\boldsymbol{k}c/\omega$.

The gradients $\partial \mathcal H/\partial \boldsymbol{r}$ and $\partial \mathcal H/\partial \boldsymbol{n}$ are computed in Raytrax using automatic differentiation in the function `hamiltonian_gradients`, starting from a scalar Hamiltonian function.

## Cold Tracing

The easiest situation is that of a "cold" plasma, where relativistic effects can be neglected. We start with the fluid momentum equation, which represents the first moment of the Boltzmann equation for electrons (all other species are treated as static):

$$\frac{\partial \boldsymbol{v}_e}{\partial t} + (\boldsymbol{v}_e \cdot \nabla) \boldsymbol{v}_e = -\frac{e}{m_e} \left[ \boldsymbol{E} + \boldsymbol{v}_e \times \boldsymbol{B} \right] - \frac{\nabla P_e}{m_e n_e} + \boldsymbol{C}_e$$

Considering perturbations of these quantities in a background magnetic field $\boldsymbol{B}_0$ assuming plane wave solutions $\sim \exp(i(\boldsymbol{k}\cdot\boldsymbol{r} - \omega t))$,

- $\boldsymbol{v}_e = \boldsymbol{v}_{0e} + \tilde{\boldsymbol{v}}_e$
- $\boldsymbol{B} = \boldsymbol{B}_0 + \tilde{\boldsymbol{B}}$
- $\boldsymbol{E} = \mathbf{0} + \tilde{\boldsymbol{E}}$

neglecting background flow ($\boldsymbol{v}_{0e} = \mathbf{0}$), dropping second-order terms, and neglecting $\nabla P_e$ and $\boldsymbol{C}_e$ (cold, collisionless plasma), one arrives at

$$-i\omega m_e \tilde{\boldsymbol{v}}_e = -e (\tilde{\boldsymbol{E}} + \tilde{\boldsymbol{v}}_e \times \boldsymbol{B}_0)$$

Solving this for the electron velocity perturbation $\tilde{\boldsymbol{v}}_e$ allows us to calculate the induced current density $\boldsymbol{j} = -e n_{e0} \tilde{\boldsymbol{v}}_e$, where $e > 0$ is the elementary charge. By relating $\boldsymbol{j}$ to the electric field through $\boldsymbol{j} = \boldsymbol{\sigma} \cdot \tilde{\boldsymbol{E}}$ and using the relation $\boldsymbol{\varepsilon} = \boldsymbol{I} + \frac{i\boldsymbol{\sigma}}{\omega \epsilon_0}$, we obtain the cold plasma dielectric tensor. 

In a local coordinate system where $\boldsymbol{B}_0$ is aligned with the $z$-axis, the dielectric tensor takes the following simplified form using the Stix notation:

$$\boldsymbol{\varepsilon} = \begin{pmatrix} S & -iD & 0 \\ iD & S & 0 \\ 0 & 0 & P \end{pmatrix}$$

Here, the Stix parameters $S$, $D$, and $P$ are defined in terms of the standard dimensionless parameters $X = \omega_p^2/\omega^2$ and $Y = \omega_c/\omega$ (with $\omega_p = \sqrt{e^2 n_e/(\epsilon_0 m_e)}$ the electron plasma frequency and $\omega_c = eB_0/m_e$ the electron cyclotron frequency):

- $S = 1 - \frac{X}{1-Y^2}$
- $D = \frac{XY}{1-Y^2}$
- $P = 1 - X$

Returning to the dispersion tensor $\boldsymbol{\mathsf{D}} = \boldsymbol{\varepsilon} - n^2 \boldsymbol{I} + \boldsymbol{n}\boldsymbol{n}$, we assume the wave vector $\boldsymbol{n}$ lies in the $x$-$z$ plane at an angle $\theta$ to the background magnetic field, such that $\boldsymbol{n} = (n \sin \theta, 0, n \cos \theta)$.

Substituting $\boldsymbol{\varepsilon}$ into $\boldsymbol{\mathsf{D}}$ and enforcing the non-trivial solution condition $\det \boldsymbol{\mathsf{D}} = 0$ yields a quadratic equation in $n^2$. Solving it gives the Appleton-Hartree dispersion relation, which expresses the refractive index as a function of the propagation angle $\theta$ and plasma parameters:

$$n_{AH}^2 = 1 - \frac{X}{1 - \frac{Y^2 \sin^2 \theta}{2(1-X)} \pm \sqrt{\left(\frac{Y^2 \sin^2 \theta}{2(1-X)}\right)^2 + Y^2 \cos^2 \theta}}$$

The $\pm$ sign dictates the two characteristic wave modes of the magnetized plasma: the Ordinary (O) wave (−) and the Extraordinary (X) wave (+).

Rather than solving for the roots of the dispersion relation directly, Raytrax uses a Hamiltonian formulation based on the implicit dispersion relation. The code implements the cold plasma Hamiltonian as:

$$ \mathcal{H}(\boldsymbol{r}, \boldsymbol{n}) = |\boldsymbol{n}|^2 - n_{AH}^2(\omega, \theta(\boldsymbol{n}, \boldsymbol{r}), \boldsymbol{r}) $$

where $n_{AH}^2$ is the Appleton-Hartree refractive index squared calculated above. This form ensures that $\mathcal{H}=0$ is equivalent to satisfying the dispersion relation. The gradients of this Hamiltonian with respect to position and wave vector drive the ray tracing equations.

The appropriate sign in the Appleton-Hartree formula is determined by the polarization of the wave mode being traced. Raytrax allows users to specify which mode (O or X) they wish to trace, ensuring that the correct branch of the dispersion relation is used in the Hamiltonian.


!!! info

    Currently, only the cold plasma Hamiltonian is used for tracing in Raytrax. The [absorption coefficient](absorption.md) is calculated fully relativistically.