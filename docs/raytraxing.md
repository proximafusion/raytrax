---
icon: lucide/move
---

# Ray Tracing Equations

This document explains the ray tracing equations used in Raytrax, starting from the dispersion relation defined in [Theory.md](theory.md). We derive the Hamiltonian formulation of ray equations and show how the cold plasma approximation leads to the implementation in `hamiltonian.py`.

## Hamiltonian Formulation

The ray equations describe how the position **$\boldsymbol{r}$** and refractive index **$\boldsymbol{N}$** evolve along the ray trajectory. In the geometrical optics approximation, these equations can be elegantly expressed in Hamiltonian form using the arc length $s$ as the independent variable:

$$\frac{d\boldsymbol{r}}{ds} = \frac{\partial H}{\partial \boldsymbol{N}}$$

$$\frac{d\boldsymbol{N}}{ds} = -\frac{\partial H}{\partial \boldsymbol{r}}$$

where $H(\boldsymbol{r}, \boldsymbol{N})$ is the Hamiltonian function. This formulation is analogous to Hamilton's equations in classical mechanics, with $\boldsymbol{r}$ playing the role of position and $\boldsymbol{N}$ the role of momentum.

### Connection to the Dispersion Relation

The Hamiltonian is derived from the dispersion relation $\det \boldsymbol{D} = 0$ introduced in Theory.md. For a cold plasma, the dispersion relation can be written as:

$$H(\boldsymbol{r}, \boldsymbol{N}) = N^2 - N^2_{\text{disp}}(\boldsymbol{r}, \boldsymbol{N}) = 0$$

where $N = |\boldsymbol{N}|$ and $N^2_{\text{disp}}$ is obtained from solving the dispersion relation for a given wave mode (ordinary or extraordinary). The Hamiltonian is constructed such that it vanishes when the dispersion relation is satisfied, i.e., $H = 0$ along the ray.

The ray equations then follow from the requirement that $H$ remains zero along the trajectory. Taking the total derivative:

$$\frac{dH}{ds} = \frac{\partial H}{\partial \boldsymbol{r}} \cdot \frac{d\boldsymbol{r}}{ds} + \frac{\partial H}{\partial \boldsymbol{N}} \cdot \frac{d\boldsymbol{N}}{ds} = 0$$

This is automatically satisfied if we choose the Hamiltonian form of the ray equations given above.

## Cold Plasma Approximation

**Raytrax currently only implements cold plasma ray tracing.** The cold plasma approximation assumes that thermal effects can be neglected, which is valid when the electron temperature is much smaller than the wave energy. This simplifies the dielectric tensor significantly and leads to an analytic dispersion relation.

In the cold plasma limit, the dispersion relation is given by the **Appleton-Hartree equation**, which relates $N^2$ to the local plasma parameters:

$$N^2 = 1 - \frac{2X(1-X)}{2(1-X) - Y^2 \sin^2\theta \pm \sqrt{Y^4 \sin^4\theta + 4(1-X)^2 Y^2 \cos^2\theta}}$$

where:

- $X = \omega_{pe}^2 / \omega^2$ is the normalized plasma frequency squared
- $Y = \omega_{ce} / \omega$ is the normalized cyclotron frequency
- $\theta$ is the angle between the wave vector $\boldsymbol{k}$ (or equivalently $\boldsymbol{N}$) and the magnetic field $\boldsymbol{B}$
- The $\pm$ sign determines the wave mode: minus for extraordinary mode (X-mode), plus for ordinary mode (O-mode)

The plasma frequency $\omega_{pe}$ and cyclotron frequency $\omega_{ce}$ depend on the local electron density $n_e$ and magnetic field strength $|\boldsymbol{B}|$:

$$\omega_{pe} = \sqrt{\frac{n_e e^2}{\epsilon_0 m_e}}$$

$$\omega_{ce} = \frac{e |\boldsymbol{B}|}{m_e}$$

## Cold Hamiltonian in `hamiltonian.py`

The implementation in `hamiltonian.py` directly follows from the Hamiltonian formulation described above. The `_hamiltonian_cold` function computes:

$$H = N^2 - N^2_{\text{disp}}$$

where $N^2_{\text{disp}}$ is evaluated using the Appleton-Hartree formula implemented in `dispersion.py`.

### Key Steps in the Implementation

1. **Decompose the refractive index**: The refractive index vector $\boldsymbol{N}$ is decomposed into components parallel and perpendicular to the magnetic field:
   - $N_\parallel = \boldsymbol{N} \cdot \hat{\boldsymbol{B}}$
   - $N_\perp = |\boldsymbol{N} - N_\parallel \hat{\boldsymbol{B}}|$

2. **Compute plasma parameters**: The cyclotron frequency $\omega_{ce}$ and plasma frequency $\omega_{pe}$ are computed from the local magnetic field and electron density.

3. **Evaluate dispersion relation**: The `dispersion_cold` function computes $N^2_{\text{disp}}$ using the Appleton-Hartree equation with the decomposed refractive index components.

4. **Form the Hamiltonian**: The Hamiltonian is $H = N^2 - \text{Re}(N^2_{\text{disp}})$, where only the real part is taken since cold plasma theory yields a real-valued dispersion relation (neglecting absorption).

### Vacuum Case

In vacuum (when electron density is negligible), the Hamiltonian simplifies to:

$$H_{\text{vacuum}} = N^2 - 1$$

This corresponds to the vacuum dispersion relation $k^2 = \omega^2/c^2$, or equivalently $N^2 = 1$.

## Ray Trajectory Computation

The ray equations are integrated numerically using the gradients of the Hamiltonian:

- $\frac{d\boldsymbol{r}}{ds} = \frac{\partial H}{\partial \boldsymbol{N}}$ determines how the position changes along the ray
- $\frac{d\boldsymbol{N}}{ds} = -\frac{\partial H}{\partial \boldsymbol{r}}$ determines how the refractive index (wave vector direction) changes due to spatial variations in plasma parameters

The `hamiltonian_gradients` function in `hamiltonian.py` uses JAX's automatic differentiation to compute both gradients efficiently in a single backward pass.

## Limitations and Future Extensions

Since Raytrax currently implements only **cold plasma ray tracing**, several physical effects are not captured:

- **Thermal effects**: At high temperatures or when the wave frequency is close to a resonance, finite temperature corrections become important
- **Kinetic effects**: Landau damping and other kinetic phenomena require a kinetic treatment beyond the cold plasma approximation
- **Relativistic effects**: At very high temperatures or energies, relativistic corrections to the dispersion relation may be needed

These effects could be incorporated in future versions by extending the dispersion relation and Hamiltonian to include thermal or kinetic corrections.
