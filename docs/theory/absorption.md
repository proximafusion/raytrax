---
icon: lucide/activity
---

# Absorption Calculation

The absorption coefficient $\alpha$ is the ratio of the power absorbed per unit volume to the power flux carried by the wave. Raytrax evaluates it at every point of the ray trajectory from three ingredients:

1. the **polarization vector** $\hat{\boldsymbol{e}}$ of the wave, obtained from the weakly relativistic dielectric tensor,
2. the **anti-Hermitian part** of the dielectric tensor, obtained from a fully relativistic resonance integral over the electron distribution function, and
3. the **power flux vector** $\boldsymbol{F}$, obtained from the cold dielectric tensor.

The sections below derive the absorption coefficient and describe each ingredient.

## Loss Density

We start from Poynting's theorem in the absence of free currents,

$$\nabla \cdot \boldsymbol{S} = -\frac{\partial u}{\partial t} = -\boldsymbol{E} \cdot \frac{\partial \boldsymbol{D}}{\partial t} - \boldsymbol{H} \cdot \frac{\partial \boldsymbol{B}}{\partial t}$$

where $\boldsymbol{S}$ is the power flux density, $u$ the energy density, $\boldsymbol{D}=\varepsilon_0 \boldsymbol{\varepsilon_r}\boldsymbol{E}$, and $\boldsymbol{B}=\mu_0 \boldsymbol{H}$.

For harmonic fields $\boldsymbol{E}(\boldsymbol{r},t)=\mathrm{Re}\left[\hat{\boldsymbol{E}}(\boldsymbol{r}) e^{-i \omega t}\right]$, the time average of this equation can be written as

$$\langle \nabla \cdot \boldsymbol{S} \rangle = -\frac{\varepsilon_0 \omega}{2} \mathrm{Im}(\hat{\boldsymbol{E}}^* \cdot \boldsymbol{\varepsilon_r} \cdot \hat{\boldsymbol{E}})$$

Splitting the relative dielectric tensor into its Hermitian and anti-Hermitian parts, $\boldsymbol{\varepsilon_r}=\boldsymbol{\varepsilon}_r^H + i \boldsymbol{\varepsilon}_r^A$, where

$$\boldsymbol{\varepsilon}_{r}^H = \frac{1}{2}(\boldsymbol{\varepsilon}_{r} + \boldsymbol{\varepsilon}_{r}^\dagger),$$

$$\boldsymbol{\varepsilon}_{r}^A = \frac{1}{2i}(\boldsymbol{\varepsilon}_{r} - \boldsymbol{\varepsilon}_{r}^\dagger),$$

the time-averaged power loss density (units W/m³) becomes

$$Q = -\langle \nabla \cdot \boldsymbol{S} \rangle = \frac{\varepsilon_0 \omega}{2} (\hat{\boldsymbol{E}}^* \cdot \boldsymbol{\varepsilon}_r^A \cdot \hat{\boldsymbol{E}})$$

Both $\boldsymbol{\varepsilon}_r^H$ and $\boldsymbol{\varepsilon}_r^A$ are Hermitian matrices, so the quadratic form on the right is real. It is positive in an absorbing plasma.

## Power Flux Density

The time average of the power flux density (Poynting vector) is given by the energy density times the group velocity,

$$\left\langle\boldsymbol{S}\right\rangle = \left\langle u \right\rangle \boldsymbol{v}_g$$

Using the Hermitian part of the dispersion tensor $\boldsymbol{\mathsf{D}}^H=\frac{1}{2}(\boldsymbol{\mathsf{D}} + \boldsymbol{\mathsf{D}}^\dagger)$ (see [overview](../theory.md#dispersion-relation)), it can be shown that this quantity can also be written as

$$\left\langle\boldsymbol{S}\right\rangle =  -\frac{\varepsilon_0 \omega}{4} \,\mathrm{Re}(\hat{\boldsymbol{E}}^* \cdot \frac{\partial\boldsymbol{\mathsf{D}}^H}{\partial {\boldsymbol{k}} } \cdot \hat{\boldsymbol{E}})$$

Writing $\hat{\boldsymbol{E}} = \hat E \hat{\boldsymbol{e}}$, where $\hat{\boldsymbol{e}}$ is the unit polarization vector, and using $\partial/\partial\boldsymbol{k} = (c/\omega)\,\partial/\partial\boldsymbol{N}$ with the refractive index vector $\boldsymbol{N} = c\boldsymbol{k}/\omega$, this becomes

$$\left\langle\boldsymbol{S}\right\rangle = \frac{\varepsilon_0 c}{2}|\hat E|^2 \boldsymbol{F}, \qquad \boldsymbol{F} = -\frac{1}{2}\frac{\partial}{\partial\boldsymbol{N}}\mathrm{Re}\!\left(\hat{\boldsymbol{e}}^* \cdot \boldsymbol{\mathsf{D}}^H \cdot \hat{\boldsymbol{e}}\right)$$

The dimensionless vector $\boldsymbol{F}$ is the **power flux vector**. In vacuum, $\hat{\boldsymbol{e}}^* \cdot \boldsymbol{\mathsf{D}} \cdot \hat{\boldsymbol{e}} = 1 - N^2$ for a transverse wave, so $\boldsymbol{F} = \boldsymbol{N}$ with $|\boldsymbol{F}| = 1$. This recovers the vacuum Poynting flux $\varepsilon_0 c |\hat E|^2/2$.

## Absorption Coefficient

The power carried by the ray decays as $dP/ds = -\alpha P$, so the **absorption coefficient** (units 1/m) is the ratio of the power loss density (units W/m³) to the power flux density (units W/m²),

$$\alpha = \frac{Q}{|\langle\boldsymbol{S}\rangle|}$$

The magnitude of the electric field drops out, and we obtain

$$\alpha = \frac{\omega}{c}\,\frac{\hat{\boldsymbol{e}}^* \cdot \boldsymbol{\varepsilon}_r^A \cdot \hat{\boldsymbol{e}}}{|\boldsymbol{F}|}$$

It is implemented in [`absorption_coefficient`][raytrax.physics.absorption.absorption_coefficient].

## Optical Depth

The absorption coefficient $\alpha$ is related to the (dimensionless) **optical depth** $\tau$ as

$$\frac{d\tau}{ds}=\alpha$$

where $s$ is the arc length, and the **linear absorption power density** (units W/m) along the ray trajectory is given as

$$\frac{dP}{ds}=P_0\alpha e^{-\tau}$$

where $P_0$ is the initial power of the ray.

## Polarization Vector

All tensors in this section are expressed in **Stix coordinates**: the $z$ axis is aligned with the magnetic field $\boldsymbol{B}_0$ and the $x$ axis is chosen such that the refractive index vector lies in the $x$–$z$ plane, $\boldsymbol{N} = (N_\perp, 0, N_\parallel)$. In these coordinates the dispersion tensor reads

$$\boldsymbol{\mathsf{D}} = \begin{pmatrix} \varepsilon_{xx} - N_\parallel^2 & \varepsilon_{xy} & \varepsilon_{xz} + N_\perp N_\parallel \\ \varepsilon_{yx} & \varepsilon_{yy} - N^2 & \varepsilon_{yz} \\ \varepsilon_{zx} + N_\perp N_\parallel & \varepsilon_{zy} & \varepsilon_{zz} - N_\perp^2 \end{pmatrix}$$

The polarization vector is a solution of $\boldsymbol{\mathsf{D}} \cdot \hat{\boldsymbol{e}} = 0$. Raytrax fixes the component that dominates for quasi-perpendicular propagation and solves a $2\times 2$ subsystem for the other two:

- **O mode**: $e_z = 1$; $e_x$ and $e_y$ follow from the first and second rows.
- **X mode**: $e_y = 1$; $e_x$ and $e_z$ follow from the first and third rows.

The result is then normalized to $|\hat{\boldsymbol{e}}| = 1$.

The refractive index $\boldsymbol{N}$ at which $\boldsymbol{\mathsf{D}}$ is evaluated comes from the ray trajectory, which is traced with the cold plasma Hamiltonian (see [ray tracing](ray-tracing.md)). It therefore does not satisfy the dispersion relation of the warm tensor exactly, and the equation that is left out is not satisfied exactly either. Selecting a $2\times 2$ subsystem gives a well-defined polarization vector in this situation.

It is implemented in [`polarization`][raytrax.physics.polarization.polarization].

### Weakly Relativistic Dielectric Tensor

Absorption depends on how $\hat{\boldsymbol{e}}$ projects onto the gyration of resonant electrons (see below). This projection can be small, e.g. for the X mode at the fundamental harmonic. It is then sensitive to thermal corrections to the Hermitian part of the dielectric tensor close to the resonance. For this reason, raytrax follows Travis[^1] and computes $\hat{\boldsymbol{e}}$ from the weakly relativistic dielectric tensor of Krivenski and Orefice[^2] rather than from the cold tensor.

In this approximation the dielectric tensor takes the form

$$\boldsymbol{\varepsilon}_r = \boldsymbol{I} - \mu X\, \boldsymbol{\mathcal{D}}$$

where $X = \omega_p^2/\omega^2$, $\mu = m_e c^2 / T_e$, and the elements of $\boldsymbol{\mathcal{D}}$ are finite sums over harmonics $|s| \le s_\text{max}$ and finite Larmor radius (FLR) orders $k \le k_\text{max}$. Each term is a power $\lambda^{|s|+k-1}$ of the FLR parameter

$$\lambda = \frac{(k_\perp \rho_\text{th})^2}{2} = \frac{N_\perp^2 \omega^2}{\mu\, \omega_c^2}$$

multiplied by combinations of the Shkarofsky functions $\mathcal{F}_{q+1/2}$[^3], which contain the weakly relativistic resonance denominators. As in Krivenski and Orefice, the electron cyclotron frequency enters with its sign, $\omega_c < 0$.

Raytrax sets $s_\text{max} = k_\text{max}$ to [`Beam.max_harmonic`][raytrax.types.Beam]. The FLR expansion order thus is at least as large as the highest resonant harmonic, so the polarization vector contains the leading-order FLR correction at each resonance that is included.

It is implemented in [`weakly_relativistic_dielectric_tensor`][raytrax.physics.dielectric_tensor.weakly_relativistic_dielectric_tensor].

## Relativistic Resonance Integral

The weakly relativistic tensor expands the relativistic factor of the electrons to first order. The anti-Hermitian part, which controls the absorbed power, is sensitive to the shape of the distribution function in the resonant region. Raytrax therefore evaluates it from the fully relativistic expression[^4] [^5].

### Resonance Condition

We use the normalized momentum $\boldsymbol{u} = \boldsymbol{p}/(m_e c)$, with components $u_\parallel$ and $u_\perp$ relative to $\boldsymbol{B}_0$, the Lorentz factor $\gamma = \sqrt{1 + u_\parallel^2 + u_\perp^2}$, and $Y = |\omega_c|/\omega$. An electron interacts resonantly with the wave at harmonic $s$ if

$$\gamma = sY + N_\parallel u_\parallel$$

Inserting this into the definition of $\gamma$ gives

$$u_\perp^2 = (sY + N_\parallel u_\parallel)^2 - 1 - u_\parallel^2$$

For $|N_\parallel| < 1$ this is an ellipse in the $(u_\parallel, u_\perp)$ plane. It intersects the $u_\parallel$ axis at

$$u_\pm = \frac{sY N_\parallel \pm \sqrt{s^2 Y^2 + N_\parallel^2 - 1}}{1 - N_\parallel^2}$$

In this regime, a resonance exists only if $s^2 Y^2 + N_\parallel^2 > 1$. Since $\gamma \ge 1$, absorption therefore takes place only where $s|\omega_c| > \omega\sqrt{1 - N_\parallel^2}$, i.e. on the high-field side of the cold resonance for perpendicular propagation (*relativistic downshift*). For $s \le 0$ and $|N_\parallel| < 1$ the resonance condition has no solution, so only positive harmonics $s = 1, \dots, s_\text{max}$ contribute.

At $|N_\parallel|=1$ the resonance is a parabola; above 1 it is a hyperbola. The positive-energy branch extends from a finite endpoint to infinity in the direction of $N_\parallel$. Zero and negative harmonics can resonate above 1 but remain omitted.

### Anti-Hermitian Part

For an isotropic distribution function $f(\gamma)$, normalized as $\int f\, d^3u = 1$, the anti-Hermitian part contracted with the polarization vector is

$$\hat{\boldsymbol{e}}^* \cdot \boldsymbol{\varepsilon}_r^A \cdot \hat{\boldsymbol{e}} = -2\pi^2 X \sum_{s=1}^{s_\text{max}} \int_{\mathcal{R}_s} du_\parallel\, \left|\hat{\boldsymbol{e}} \cdot \boldsymbol{\Pi}_s\right|^2 \frac{\partial f}{\partial \gamma}\bigg|_{\gamma = sY + N_\parallel u_\parallel}$$

where $\mathcal{R}_s$ is $[u_-,u_+]$ for an ellipse and the positive-energy branch otherwise, $u_\perp$ is given by the resonance condition, and

$$\boldsymbol{\Pi}_s = \left(\frac{sY}{N_\perp} J_s(b),\; -i u_\perp J_s'(b),\; u_\parallel J_s(b)\right), \qquad b = \frac{N_\perp u_\perp}{Y}$$

Here $b = k_\perp v_\perp \gamma / |\omega_c|$ is the Larmor radius argument and $J_s$ is the Bessel function of the first kind. The product $\hat{\boldsymbol{e}} \cdot \boldsymbol{\Pi}_s = \sum_i e_i \Pi_{s,i}$ is taken without complex conjugation. Up to normalization, $|\hat{\boldsymbol{e}} \cdot \boldsymbol{\Pi}_s|^2$ is the quasilinear diffusion coefficient of the resonant electrons[^4]. For a general, anisotropic distribution, $\partial f/\partial\gamma$ is replaced by $\partial f/\partial\gamma + N_\parallel\, \partial f/\partial u_\parallel|_\gamma$, but the second term vanishes for isotropic $f$.

Raytrax uses the relativistic Maxwell–Jüttner distribution

$$f(\gamma) = \frac{\mu}{4\pi K_2(\mu)}\, e^{-\mu\gamma}, \qquad \frac{\partial f}{\partial\gamma} = -\mu f$$

where $K_2$ is the modified Bessel function of the second kind. Since $\partial f/\partial \gamma < 0$, the result is positive, as required for an absorbing plasma.

In the weakly relativistic limit, the expression above agrees with the anti-Hermitian part of the Krivenski–Orefice tensor. The unit tests check this.

### Numerical Evaluation

Along the physical resonance curve, the Maxwell–Jüttner factor decays from the low-$\gamma$ endpoint like $e^{-\mu |N_\parallel| \Delta u_\parallel}$. Gauss–Legendre quadrature covers $K$ decay lengths $1/(\mu |N_\parallel|)$ from that endpoint toward increasing $\gamma$, capped at the full interval for an ellipse. The integrand is analytic on the window, so the quadrature converges exponentially.

The window length $K$ and the number of nodes follow from the requested relative accuracy of each integral, the `resonance_rtol` argument of [`absorption_coefficient`][raytrax.physics.absorption.absorption_coefficient], and from the harmonic $s$, since the integrand behaves like $t^s e^{-t}$ near the low-$\gamma$ end ($t$ in decay lengths). For the default $10^{-14}$, the window is about 36 decay lengths with 24 nodes at $s = 1$ and 39 decay lengths with 26 nodes at $s = 2$.

A harmonic whose low-energy endpoint lies outside $|u_\parallel| < 7\,v_\text{th}/c$ contributes zero. If no harmonic reaches it, the absorption coefficient is zero, and the dielectric tensor, polarization and power flux are not evaluated.

It is implemented in [`anti_hermitian_dielectric_form`][raytrax.physics.absorption.anti_hermitian_dielectric_form].

## Power Flux Vector

The power flux vector $\boldsymbol{F}$ in the denominator of the absorption coefficient is evaluated with the cold dielectric tensor, and the polarization vector inside the gradient is computed from the cold tensor as well. Differentiating the weakly relativistic tensor with respect to $\boldsymbol{N}$ near a cyclotron harmonic leads to resonant singularities, which the cold tensor avoids. The thermal corrections to $\boldsymbol{F}$ are of order $v_\text{th}/c$. This approximation is least accurate at the fundamental harmonic, in particular for X mode, where the thermal corrections to the Hermitian part of the dielectric tensor are largest.

It is implemented in [`cold_power_flux_vector_stix`][raytrax.physics.power_flux.cold_power_flux_vector_stix].

## Summary of Plasma Models

Different parts of the calculation use different plasma models:

| Quantity | Plasma model | Rationale |
|---|---|---|
| Ray trajectory $\boldsymbol{r}(s)$, $\boldsymbol{N}(s)$ | cold | robust Hamiltonian, no resonant singularities |
| Polarization vector $\hat{\boldsymbol{e}}$ | weakly relativistic | thermal corrections to $\boldsymbol{\varepsilon}_r^H$ near resonance |
| $\hat{\boldsymbol{e}}^* \cdot \boldsymbol{\varepsilon}_r^A \cdot \hat{\boldsymbol{e}}$ | fully relativistic | accurate resonant electron dynamics |
| Power flux vector $\boldsymbol{F}$ | cold | avoids singular $\partial\boldsymbol{\varepsilon}_r/\partial\boldsymbol{N}$ near resonance |

Because the trajectory is traced with the cold Hamiltonian, $\boldsymbol{N}$ does not include the thermal modification of the refractive index near the resonance. See [limitations](../limitations.md).

[^1]: Marushchenko, Nikolai B., Yu Turkin, and Henning Maaßberg. "Ray-tracing code TRAVIS for ECR heating, EC current drive and ECE diagnostic." Computer Physics Communications 185.1 (2014): 165-176. [doi:10.1016/j.cpc.2013.09.002](https://doi.org/10.1016/j.cpc.2013.09.002)
[^2]: Krivenski and Orefice, J. Plasma Physics (1983), vol. 30, part 1, pp. 125-131, [doi:10.1017/S0022377800001045](https://doi.org/10.1017/S0022377800001045)
[^3]: Shkarofsky, J. Plasma Physics (1986), vol. 35, part 2, pp. 319-331.
[^4]: Marushchenko, N. B., Maassberg, H., & Turkin, Yu. (2008). Nuclear Fusion, 48(5), 054002. [doi:10.1088/0029-5515/48/5/054002](https://doi.org/10.1088/0029-5515/48/5/054002)
[^5]: Bornatici, M., Cano, R., De Barbieri, O., & Engelmann, F. (1983). Electron cyclotron emission and absorption in fusion plasmas. Nuclear Fusion, 23(9), 1153-1257. [doi:10.1088/0029-5515/23/9/005](https://doi.org/10.1088/0029-5515/23/9/005)
