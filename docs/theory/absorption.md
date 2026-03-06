---
icon: lucide/activity
---

# Absorption Calculation

## Loss Density


We start from Poynting's theorem in the absence of free currents,

$$\nabla \cdot \boldsymbol{S} = -\frac{\partial u}{\partial t} = \boldsymbol{E} \cdot \frac{\partial \boldsymbol{D}}{\partial t} - \boldsymbol{H} \cdot \frac{\partial \boldsymbol{B}}{\partial t}$$

where $\boldsymbol{S}$ is the power flux density, $u$ the energy density, $\boldsymbol{D}=\varepsilon_0 \boldsymbol{\varepsilon_r}\boldsymbol{E}$, and $\boldsymbol{B}=\mu_0 \boldsymbol{H}$.

For harmonic fields $\boldsymbol{E}(\boldsymbol{r},t)=\mathrm{Re}\left[\hat{\boldsymbol{E}}(\boldsymbol{r}) e^{-i \omega t}\right]$, the time average of this equation can be written as

$$\langle \nabla \cdot \boldsymbol{S} \rangle = -\frac{\varepsilon_0 \omega}{2} \mathrm{Im}(\hat{\boldsymbol{E}}^* \cdot \boldsymbol{\varepsilon_r} \cdot \hat{\boldsymbol{E}})$$

Splitting the relative dielectric tensor into its Hermitian and anti-Hermitian parts, $\boldsymbol{\varepsilon_r}=\boldsymbol{\varepsilon}_r^H + i \boldsymbol{\varepsilon}_r^A$, where

$$\boldsymbol{\varepsilon}_{r}^H = \frac{1}{2}(\boldsymbol{\varepsilon}_{r} + \boldsymbol{\varepsilon}_{r}^\dagger),$$

$$\boldsymbol{\varepsilon}_{r}^A = \frac{1}{2i}(\boldsymbol{\varepsilon}_{r} - \boldsymbol{\varepsilon}_{r}^\dagger),$$

 the time-averaged power loss density (units W/m³) becomes

$$Q = \left\langle\frac{\partial u}{\partial t}\right\rangle = \frac{\varepsilon_0 \omega}{2} (\hat{\boldsymbol{E}}^* \cdot \boldsymbol{\varepsilon}_r^A \cdot \hat{\boldsymbol{E}})$$

## Power Flux Density

The time average of the power flux density (Poynting vector) is given by the energy density times the group velocity,

$$\left\langle\boldsymbol{S}\right\rangle = \left\langle u \right\rangle \boldsymbol{v}_g$$

Using the Hermitian part of the dispersion tensor $\boldsymbol{\mathsf{D}}^H=\frac{1}{2}(\boldsymbol{\mathsf{D}} + \boldsymbol{\mathsf{D}}^\dagger)$, it can be shown that this quantity can also be written as

$$\left\langle\boldsymbol{S}\right\rangle =  -\frac{\varepsilon_0 \omega}{4} \,\mathrm{Re}(\hat{\boldsymbol{E}}^* \cdot \frac{\partial\boldsymbol{\mathsf{D}}^H}{\partial {\boldsymbol{k}} } \cdot \hat{\boldsymbol{E}})$$


## Absorption Coefficient


The **absorption coefficient**  (units 1/m) can now be expressed in terms of the power loss density (units W/m³) and the power flux density (units W/m²) as

$$\alpha = \frac{Q}{2|\langle\boldsymbol{S}\rangle|}$$

Writing $\hat{\boldsymbol{E}} = \hat E \hat{\boldsymbol{e}}$ where $\hat{\boldsymbol{e}}$ is the unit polarization vector, the magnitude of the E field drops out and we obtain

$$\alpha = \frac{\hat{\boldsymbol{e}}^* \cdot \boldsymbol{\varepsilon}_r^A \cdot \hat{\boldsymbol{e}}}{|\frac{\partial}{\partial {\boldsymbol{k}} }\mathrm{Re}(\hat{\boldsymbol{e}}^* \cdot \boldsymbol{\mathsf{D}}^H \cdot \hat{\boldsymbol{e}})|}$$


## Optical Depth


The absorption coefficient $\alpha$ is related to the (dimensionless) **optical depth** $\tau$ as

$$\frac{d\tau}{ds}=\alpha$$

where $s$ is the arc length, and the **linear absorption power density** (units W/m) along the ray trajectory is given as

$$\frac{dP}{ds}=P_0\alpha e^{-\tau}$$

where $P_0$ is the initial power of the ray.

## Weakly Relativistic Dielectric Tensor

For the hermitian part of the dispersion tensor appearing in the denominator of the absorption coefficient,

$$\boldsymbol{\mathsf{D}}^H = \boldsymbol{\varepsilon_r}^H - n^2 \boldsymbol{I} + \boldsymbol{n}\boldsymbol{n}$$

raytrax follows Travis[^1] by using the weakly relativistic dielectric tensor taken from Krivenski and Orefice[^2].

It is implemented in `raytrax.physics.dielectric_tensor.weakly_relativistic_dielectric_tensor`.


## Fully Relativistic Dielectric Tensor

For the anti-hermitian part of the dielectric tensor appearing in the numerator of the absorption coefficient, the fully relativistic dielectric tensor is required, which is computed using the integral approach[^3].

[^1]: Marushchenko, Nikolai B., Yu Turkin, and Henning Maaßberg. "Ray-tracing code TRAVIS for ECR heating, EC current drive and ECE diagnostic." Computer Physics Communications 185.1 (2014): 165-176. [doi:10.1016/j.cpc.2013.09.002](https://doi.org/10.1016/j.cpc.2013.09.002)
[^2]: Krivenski and Orefice, J. Plasma Physics (1983), vol. 30, part 1, pp. 125-131, [doi:10.1017/S0022377800001045](https://doi.org/10.1017/S0022377800001045)
[^3]: Bornatici, M., Cano, R., De Barbieri, O., & Engelmann, F. (1983). Electron cyclotron emission and absorption in fusion plasmas. Nuclear Fusion, 23(9), 1153-1257. [doi:10.1088/0029-5515/23/9/005](https://doi.org/10.1088/0029-5515/23/9/005)

