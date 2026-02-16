---
icon: lucide/spline
---

# Ray Tracing

## Ray Tracing Equations

Since the determinant of the dispersion tensor $\boldsymbol{D}$ has to vanish along the actual ray path $\boldsymbol{r}(s)$, it can be interpreted as a Hamiltonian $\mathcal H(\boldsymbol{r}, \boldsymbol{k})$ in the six-dimensional phase space of position $\boldsymbol{r}$ and wave vector $\boldsymbol{k}$.

The ray equations can then be derived from Hamilton's equations in terms of a time-like parameter $\tau$ along the ray:

$$\frac{d\boldsymbol{r}}{d\tau} = \frac{\partial \mathcal H}{\partial \boldsymbol{k}}, \quad \frac{d\boldsymbol{k}}{d\tau} = -\frac{\partial \mathcal H}{\partial \boldsymbol{r}}$$

Using the arc length $s$ along the ray, which satisfies

$$\frac{ds}{d\tau} = \left|\frac{d\boldsymbol{r}}{d\tau}\right|$$

we can write

$$\frac{d\boldsymbol{r}}{ds} = \left|\frac{\partial \mathcal H}{\partial \boldsymbol{k}}\right|^{-1} \frac{\partial \mathcal H}{\partial \boldsymbol{k}}, \quad \frac{d\boldsymbol{k}}{ds} = -\left|\frac{\partial \mathcal H}{\partial \boldsymbol{k}}\right|^{-1} \frac{\partial \mathcal H}{\partial \boldsymbol{r}}$$

or, equivalently,

$$\frac{d\boldsymbol{r}}{ds} = \left|\frac{\partial \mathcal H}{\partial \boldsymbol{n}}\right|^{-1} \frac{\partial \mathcal H}{\partial \boldsymbol{n}}, \quad \frac{d\boldsymbol{n}}{ds} = -\left|\frac{\partial \mathcal H}{\partial \boldsymbol{n}}\right|^{-1} \frac{\partial \mathcal H}{\partial \boldsymbol{r}}$$

where $\boldsymbol{n}=\boldsymbol{k}c/\omega$.

The gradients $\partial \mathcal H/\partial \boldsymbol{r}$ and $\partial \mathcal H/\partial \boldsymbol{n}$ are computed in Raytrax using automated differentation in the function `hamiltonian_gradients`, starting from a scalar Hamiltonian function.

## Cold Tracing

The easiest situation is that of a „cold“ plasma, where relativistic effects can be neglected. In this case, ... TBC