---
icon: lucide/home
---

# Raytrax: ECRH ray tracing in JAX

Raytrax is a Python library for the simulation of Electron Cyclotron Resonance Heating (ECRH) of fusion plasmas. Based on [JAX](https://docs.jax.dev), it features fast computation due to just-in-time (JIT) compilation and allows for automated differentation of its result, making it particularly suited for parameter optimization in fusion power plant design.

Raytrax employs the geometric optics (or WKB) approximation for microwave rays and solves the ray tracing equations to determine the ray trajectory. The energy absorption that determines the heating deposition is calculated relativistically. The physics approach closely follows the Travis[^1] code.

Raytrax is released under the MIT License.

!!! info

    Raytrax is currently in an early stage of development. Please expect API changes. The code has not been fully validated against existing ray tracing codes or experimental data yet. See [limitations](limitations.md) for a list of current limitations.


[^1]: Marushchenko, Nikolai B., Yu Turkin, and Henning Maaßberg. "Ray-tracing code TRAVIS for ECR heating, EC current drive and ECE diagnostic." Computer Physics Communications 185.1 (2014): 165-176. [doi:10.1016/j.cpc.2013.09.002](https://doi.org/10.1016/j.cpc.2013.09.002)



---

[![](https://upload.wikimedia.org/wikipedia/commons/d/d3/Logo_Proxima_Fusion.svg){ width="200" align="left" }](https://www.proximafusion.com)

[![](https://upload.wikimedia.org/wikipedia/de/1/1e/Hochschule_f%C3%BCr_angewandte_Wissenschaften_M%C3%BCnchen_logo.svg){width="150" align="right"}](https://www.hm.edu)

<div style="clear:both"></div>