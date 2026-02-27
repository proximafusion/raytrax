# Raytrax

**An ECRH ray tracer for fusion plasmas, built on JAX.**

[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](https://opensource.org/licenses/MIT)
[![Python ≥ 3.11](https://img.shields.io/badge/python-%E2%89%A53.11-blue)](https://www.python.org/)
[![Documentation](https://img.shields.io/badge/docs-latest-informational)](https://didactic-memory-qm87kjk.pages.github.io/)

Raytrax is a Python library for simulating Electron Cyclotron Resonance Heating (ECRH) of magnetic confinement fusion plasmas. Powered by [JAX](https://docs.jax.dev), it features fast JIT-compiled computation and automatic differentiation for gradient-based optimization.

> **Note:** Raytrax is in early development. Expect API changes and incomplete validation.

## Installation

```bash
python -m pip install raytrax
```

## Acknowledgements

The development of Raytrax is a collaboration between [Proxima Fusion](https://www.proximafusion.com) and the [Munich University of Applied Sciences (HM)](https://www.hm.edu) and was partially supported by the German Federal Ministry of Research, Technology and Space (BMFTR) under grant FPP-MC (13F1001B).

<div style="display: flex; gap: 2rem; align-items: center; justify-content: center; margin-top: 2rem;">
<img src="https://upload.wikimedia.org/wikipedia/commons/d/d3/Logo_Proxima_Fusion.svg" alt="Proxima Fusion" width="200"/>
<img src="https://upload.wikimedia.org/wikipedia/de/1/1e/Hochschule_f%C3%BCr_angewandte_Wissenschaften_M%C3%BCnchen_logo.svg" alt="HM" width="160"/>
<img src="docs/assets/bmftr.svg" alt="BMFTR" width="200"/>
</div>

## License

Raytrax is released under the MIT License. See [LICENSE](LICENSE.md) for details.