![](./images/INNLab.png)

--------------------------------------------------------------------------------


![](https://img.shields.io/static/v1?label=pytorch&message=≥1.6&color=yellow)

INNLab is a Python package that provides easy access of using Invertible Neural Networks (INNs). In this package, we included following INN practices:

* NICE
* RealNVP
* Residual Flow

* Other supporting blocks (resize, reshape, invertible batchnrom, invertible pixel shuffle, etc.)

We will not only providing strictly invertible modules, but also including some semi-invertible modules:

* (developing) Noisy kernels: for increasing the representation power
* (developing) InfoGAN: the network **not restricted at all**, and you can control the distribution of the embedding. However, invertibility is not guaranteed. Sampling may also have problem of mode collapse.

# Analogy to PyTorch

INNLab using the simular format as PyTorch, so users can getting start with easier. See documents here: [Analogy to PyTorch](https://github.com/ELIFE-ASU/INNLab/wiki/Analogy-to-PyTorch)

# Install

In the project folder, run:

```bash
python setup.py install
```

The package requires PyTorch >= 1.8.0. If it is lower than it, you will not be able to use `INN.PixelShuffle2d`.

# Documentation

See [INNLab Wiki - Home](https://github.com/ELIFE-ASU/INNLab/wiki)

# License

This work is licensed under a GNU General Public License v3.0
