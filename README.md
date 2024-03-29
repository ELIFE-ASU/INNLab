![](./images/INNLab.png)

--------------------------------------------------------------------------------


[![](https://img.shields.io/static/v1?label=pytorch&message=≥1.8.0&color=yellow)](https://pytorch.org)

INNLab is a Python package that provides easy access of using Invertible Neural Networks (INNs). In this package, we included INN practice like NICE, RealNVP and Residual Flow. Other supporting blocks (resize, reshape, invertible batchnrom, invertible pixel shuffle, etc.) are also included for easier implementation.

> This package is used in this paper: Yanbo Zhang, and Sara Imari Walker. "A Relational Macrostate Theory Guides Artificial Intelligence to Learn Macro and Design Micro." arXiv preprint arXiv:2210.07374 (2022). https://arxiv.org/abs/2210.07374

# Install

In the project folder, run the following commands step-by-step:

```bash
git clone https://github.com/ELIFE-ASU/INNLab
cd INNLab/
python setup.py install
```

The package requires PyTorch >= 1.8.0. If it is lower than it, you will not be able to use `INN.PixelShuffle2d`.

# Documentation

See [INNLab Wiki - Home](https://github.com/ELIFE-ASU/INNLab/wiki)

INNLab using the simular format as PyTorch, so users can getting start with easier. See documents here: [Analogy to PyTorch](https://github.com/ELIFE-ASU/INNLab/wiki/Analogy-to-PyTorch)

# License

This work is licensed under a GNU General Public License v3.0
