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

# Examples

## Resize

The `INN.ResizeFeatures` method will simple abandon some features. When doing inverse, the abandoned information will be replaced by sampled number. 

```python
import INN
import torch

model = INN.ResizeFeatures(feature_in=3, feature_out=1)

x = torch.Tensor([[1,2,3],
                  [4,5,6],
                  [7,8,9]])

y, logp, logdet = model(x)
print(y)

x_hat = model.inverse(y)
print(x_hat)
```

Output:

```
# y = model(x)
tensor([[1.],
        [4.],
        [7.]])
# x_hat = model.inverse(y)
tensor([[ 1.0000,  1.5800, -0.6237],
        [ 4.0000,  0.5238,  0.3988],
        [ 7.0000,  1.0111, -0.0900]])
```

# License

This work is licensed under a GNU General Public License v3.0
