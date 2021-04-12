# iResNetLab
A simple unofficial iResNet library that intend to make iResNet easy to use.

# Analogy to `torch.nn`

| pytorch                      | iResNetLab                     |
| ---------------------------- | ------------------------------ |
| `nn.Linear(dim_in, dim_out)` | `iResNet.FCN(dim_in, dim_out)` |
| `nn.Sequential`              | `iResNet.Sequential`           |

## Fully connected layers

Define a fully connected i-ResNet:

```python
model = iResNet.FCN(2, 2)
model.train()
```

Forward computing:

```python
# input data
x = torch.Tensor([[1,2],
                  [3,4]])
x.requires_grad = True # x must requires gradient

# forward
y, p, logdet = model(x)
# output:
>>> y = tensor([[1.3179, 2.2021],
>>>             [3.3864, 4.2015]], grad_fn=<SliceBackward>)
>>> p = tensor([0., 0.], grad_fn=<AddBackward0>)
>>> logdet = tensor([0.0153, 0.1178], grad_fn=<AddBackward0>)
```

Inverse process:

```python
# inverse
model.inverse(y.detach())
# output:
>>> output:
>>> tensor([[1., 2.],
>>>         [3., 4.]])
```

The `iResNet.FCN` provides a i-ResNet block that has the form of `model(x, log_p0, log_det_J0) --> y, log_p, log_det_J`.
The input not only have the feature `x`, but also have the `log_p0` and `log_det_J0`.

The `log_p0` is the log probability inherited from previous abandoned features. 
The `log_det_J` is the log(det J) from previous layers. They both be 0 by default.

## Sequential

Defining a sequential of FCN i-ResNet:

```python
model = iResNet.Sequential(iResNet.FCN(2, 2),
                           iResNet.FCN(2, 2),
                           iResNet.FCN(2, 2))
model.train()
```

Forward and inverse process:

```python
# input data
x = torch.Tensor([[1,2],
                  [3,4]])
x.requires_grad = True
y, p, logdet = model(x)

model.inverse(y.detach())
# output:
>>> output:
>>> tensor([[1., 2.],
>>>         [3., 4.]])
```

The modules for `iResNet.Sequential` must have this form: `model(x, log_p0, log_det_J0) --> y, log_p, log_det_J`. 
It must contains following methods:

1. `self.inverse(y) --> x`: a inverse method
2. `self.forward(x, log_p0, log_det_J0) --> y, log_p, log_det_J`: a forward method

## 1D Convolutional Network

> working

## 2D Convolutional Network

> working