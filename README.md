# iResNetLab
A simple unofficial iResNet library that intend to make iResNet easy to use.

# Analogy to `torch.nn`

## Fully connected layers
`nn.Linear(dim_in, dim_out)` --> `iResNet.FCN(dim_in, dim_out)`

The `iResNet.FCN` provides a i-ResNet block that has the form of `model(x, log_p0, log_det_J0) --> y, log_p, log_det_J`.
The input not only have the feature `x`, but also have the `log_p0` and `log_det_J0`.

The `log_p0` is the log probability inherited from previous abandoned features. 
The `log_det_J` is the log(det J) from previous layers. They both be 0 by default.

## Sequential
`nn.Sequential` --> `iResNet.Sequential`

The modules for `iResNet.Sequential` must have this form: `model(x, log_p0, log_det_J0) --> y, log_p, log_det_J`. 
It must contains following methods:

1. `self.inverse(y) --> x`: a inverse method
2. `self.forward(x, log_p0, log_det_J0) --> y, log_p, log_det_J`: a forward method