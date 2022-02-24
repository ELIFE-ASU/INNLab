import torch

def forward_test_normal(model, x):
    model.computing_p(True)
    y, logp, logdet = model(x)

def forward_test_not_compute_p(model, x):
    model.computing_p(False)
    y = model(x)

def inverse_test(model, x):
    model.computing_p(False)
    y = model(x)
    x_hat = model.inverse(y).detach()

    diff = torch.mean((x - x_hat) ** 2) ** 0.5
    assert diff < 1e-5

def forward_test(model, x):
    forward_test_normal(model, x)
    forward_test_not_compute_p(model, x)