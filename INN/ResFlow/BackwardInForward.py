import torch

def safe_detach(tensor):
    return tensor.detach().requires_grad_(tensor.requires_grad)

class MemoryEfficientLogDetEstimator(torch.autograd.Function):
    r''' Memory efficient logdet estimator by backward-in-forward
    from https://github.com/rtqichen/residual-flows/blob/master/lib/layers/iresblock.py
    by MIT license

    Reference: https://arxiv.org/abs/1906.02735
    '''
    
    @staticmethod
    def forward(ctx, estimator_fn, gnet, x, N, *g_params):
        with torch.enable_grad():
            x = x.detach().requires_grad_(True)
            g = gnet(x)
            ctx.g = g
            ctx.x = x
            logdetgrad = estimator_fn(x, g, N)

            # compute gradients to release memory
            grad_x, *grad_params = torch.autograd.grad(
                logdetgrad.sum(), (x,) + g_params, retain_graph=True, allow_unused=True
            )
            if grad_x is None:
                grad_x = torch.zeros_like(x)
            ctx.save_for_backward(grad_x, *g_params, *grad_params)

        return safe_detach(g), safe_detach(logdetgrad)

    @staticmethod
    def backward(ctx, grad_g, grad_logdetgrad):

        with torch.enable_grad():
            grad_x, *params_and_grad = ctx.saved_tensors
            g, x = ctx.g, ctx.x

            # Precomputed gradients.
            g_params = params_and_grad[:len(params_and_grad) // 2]
            grad_params = params_and_grad[len(params_and_grad) // 2:]

            dg_x, *dg_params = torch.autograd.grad(g, [x] + g_params, grad_g, allow_unused=True)

        # Update based on gradient from logdetgrad.
        dL = grad_logdetgrad[0].detach()
        with torch.no_grad():
            grad_x.mul_(dL)
            grad_params = tuple([g.mul_(dL) if g is not None else None for g in grad_params])

        # Update based on gradient from g.
        with torch.no_grad():
            grad_x.add_(dg_x)
            grad_params = tuple([dg.add_(djac) if djac is not None else dg for dg, djac in zip(dg_params, grad_params)])

        return (None, None, grad_x, None) + grad_params