import torch
from dataclasses import dataclass

@dataclass
class OptimConfig:
    adamw_beta1: float = 0.9
    adamw_beta2: float = 0.95
    adamw_eps: float = 1e-8
    adamw_weight_decay: float = 0.01

class Optimizer:

    def __init__(self, config, model):
        self.model = model

        self.adamw_beta1 = config.adamw_beta1
        self.adamw_beta2 = config.adamw_beta2
        self.adamw_eps = config.adamw_eps
        self.adamw_weight_decay = config.adamw_weight_decay

        # Not currently exposed via OptimConfig:
        self.optim_dtype = torch.float32
        self.model_dtype = model.mc.dtype
        self.compute_device = next(model.parameters()).device

        # Per-parameter optimizer buffers. We allocate exactly the buffers
        # the configured kinds need and `.copy_` into them each step, so
        # tensor identities are stable across torch.compile invocations.
        self.adamw_m = {}
        self.adamw_v = {}

        for name, param in self.model.named_parameters():
            self.adamw_m[name] = torch.zeros(param.view(-1).shape, dtype=self.optim_dtype, device=self.compute_device)
            self.adamw_v[name] = torch.zeros(param.view(-1).shape, dtype=self.optim_dtype, device=self.compute_device)

        # Specialize step() for first step. (Equivalent to "bias correction").
        # Stored as a 0-d tensor to avoid compiling two separate functions.
        self.adamw_first_step = torch.ones((), dtype=torch.bool, device=self.compute_device)

    @torch.no_grad()
    def step(self, lr):

        for name, param in self.model.named_parameters():
            if param.grad is not None:
                x = param.view(-1)
                g = param.grad.view(-1)
                x_new = self._adamw_step(
                    x.to(self.optim_dtype),
                    g.to(self.optim_dtype),
                    name,
                    lr
                )
                x.copy_(x_new.to(self.model_dtype))

        self.adamw_first_step.fill_(False)

    def _adamw_step(self, x, g, param_name, lr):
        beta1 = self.adamw_beta1
        beta2 = self.adamw_beta2

        if self.adamw_first_step:
            m = g
            v = g.square()
        else:
            m = beta1 * self.adamw_m[param_name] + (1 - beta1) * g
            v = beta2 * self.adamw_v[param_name] + (1 - beta2) * g.square()

        x = x * (1 - lr * self.adamw_weight_decay)
        x = x - lr * m / (v.sqrt() + self.adamw_eps)

        self.adamw_m[param_name].copy_(m)
        self.adamw_v[param_name].copy_(v)

        return x
