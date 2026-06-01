import torch
from dataclasses import dataclass

@dataclass
class OptimConfig:
    adamw_beta1: float = 0.9
    adamw_beta2: float = 0.999
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


        # Shared step counter for bias correction. Stored as a 0-d float
        # tensor (rather than a Python int) so that incrementing it inside
        # the compiled `step` doesn't force dynamo to recompile on the new
        # value, and so that `beta ** step` stays tensor.
        self.adamw_step = torch.zeros((), dtype=self.optim_dtype, device=self.compute_device)

    @torch.no_grad()
    def step(self, lr):

        # AdamW's bias correction uses a 1-based step count. Increment once
        # per `step()` call so all AdamW invocations below see the same value.
        self.adamw_step.add_(1)

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

    def _adamw_step(self, x, g, param_name, lr):
        beta1 = self.adamw_beta1
        beta2 = self.adamw_beta2

        m = beta1 * self.adamw_m[param_name] + (1 - beta1) * g
        v = beta2 * self.adamw_v[param_name] + (1 - beta2) * g.square()

        # step counter was updated above
        bc1 = 1 - beta1 ** self.adamw_step
        bc2 = 1 - beta2 ** self.adamw_step

        m_hat = m / bc1
        v_hat = v / bc2

        x = x * (1 - lr * self.adamw_weight_decay)
        x = x - lr * m_hat / (v_hat.sqrt() + self.adamw_eps)

        self.adamw_m[param_name].copy_(m)
        self.adamw_v[param_name].copy_(v)

        return x