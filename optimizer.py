import torch
from dataclasses import dataclass

@dataclass
class OptimConfig:
    max_lr: float = 1e-4
    max_wd: float = 0.01 # divide by
    adamw_beta1: float = 0.9
    adamw_beta2: float = 0.95
    adamw_eps: float = 1e-8

class Optimizer:

    def __init__(self, config, model):
        self.model = model

        self.max_lr = config.max_lr
        self.max_wd = config.max_wd
        self.adamw_beta1 = config.adamw_beta1
        self.adamw_beta2 = config.adamw_beta2
        self.adamw_eps = config.adamw_eps

        self.adamw_m = {}
        self.adamw_v = {}

        for name, param in self.model.named_parameters():
            self.adamw_m[name] = torch.zeros_like(param)
            self.adamw_v[name] = torch.zeros_like(param)

    @torch.no_grad()
    def step(self, step):

        if torch.compiler.is_dynamo_compiling():
            assert isinstance(step, torch.Tensor), "Ensure you pass step as a Tensor when using torch.compile."

        for name, param in self.model.named_parameters():
            if param.grad is not None:
                self._adamw_step(param, param.grad, self.adamw_m[name], self.adamw_v[name], step)

    def _adamw_step(self, x, g, m, v, step):

        # We use a different weight decay parameterization than PyTorch, that instead follows the AdamW paper.
        # See more discussion at https://fabian-sp.github.io/posts/2024/02/decoupling/

        # Modify these lines to implement weight decay and learning rate scheduling.
        wd = self.max_wd
        lr = self.max_lr

        beta1 = self.adamw_beta1
        beta2 = self.adamw_beta2

        m_new = beta1 * m + (1 - beta1) * g
        v_new = beta2 * v + (1 - beta2) * g.square()

        # This arithmetic can be avoided by reassociating it within the x-update below,
        # or within the recurrences above. We keep it separate for clarity.
        m_hat = m_new / (1 - beta1 ** (step + 1))
        v_hat = v_new / (1 - beta2 ** (step + 1))

        x_new = (1 - wd) * x - lr * m_hat / (v_hat.sqrt() + self.adamw_eps)

        x.copy_(x_new)
        m.copy_(m_new)
        v.copy_(v_new)
