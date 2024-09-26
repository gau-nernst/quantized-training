# https://github.com/facebookresearch/schedule_free/blob/5afd85afb695965c8175b0950344a29973391862/schedulefree/adamw_schedulefree_reference.py

import torch
from torch import Tensor
from torch.optim.optimizer import ParamsT
from torchao.prototype.low_bit_optim.subclass_8bit import OptimState8bit


class AdamWScheduleFree(torch.optim.Optimizer):
    """This optimizer requires that .train() and .eval() be called before the
    beginning of training and evaluation respectively. The optimizer should
    also be placed in eval mode when saving checkpoints.
    """

    def __init__(
        self,
        params: ParamsT,
        lr: float = 0.0025,
        betas: tuple[float, float] = (0.9, 0.999),
        eps: float = 1e-8,
        weight_decay: float = 0,
        warmup_steps: int = 0,
        r: float = 0,
        weight_lr_power: float = 2,
    ):
        defaults = dict(
            lr=lr,
            betas=betas,
            eps=eps,
            r=r,
            warmup_steps=warmup_steps,
            weight_lr_power=weight_lr_power,
            weight_decay=weight_decay,
        )
        super().__init__(params, defaults)

    @torch.no_grad()
    def eval(self):
        for group in self.param_groups:
            for p in group["params"]:
                state = self.state[p]
                if len(state) == 0 or not state["train_mode"]:
                    continue

                p.lerp_(state["z"], 1 - 1 / group["betas"][0])
                state["train_mode"] = False

    @torch.no_grad()
    def train(self):
        for group in self.param_groups:
            for p in group["params"]:
                state = self.state[p]
                if len(state) == 0 or state["train_mode"]:
                    continue

                p.lerp_(state["z"], 1 - group["betas"][0])
                state["train_mode"] = True

    def _new_exp_avg_sq(self, p: Tensor):
        return torch.zeros_like(p)

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        with torch._dynamo.utils.disable_cache_limit():
            for group in self.param_groups:
                for p in group["params"]:
                    if p.grad is None:
                        continue

                    state = self.state[p]
                    if len(state) == 0:  # state init
                        state["train_mode"] = True
                        state["step"] = torch.tensor(0.0)
                        state["lr_max"] = torch.tensor(-1.0)
                        state["weight_sum"] = torch.tensor(0.0)
                        state["z"] = p.clone()
                        state["exp_avg_sq"] = self._new_exp_avg_sq(p)

                    state["step"] += 1
                    torch.compile(schedule_free_adamw, fullgraph=True, dynamic=False)(
                        p,
                        p.grad,
                        state["step"],
                        state["z"],
                        state["exp_avg_sq"],
                        state["lr_max"],
                        state["weight_sum"],
                        group["lr"],
                        group["betas"][0],
                        group["betas"][1],
                        group["weight_decay"],
                        group["eps"],
                        group["r"],
                        group["warmup_steps"],
                        group["weight_lr_power"],
                    )

        return loss


class AdamWScheduleFree8bit(AdamWScheduleFree):
    def _new_exp_avg_sq(self, p: Tensor):
        if p.numel() >= 4096 and p.numel() % 256 == 0:
            return OptimState8bit.zeros(p.shape, signed=False, device=p.device)
        else:
            return torch.zeros_like(p)


def schedule_free_adamw(
    p: Tensor,
    grad: Tensor,
    step: Tensor,
    z: Tensor,  # replace momentum
    exp_avg_sq: Tensor,
    lr_max: Tensor,
    weight_sum: Tensor,
    # hparams
    lr: float,
    beta1: float,
    beta2: float,
    weight_decay: float,
    eps: float,
    r: float,
    warmup_steps: int,
    weight_lr_power: float,
) -> None:
    sched = (step / warmup_steps).clip(max=1.0)
    bias_correction2 = 1 - beta2**step

    lr = lr * sched * bias_correction2**0.5
    torch.maximum(lr_max, lr, out=lr_max)

    weight = (step**r) * (lr_max**weight_lr_power)
    weight_sum += weight
    ckp1 = weight / weight_sum

    new_exp_avg_sq = exp_avg_sq.lerp(grad.square(), 1 - beta2)
    exp_avg_sq.copy_(new_exp_avg_sq)

    denom = new_exp_avg_sq.sqrt() + eps
    grad_normalized = weight_decay * p + grad / denom

    p.copy_(p.lerp(z, ckp1) + grad_normalized * lr * (beta1 * (1 - ckp1) - 1))
    z.add_(-lr * grad_normalized)


if __name__ == "__main__":
    import copy

    import schedulefree

    model_ref = torch.nn.Sequential(
        torch.nn.Linear(128, 128),
        torch.nn.GELU(),
        torch.nn.Linear(128, 128),
    ).cuda()
    model = copy.deepcopy(model_ref)

    optim_ref = schedulefree.AdamWScheduleFree(model_ref.parameters(), warmup_steps=2)
    optim = AdamWScheduleFree(model.parameters(), warmup_steps=2)

    for _ in range(10):
        inputs = torch.randn(32, 128, device="cuda")

        model_ref(inputs).sum().backward()
        optim_ref.step()
        optim_ref.zero_grad()

        model(inputs).sum().backward()
        optim.step()
        optim.zero_grad()

        for p, p_ref in zip(model.parameters(), model_ref.parameters()):
            torch.testing.assert_close(p, p_ref)

    optim_ref.eval()
    optim.eval()
    for p, p_ref in zip(model.parameters(), model_ref.parameters()):
        torch.testing.assert_close(p, p_ref)

    optim_ref.train()
    optim.train()
    for p, p_ref in zip(model.parameters(), model_ref.parameters()):
        torch.testing.assert_close(p, p_ref)
