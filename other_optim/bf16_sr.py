import torch
from torch import Tensor
from torch.optim.optimizer import Optimizer, ParamsT


class AdamWBF16SR(Optimizer):
    """AdamW with 3 key differences:
    1. Optimizer states are always BF16.
    2. Optimizer calculations are always in FP32.
    3. If weight is BF16, the updated FP32 weight (produced by FP32 calculations)
    will be stochastically rounded to BF16 before writing back to BF16 weight.
    """

    def __init__(
        self,
        params: ParamsT,
        lr: float = 1e-3,
        betas: tuple[float, float] = (0.9, 0.999),
        eps: float = 1e-8,
        weight_decay: float = 1e-2,
    ) -> None:
        defaults = dict(
            lr=torch.as_tensor(lr),
            betas=betas,
            eps=eps,
            weight_decay=weight_decay,
        )
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        # for a given model, the number of different argument combinations to single_param_adam() is fixed.
        # thus, it is safe to disable cache limit without the risk of always re-compiling.
        with torch._dynamo.utils.disable_cache_limit():
            for group in self.param_groups:
                for p in group["params"]:
                    if p.grad is None:
                        continue

                    state = self.state[p]

                    # State initialization. always use BF16
                    if len(state) == 0:
                        state["step"] = torch.tensor(0.0)
                        state["exp_avg"] = torch.zeros_like(p, dtype=torch.bfloat16)
                        state["exp_avg_sq"] = torch.zeros_like(p, dtype=torch.bfloat16)

                    state["step"] += 1

                    if not isinstance(group["lr"], Tensor):
                        raise RuntimeError(
                            "lr was changed to a non-Tensor object. If you want to update lr, please use "
                            "optim.param_groups[0]['lr'].fill_(new_lr)"
                        )

                    torch.compile(adamw, fullgraph=True, dynamic=False)(
                        p,
                        p.grad,
                        state["step"],
                        state["exp_avg"],
                        state["exp_avg_sq"],
                        group["lr"],
                        group["betas"][0],
                        group["betas"][1],
                        group["weight_decay"],
                        group["eps"],
                        p.dtype is torch.bfloat16,
                    )

        return loss


def adamw(
    p: Tensor,
    grad: Tensor,
    step: Tensor,
    exp_avg: Tensor,
    exp_avg_sq: Tensor,
    lr: Tensor,
    beta1: float,
    beta2: float,
    weight_decay: float,
    eps: float,
    bf16_sr: bool,
):
    bias_correction1 = 1 - beta1**step
    bias_correction2 = 1 - beta2**step

    # upcast everything to FP32 for internal calculations
    grad_fp32 = grad.float()
    new_exp_avg = exp_avg.float().lerp(grad_fp32, 1 - beta1)
    new_exp_avg_sq = exp_avg_sq.float().lerp(grad_fp32.square(), 1 - beta2)

    exp_avg.copy_(new_exp_avg)
    exp_avg_sq.copy_(new_exp_avg_sq)

    denom = new_exp_avg_sq.sqrt() / bias_correction2.sqrt() + eps
    numer = new_exp_avg / bias_correction1

    p_fp32 = p.float()
    new_p = p_fp32 - lr * weight_decay * p_fp32 - lr * numer / denom

    if bf16_sr:
        new_p_i32 = new_p.view(torch.int32)

        # get random 16 bits
        # this will only produce random 31 bits since PyTorch will not
        # produce negative numbers by default. this is fine since
        # we only need 16 bits.
        rand = torch.empty_like(new_p_i32).random_()
        rand = rand & 0xFFFF  # take the lower 16 bits

        # NOTE: we are using signed int addition here
        # is it the right thing to do? compared to unsigned int addition?
        new_p_i32 = new_p_i32 + rand
        new_p_i32 = new_p_i32 & 0xFFFF0000  # truncate the lower 16 bits
        new_p = new_p_i32.view(torch.float32)

    p.copy_(new_p)
