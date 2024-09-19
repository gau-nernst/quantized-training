import math
from functools import partial

import bitsandbytes as bnb
import torch
import torchao.prototype.low_bit_optim as low_bit_optim
from torch import Tensor, nn

from subclasses import Int8QTConfig, MixedPrecisionConfig, convert_int8_quantized_training, convert_mixed_precision


def get_grad_norm(model: nn.Module):
    return sum(p.grad.square().sum().item() for p in model.parameters() if p.grad is not None) ** 0.5


def get_optimizer(optim: str, model: nn.Module, lr: float, weight_decay: float, **kwargs):
    allowed = dict(torch=torch, low_bit_optim=low_bit_optim, bnb=bnb, partial=partial)
    optim_cls = eval(optim, allowed)
    return optim_cls(model.parameters(), lr=lr, weight_decay=weight_decay, **kwargs)


def quantize_model(model: nn.Module, quantize: str | None, **kwargs):
    if quantize == "mixed_precision":
        config = MixedPrecisionConfig(**kwargs)
        print(f"Mixed precision with {config=}")
        convert_mixed_precision(model, config=config)

    elif quantize == "int8_quantized_training":
        config = Int8QTConfig(**kwargs)
        print(f"INT8 quantized training with {config=}")
        convert_int8_quantized_training(model, config=config)

    else:
        assert quantize is None


def print_model_stats(model: nn.Module):
    print(f"No. of trainable params: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")
    print(f"No. of non-trainable params: {sum(p.numel() for p in model.parameters() if not p.requires_grad):,}")
    print(f"No. of buffers: {sum(p.numel() for p in model.buffers()):,}")


class LRSchedule:
    def __init__(
        self,
        lr: float,
        n_steps: int,
        warmup: float = 0.0,
        decay: float = 0.0,
        decay_type: str = "linear",
    ) -> None:
        self.lr = lr
        self.t1 = int(n_steps * warmup)
        self.t2 = int(n_steps * (1 - decay))
        self.t3 = n_steps
        self.decay_type = decay_type
        assert self.t1 <= self.t2
        assert decay_type in ("linear", "cosine")

    def get_lr(self, step: int) -> float:
        if step < self.t1:
            return self.lr * step / self.t1
        if step < self.t2:
            return self.lr
        if step < self.t3:
            progress = (step - self.t2) / (self.t3 - self.t2)
            if self.decay_type == "linear":
                return self.lr * (1 - progress)
            elif self.decay_type == "cosine":
                return 0.5 * self.lr * (1 + math.cos(progress * math.pi))
        return 0.0

    def set_lr(self, step: int, optim: torch.optim.Optimizer):
        lr = self.get_lr(step)
        for param_group in optim.param_groups:
            if isinstance(param_group["lr"], Tensor):
                param_group["lr"].fill_(lr)
            else:
                param_group["lr"] = lr
