from functools import partial

import torch
from torch import nn

import bnb_optim
import optimizers
from subclass import quantize_linear_weight_int4, quantize_linear_weight_int8


def get_grad_norm(model: nn.Module):
    return sum(p.grad.square().sum().item() for p in model.parameters() if p.grad is not None) ** 0.5


def get_optim_cls(optim):
    return eval(optim, dict(torch=torch, optimizers=optimizers, bnb_optim=bnb_optim, partial=partial))


def quantize_model(model: nn.Module, quantization: str | None = None):
    if quantization == "int8":
        quantize_linear_weight_int8(model)
    elif quantization == "int8_quantize_activation":
        quantize_linear_weight_int8(model, quantize_activation=True)
    elif quantization == "int4":
        quantize_linear_weight_int4(model)
    elif quantization is not None:
        raise ValueError(f"Unsupported {quantization=}")


def print_model_stats(model: nn.Module):
    print(f"No. of trainable params: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")
    print(f"No. of non-trainable params: {sum(p.numel() for p in model.parameters() if not p.requires_grad):,}")
    print(f"No. of buffers: {sum(p.numel() for p in model.buffers()):,}")
