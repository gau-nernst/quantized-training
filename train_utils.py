from functools import partial

import torch
from torch import nn

import bnb_optim
import optimizers
from subclasses import (
    Int8MixedPrecisionConfig,
    Int8QTConfig,
    convert_int4_quantized_training,
    convert_int8_mixed_precision,
    convert_int8_quantized_training,
)


def get_grad_norm(model: nn.Module):
    return sum(p.grad.square().sum().item() for p in model.parameters() if p.grad is not None) ** 0.5


def get_optim_cls(optim):
    return eval(optim, dict(torch=torch, optimizers=optimizers, bnb_optim=bnb_optim, partial=partial))


def quantize_model(model: nn.Module, weight: str, activation: str):
    # TODO: rethink this again...
    if weight == "int8_mixed_precision":
        assert activation == "none"
        convert_int8_mixed_precision(model, config=Int8MixedPrecisionConfig(forward=True, backward_input=True))
    elif weight == "int8":
        convert_int8_quantized_training(model, config=Int8QTConfig(activation))
    elif weight == "int4":
        assert activation == "none"
        convert_int4_quantized_training(model)
    elif weight != "none":
        raise ValueError(f"Unsupported {weight=}")


def print_model_stats(model: nn.Module):
    print(f"No. of trainable params: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")
    print(f"No. of non-trainable params: {sum(p.numel() for p in model.parameters() if not p.requires_grad):,}")
    print(f"No. of buffers: {sum(p.numel() for p in model.buffers()):,}")
