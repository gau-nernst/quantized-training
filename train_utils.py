from functools import partial

import bitsandbytes as bnb
import torch
import torchao
from torch import nn

import bnb_optim
import optimizers
from subclasses import (
    Int8MixedPrecisionConfig,
    Int8QTConfig,
    convert_int8_mixed_precision,
    convert_int8_quantized_training,
)


def get_grad_norm(model: nn.Module):
    return sum(p.grad.square().sum().item() for p in model.parameters() if p.grad is not None) ** 0.5


def get_optim_cls(optim):
    allowed = dict(torch=torch, optimizers=optimizers, bnb_optim=bnb_optim, partial=partial, bnb=bnb, torchao=torchao)
    return eval(optim, allowed)


def quantize_model(model: nn.Module, int8_mixed_precision: dict | None, int8_quantized_training: dict | None):
    if int8_mixed_precision is not None:
        assert int8_quantized_training is None
        config = Int8MixedPrecisionConfig(**int8_mixed_precision)
        print(f"INT8 mixed precision with {config=}")
        convert_int8_mixed_precision(model, config=config)

    elif int8_quantized_training is not None:
        config = Int8QTConfig(**int8_quantized_training)
        print(f"INT8 quantized training with {config=}")
        convert_int8_quantized_training(model, config=config)


def print_model_stats(model: nn.Module):
    print(f"No. of trainable params: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")
    print(f"No. of non-trainable params: {sum(p.numel() for p in model.parameters() if not p.requires_grad):,}")
    print(f"No. of buffers: {sum(p.numel() for p in model.buffers()):,}")
