from typing import NamedTuple

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from kernels import scaled_int8_mm

from .int8 import quantize_int8

aten = torch.ops.aten


class Int8MixedPrecisionConfig(NamedTuple):
    forward: bool = False
    backward_grad_input: bool = False
    backward_grad_weight: bool = False
    stochastic_rounding: bool = False


class Int8MixedPrecisionLinearWeight(Tensor):
    @staticmethod
    @torch._dynamo.disable
    def __new__(cls, data: Tensor, config: Int8MixedPrecisionConfig):
        return Tensor._make_wrapper_subclass(
            cls,
            data.shape,
            dtype=data.dtype,
            device=data.device,
        )

    @torch._dynamo.disable
    def __init__(self, data: Tensor, config: Int8MixedPrecisionConfig):
        self._data = data
        self.config = config

    def __tensor_flatten__(self):
        return ["_data"], [self.config]

    @classmethod
    def __tensor_unflatten__(cls, tensor_data_dict, tensor_attributes, outer_size=None, outer_stride=None):
        return cls(tensor_data_dict["_data"], *tensor_attributes)

    def __repr__(self):
        return self._data.__repr__()

    @classmethod
    def __torch_function__(cls, func, types, args=(), kwargs=None):
        kwargs = kwargs or dict()

        if func is F.linear:
            return _Int8MixedPrecisionLinear.apply(*args, **kwargs)

        with torch._C.DisableTorchFunctionSubclass():
            return func(*args, **kwargs)

    @classmethod
    def __torch_dispatch__(cls, func, types, args, kwargs):
        if func in (aten.detach.default, aten.clone.default, aten._to_copy.default):
            return cls(func(args[0]._data, *args[1:], **kwargs), args[0].config)

        args = [x._data if isinstance(x, cls) else x for x in args]
        return func(*args, **kwargs)


class _Int8MixedPrecisionLinear(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input: Tensor, weight: Int8MixedPrecisionLinearWeight, bias: Tensor | None = None):
        ctx.save_for_backward(input, weight)
        ctx.bias = bias is not None
        sr = weight.config.stochastic_rounding

        if weight.config.forward:
            batch_dims = input.shape[:-1]
            input = input.view(-1, weight.shape[1])
            input_i8, input_scale = quantize_int8(input, sr, dim=1)
            weight_i8, weight_scale = quantize_int8(weight, sr, dim=1)
            out = scaled_int8_mm(input_i8, weight_i8.T, input_scale.view(-1), weight_scale.view(-1))
            out = out.view(*batch_dims, weight.shape[0])

        else:
            out = input @ weight.T

        out = out + bias if bias is not None else out
        return out

    @staticmethod
    def backward(ctx, grad_output):
        input, weight = ctx.saved_tensors
        weight: Int8MixedPrecisionLinearWeight
        sr = weight.config.stochastic_rounding

        batch_dims = grad_output.shape[:-1]
        grad_output = grad_output.view(-1, weight.shape[0])
        input = input.view(-1, weight.shape[1])

        if weight.config.backward_grad_input:
            grad_output_i8, grad_output_scale = quantize_int8(grad_output, sr, dim=1)
            weight_i8_t, weight_scale = quantize_int8(weight.T, sr, dim=1)
            grad_input = scaled_int8_mm(
                grad_output_i8, weight_i8_t.T, grad_output_scale.view(-1), weight_scale.view(-1)
            )

        else:
            grad_input = grad_output @ weight

        grad_input = grad_input.view(*batch_dims, weight.shape[1])

        if not weight.requires_grad:
            grad_weight = None

        elif weight.config.backward_grad_weight:
            grad_output_i8_t, grad_output_scale = quantize_int8(grad_output.T, sr, dim=1)
            input_i8_t, input_scale = quantize_int8(input.T, sr, dim=1)
            grad_weight = scaled_int8_mm(
                grad_output_i8_t, input_i8_t.T, grad_output_scale.view(-1), input_scale.view(-1)
            )

        else:
            grad_weight = grad_output.T @ input

        grad_bias = grad_output.sum(0) if ctx.bias else None
        return grad_input, grad_weight, grad_bias


def convert_int8_mixed_precision(module: nn.Module, *, config: Int8MixedPrecisionConfig = Int8MixedPrecisionConfig()):
    if isinstance(module, nn.Linear):
        module.weight = nn.Parameter(
            Int8MixedPrecisionLinearWeight(module.weight.detach(), config=config),
            requires_grad=module.weight.requires_grad,
        )
    else:
        for m in module.children():
            convert_int8_mixed_precision(m, config=config)
    return module
