from typing import NamedTuple

import torch
import torch.nn.functional as F
import torch.utils._pytree as pytree
from torch import Tensor, nn

from kernels import int8_mm_dequant

from .int8 import quantize_int8

aten = torch.ops.aten


class Int8MixedPrecisionConfig(NamedTuple):
    output: bool = True
    grad_input: bool = True
    grad_weight: bool = True
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
        return f"{self.__class__.__name__}(data={self._data}, config={self.config})"

    @classmethod
    def __torch_function__(cls, func, types, args=(), kwargs=None):
        kwargs = kwargs or dict()

        if func is F.linear:
            return _Int8MixedPrecisionLinear.apply(*args, **kwargs)

        with torch._C.DisableTorchFunctionSubclass():
            return func(*args, **kwargs)

    # adapated from FP8 implementation of WeightWithDynamicFloat8CastTensor
    @classmethod
    def __torch_dispatch__(cls, func, types, args, kwargs):
        config = None

        def unwrap(x: cls):
            nonlocal config
            if config is None:
                config = x.config
            else:
                assert x.config == config
            return x._data

        out = func(
            *pytree.tree_map_only(cls, unwrap, args),
            **pytree.tree_map_only(cls, unwrap, kwargs),
        )

        if func is aten.copy_.default:
            # return original object
            return args[0]
        elif func in {
            aten.t.default,
            aten.detach.default,
            aten.empty_like.default,
            aten.new_zeros.default,
            aten.slice.Tensor,
            aten.view.default,
            aten.as_strided.default,
            aten._to_copy.default,
            aten._pin_memory.default,
            aten.split.Tensor,
            aten.clone.default,
        }:
            # return new wrapped object
            return pytree.tree_map_only(Tensor, lambda x: cls(x, config), out)
        else:
            # return new unwrapped object
            return out


def _dynamic_int8_mm(A: Tensor, B: Tensor, sr: bool) -> Tensor:
    A_i8, A_scale_rowwise = quantize_int8(A, sr, dim=1)
    B_t_i8, B_scale_colwise = quantize_int8(B.T, sr, dim=1)
    return int8_mm_dequant(
        A_i8.contiguous(),
        B_t_i8.contiguous().T,
        A_scale_rowwise.contiguous(),
        B_scale_colwise.contiguous(),
    )


class _Int8MixedPrecisionLinear(torch.autograd.Function):
    @staticmethod
    def forward(input: Tensor, weight: Int8MixedPrecisionLinearWeight, bias: Tensor | None = None):
        if weight.config.output:
            batch_dims = input.shape[:-1]
            input = input.view(-1, weight.shape[1])
            out = _dynamic_int8_mm(input, weight._data.T, weight.config.stochastic_rounding)
            out = out.view(*batch_dims, weight.shape[0])
        else:
            out = input @ weight.T

        out = out + bias if bias is not None else out
        return out

    @staticmethod
    def setup_context(ctx, inputs, output):
        input, weight, bias = inputs
        ctx.config = weight.config
        ctx.save_for_backward(input, weight._data)
        ctx.bias = bias is not None

    @staticmethod
    def backward(ctx, grad_output):
        input, weight = ctx.saved_tensors
        grad_input = grad_weight = grad_bias = None

        batch_dims = grad_output.shape[:-1]
        grad_output = grad_output.view(-1, weight.shape[0])
        input = input.view(-1, weight.shape[1])

        if ctx.needs_input_grad[0]:
            if ctx.config.grad_input:
                grad_input = _dynamic_int8_mm(grad_output, weight, ctx.config.stochastic_rounding)
            else:
                grad_input = grad_output @ weight
            grad_input = grad_input.view(*batch_dims, weight.shape[1])

        if ctx.needs_input_grad[1]:
            if ctx.config.grad_weight:
                # this is slightly faster
                grad_weight = _dynamic_int8_mm(input.T, grad_output, ctx.config.stochastic_rounding).T
            else:
                grad_weight = grad_output.T @ input

        if ctx.needs_input_grad[2] and ctx.bias:
            grad_bias = grad_output.sum(0)

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
