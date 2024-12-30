from typing import NamedTuple

import torch
import torch.nn.functional as F
import torch.utils._pytree as pytree
from torch import Tensor, nn

from kernels import scaled_int4_mm, scaled_mm

from .int8 import quantize_int8

aten = torch.ops.aten


class MixedPrecisionConfig(NamedTuple):
    output: bool = True
    grad_input: bool = True
    grad_weight: bool = True
    dtype: str = "int8"
    stochastic_rounding: bool = False


class MixedPrecisionLinearWeight(Tensor):
    @staticmethod
    @torch._dynamo.disable
    def __new__(cls, data: Tensor, config: MixedPrecisionConfig):
        return Tensor._make_wrapper_subclass(
            cls,
            data.shape,
            dtype=data.dtype,
            device=data.device,
        )

    @torch._dynamo.disable
    def __init__(self, data: Tensor, config: MixedPrecisionConfig):
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
    A_i8, row_scale = quantize_int8(A, sr, dim=1)
    B_t_i8, col_scale = quantize_int8(B.T, sr, dim=1)
    return scaled_mm(
        A_i8.contiguous(),
        B_t_i8.contiguous().T,
        row_scale.contiguous(),
        col_scale.T.contiguous(),
    )


def quantize_int4_rowwise_absmax(x: Tensor) -> Tensor:
    # scale = x.abs().amax(dim=1) / 7

    # slightly slower, but should be more accurate (does it matter?)
    pos_scale = x.relu().amax(dim=1) / 7
    neg_scale = x.neg().relu().amax(dim=1) / 8
    scale = torch.maximum(pos_scale, neg_scale)

    inv_scale = 1.0 / scale.float().clip(1e-12)
    x = x.float() * inv_scale.view(-1, 1)
    x = x.round().to(torch.int8)
    x = (x[:, ::2] << 4) | (x[:, 1::2] & 0xF)
    return x, scale


def _dynamic_int4_mm(A: Tensor, B: Tensor) -> Tensor:
    A_i4, row_scale = quantize_int4_rowwise_absmax(A)
    B_t_i4, col_scale = quantize_int4_rowwise_absmax(B.T)
    return scaled_int4_mm(
        A_i4.contiguous(),
        B_t_i4.contiguous().T,
        row_scale.contiguous(),
        col_scale.contiguous(),
    )


def _dynamic_mm(A, B, sr, dtype):
    if dtype == "int8":
        return _dynamic_int8_mm(A, B, sr)
    elif dtype == "int4":
        return _dynamic_int4_mm(A, B)
    else:
        raise ValueError


class _Int8MixedPrecisionLinear(torch.autograd.Function):
    @staticmethod
    def forward(input: Tensor, weight: MixedPrecisionLinearWeight, bias: Tensor | None = None):
        if weight.config.output:
            batch_dims = input.shape[:-1]
            input = input.view(-1, weight.shape[1])
            out = _dynamic_mm(input, weight._data.T, weight.config.stochastic_rounding, weight.config.dtype)
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
                grad_input = _dynamic_mm(grad_output, weight, ctx.config.stochastic_rounding, ctx.config.dtype)
            else:
                grad_input = grad_output @ weight
            grad_input = grad_input.view(*batch_dims, weight.shape[1])

        if ctx.needs_input_grad[1]:
            if ctx.config.grad_weight:
                # this is slightly faster
                grad_weight = _dynamic_mm(input.T, grad_output, ctx.config.stochastic_rounding, ctx.config.dtype).T
            else:
                grad_weight = grad_output.T @ input

        if ctx.needs_input_grad[2] and ctx.bias:
            grad_bias = grad_output.sum(0)

        return grad_input, grad_weight, grad_bias


def convert_mixed_precision(module: nn.Module, *, config: MixedPrecisionConfig = MixedPrecisionConfig()):
    if isinstance(module, nn.Linear):
        module.weight = nn.Parameter(
            MixedPrecisionLinearWeight(module.weight.detach(), config=config),
            requires_grad=module.weight.requires_grad,
        )
    else:
        for m in module.children():
            convert_mixed_precision(m, config=config)
    return module
