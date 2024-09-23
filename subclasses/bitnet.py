from typing import Any, Optional, Tuple

import torch
import torch.nn.functional as F
import torch.utils._pytree as pytree
from torch import Tensor, nn

from kernels import scaled_mm

from .int8 import quantize_int8

aten = torch.ops.aten


class BitNetTrainingLinearWeight(Tensor):
    @staticmethod
    @torch._dynamo.disable
    def __new__(cls, data: Tensor):
        return Tensor._make_wrapper_subclass(
            cls,
            data.shape,
            dtype=data.dtype,
            device=data.device,
        )

    @torch._dynamo.disable
    def __init__(self, data: Tensor):
        self._data = data

    def __tensor_flatten__(self):
        return ["_data"], []

    @classmethod
    def __tensor_unflatten__(cls, tensor_data_dict, tensor_attributes, outer_size=None, outer_stride=None):
        return cls(tensor_data_dict["_data"], *tensor_attributes)

    def __repr__(self):
        return f"{self.__class__.__name__}(data={self._data})"

    @classmethod
    def __torch_function__(cls, func, types, args=(), kwargs=None):
        kwargs = kwargs or dict()

        if func is F.linear:
            return _BitNetTrainingLinear.apply(*args, **kwargs)

        with torch._C.DisableTorchFunctionSubclass():
            return func(*args, **kwargs)

    # adapated from FP8 implementation of WeightWithDynamicFloat8CastTensor
    @classmethod
    def __torch_dispatch__(cls, func, types, args, kwargs):
        out = func(
            *pytree.tree_map_only(cls, lambda x: x._data, args),
            **pytree.tree_map_only(cls, lambda x: x._data, kwargs),
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
            return pytree.tree_map_only(Tensor, lambda x: cls(x), out)
        else:
            # return new unwrapped object
            return out

    def fsdp_pre_all_gather(self, mesh):
        data = self._data
        return (data,), ()

    def fsdp_post_all_gather(
        self,
        all_gather_outputs: Tuple[Tensor, ...],
        metadata: Any,
        param_dtype: torch.dtype,
        *,
        out: Optional[Tensor] = None,
    ):
        (data,) = all_gather_outputs
        if out is not None:
            assert isinstance(out, BitNetTrainingLinearWeight)
            return
        return BitNetTrainingLinearWeight(data), all_gather_outputs


def quantize_bitnet_weight(w: Tensor, eps: float = 1e-5) -> Tensor:
    dtype = w.dtype
    w = w.float()
    scale = w.abs().mean()  # tensor-wise abs-mean. FP32
    w = w / scale.clip(eps)
    w = w.round().clip(-1, 1).to(torch.int8)
    return w, scale.to(dtype)


class _BitNetTrainingLinear(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input: Tensor, weight: BitNetTrainingLinearWeight, bias: Tensor | None = None):
        batch_dims = input.shape[:-1]
        input = input.view(-1, weight.shape[1])

        # https://github.com/microsoft/unilm/blob/master/bitnet/The-Era-of-1-bit-LLMs__Training_Tips_Code_FAQ.pdf
        # Figure 4
        input_i8, row_scale = quantize_int8(input, eps=1e-5)
        weight_i8, tensor_scale = quantize_bitnet_weight(weight._data)
        ctx.save_for_backward(input, weight_i8, tensor_scale)

        # use int8 tensor cores
        out = scaled_mm(input_i8.contiguous(), weight_i8.contiguous().T, row_scale, tensor_scale)
        out = out.view(*batch_dims, weight.shape[0])

        out = out + bias if bias is not None else out
        return out

    @staticmethod
    def backward(ctx, grad_output):
        input, weight_i8, tensor_scale = ctx.saved_tensors
        grad_input = grad_weight = grad_bias = None

        batch_dims = grad_output.shape[:-1]
        grad_output = grad_output.view(-1, weight_i8.shape[0])
        input = input.view(-1, weight_i8.shape[1])

        # NOTE: we can potentially speedup training by also quantizing the backward pass
        # to use INT8 tensor cores
        if ctx.needs_input_grad[0]:
            # mixed mm
            grad_input = scaled_mm(grad_output, weight_i8, tensor_scale, None)
            grad_input = grad_input.view(*batch_dims, weight_i8.shape[1])

        if ctx.needs_input_grad[1]:
            grad_weight = grad_output.T @ input

        if ctx.needs_input_grad[2]:
            grad_bias = grad_output.sum(0)

        return grad_input, grad_weight, grad_bias


# NOTE: this does not remove old RMSNorm layers and insert new ones
# users should do this on their own
def convert_bitnet(module: nn.Module):
    if isinstance(module, nn.Linear):
        module.weight = nn.Parameter(
            BitNetTrainingLinearWeight(module.weight.detach()),
            requires_grad=module.weight.requires_grad,
        )
    else:
        for m in module.children():
            convert_bitnet(m)
    return module
