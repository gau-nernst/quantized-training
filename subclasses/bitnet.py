import torch
import torch.nn.functional as F
import torch.utils._pytree as pytree
from torch import Tensor, nn

from kernels import int8_mm_dequant

from .int8 import quantize_int8
from .mixed_precision import _dynamic_int8_mm

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


def quantize_bitnet_weight(w: Tensor):
    scale = w.abs().mean().clip(1e-5)  # tensor-wise abs-mean
    w = (w / scale).round().clip(-1, 1).to(torch.int8)
    return w, scale


class _BitNetTrainingLinear(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input: Tensor, weight: BitNetTrainingLinearWeight, bias: Tensor | None = None):
        batch_dims = input.shape[:-1]
        input = input.view(-1, weight.shape[1])

        input_i8, row_scale = quantize_int8(input)
        weight_i8, tensor_scale = quantize_bitnet_weight(weight._data)
        ctx.save_for_backward(input, weight_i8, tensor_scale)

        out = int8_mm_dequant(input_i8.contiguous(), weight_i8.T, row_scale, tensor_scale)
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

        # NOTE: original BitNet does not quantize backward pass
        if ctx.needs_input_grad[0]:
            grad_output_i8, row_scale = quantize_int8(grad_output)
            grad_input = int8_mm_dequant(grad_output_i8, weight_i8, row_scale, tensor_scale)
            grad_input = grad_input.view(*batch_dims, weight_i8.shape[1])

        if ctx.needs_input_grad[1]:
            grad_weight = _dynamic_int8_mm(input.T, grad_output, False).T

        if ctx.needs_input_grad[2]:
            grad_bias = grad_output.sum(0)

        return grad_input, grad_weight, grad_bias


# this is not complete. still need to remove other RMSNorm layers
def convert_bitnet(module: nn.Module, *, eps: float = 1e-5):
    for name, child in module.named_children():
        if isinstance(child, nn.Linear):
            child.weight = nn.Parameter(
                BitNetTrainingLinearWeight(child.weight.detach()),
                requires_grad=child.weight.requires_grad,
            )
            # insert RMSNorm in front
            norm = nn.RMSNorm(child.in_features, eps, device=child.weight.device, dtype=child.weight.dtype)
            setattr(module, name, nn.Sequential(norm, child))
        else:
            convert_bitnet(child, eps=eps)
    return module
