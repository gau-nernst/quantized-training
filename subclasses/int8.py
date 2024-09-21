from typing import Literal, NamedTuple

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from kernels import scaled_mm

aten = torch.ops.aten


@torch.no_grad()
def quantize_int8(tensor: Tensor, stochastic_rounding: bool = False, *, dim: int = -1):
    # absmax symmetric quantization
    scale = tensor.abs().amax(dim, keepdim=True) / 127
    inv_scale = 1.0 / scale.float().clip(1e-12)
    tensor = tensor.float() * inv_scale  # slightly faster than divide directly

    if stochastic_rounding:
        tensor = (tensor + torch.rand_like(tensor)).floor()
    else:
        tensor = tensor.round()

    tensor = tensor.clip(-128, 127).to(torch.int8)
    return tensor, scale


class Int8QTConfig(NamedTuple):
    activation: Literal["none", "int8", "int8_sr"] = "none"


class Int8LinearWeight(Tensor):
    @staticmethod
    @torch._dynamo.disable
    def __new__(cls, int_data: Tensor, scale: Tensor, config: Int8QTConfig):
        return Tensor._make_wrapper_subclass(
            cls,
            int_data.shape,
            dtype=scale.dtype,
            device=int_data.device,
        )

    @torch._dynamo.disable
    def __init__(self, int_data: Tensor, scale: Tensor, config: Int8QTConfig):
        assert int_data.dtype is torch.int8
        assert int_data.ndim == 2
        assert scale.ndim == 2
        self.int_data = int_data
        self.scale = scale
        self.config = config

    def __tensor_flatten__(self):
        return ["int_data", "scale"], [self.config]

    @classmethod
    def __tensor_unflatten__(cls, tensor_data_dict, tensor_attributes, outer_size=None, outer_stride=None):
        return cls(tensor_data_dict["int_data"], tensor_data_dict["scale"], *tensor_attributes)

    @classmethod
    def from_float(cls, tensor: Tensor, *, config: Int8QTConfig = Int8QTConfig()):
        int_data, scale = quantize_int8(tensor.detach())
        out = cls(int_data, scale, config)
        out.requires_grad_(tensor.requires_grad)
        return out

    def dequantize(self):
        return self.int_data * self.scale.view(-1, 1)

    def __repr__(self):
        return (
            f"{self.__class__.__name__}(shape={tuple(self.shape)}, config={self.config}, "
            f"dtype={self.dtype}, device={self.device}, requires_grad={self.requires_grad})"
        )

    @classmethod
    def __torch_function__(cls, func, types, args=(), kwargs=None):
        kwargs = kwargs or dict()

        if func is F.linear:
            return _Int8Linear.apply(*args, **kwargs)

        with torch._C.DisableTorchFunctionSubclass():
            return func(*args, **kwargs)

    @classmethod
    def __torch_dispatch__(cls, func, types, args, kwargs):
        if func in (aten.detach.default, aten.clone.default):
            return cls(
                func(args[0].int_data, *args[1:], **kwargs),
                func(args[0].scale, *args[1:], **kwargs),
                args[0].config,
            )

        elif func in (aten._to_copy.default,):
            device = kwargs.get("device", None)
            dtype = kwargs.get("dtype", None)
            return cls(
                args[0].int_data.to(device=device),
                args[0].scale.to(device=device, dtype=dtype),
                args[0].config,
            )

        # to make training work with existing PyTorch optimizers, we return a normal tensor instead of Int8LinearWeight
        elif func is aten.zeros_like.default:
            dtype = kwargs.get("dtype", args[0].dtype)
            device = kwargs.get("device", args[0].device)
            return torch.zeros(args[0].shape, dtype=dtype, device=device)

        elif func in (aten.sub.Tensor, aten.mul.Tensor):
            args = [x.dequantize() if isinstance(x, cls) else x for x in args]
            return func(*args, **kwargs)

        elif func is aten.copy_.default:
            if isinstance(args[0], cls) and isinstance(args[1], cls):
                args[0].int_data.copy_(args[1].int_data)
                args[0].scale.copy_(args[1].scale)

            elif isinstance(args[0], cls):
                int_data, scale = quantize_int8(args[1], stochastic_rounding=True)
                args[0].int_data.copy_(int_data)
                args[0].scale.copy_(scale)

            else:
                args[0].copy_(args[1].dequantize())

            return args[0]

        # optim step
        elif func in (aten.addcdiv_.default, aten.add_.Tensor):
            original = args[0]
            out = func(args[0].dequantize(), *args[1:], **kwargs)
            return original.copy_(out)

        raise NotImplementedError(f"{cls.__name__} dispatch: attempting to run {func}, this is not supported")


class _Int8Linear(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input: Tensor, weight: Int8LinearWeight, bias: Tensor | None = None):
        ctx.save_for_backward(input, weight)
        ctx.bias = bias is not None

        if weight.config.activation == "none":
            # weight-only quantization
            # NOTE: we have to .T before .to(input.dtype) for torch.compile() mixed matmul to work
            out = (input @ weight.int_data.T.to(input.dtype)) * weight.scale.T

        else:
            # dynamic quantization
            batch_dims = input.shape[:-1]
            input = input.view(-1, weight.shape[1])
            input_int_data, input_scale = quantize_int8(input, weight.config.activation == "int8_sr")

            # optimization opportuntiy
            # we can fuse activation quantization into matmul too
            out = scaled_mm(input_int_data, weight.int_data.T, input_scale, weight.scale)
            out = out.view(*batch_dims, -1)

        out = out + bias if bias is not None else out
        return out

    @staticmethod
    def backward(ctx, grad_output):
        # generally we cannot do int8 matmul in backward because the scale is along the reduction dim
        # TODO: investigate how google/aqt does this
        input, weight = ctx.saved_tensors
        weight: Int8LinearWeight

        grad_input = (grad_output * weight.scale.T) @ weight.int_data.to(grad_output.dtype)

        grad_output = grad_output.view(-1, weight.shape[0])
        input = input.view(-1, weight.shape[1])

        # currently INT8 matmul is not fast for A.T @ B
        # thus, there is no point trying to do INT8 matmul for grad_weight
        grad_weight = grad_output.T @ input

        grad_bias = grad_output.sum(0) if ctx.bias else None
        return grad_input, grad_weight, grad_bias


def convert_int8_quantized_training(module: nn.Module, *, config: Int8QTConfig = Int8QTConfig()):
    if isinstance(module, nn.Linear):
        module.weight = nn.Parameter(
            Int8LinearWeight.from_float(module.weight, config=config),
            requires_grad=module.weight.requires_grad,
        )
    else:
        for m in module.children():
            convert_int8_quantized_training(m, config=config)
    return module
