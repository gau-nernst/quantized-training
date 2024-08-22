import math
from typing import Literal, NamedTuple

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from kernels import scaled_int8_mm_bf16

aten = torch.ops.aten


@torch.no_grad()
def quantize_int8(tensor: Tensor, stochastic_rounding: bool = False, *, dim: int = -1):
    original_dtype = tensor.dtype
    tensor = tensor.float()

    # absmax symmetric quantization
    scale = tensor.abs().amax(dim, keepdim=True) / 127
    tensor = tensor / scale.clip(1e-12)

    if stochastic_rounding:
        tensor = (tensor + torch.rand_like(tensor)).floor()
    else:
        tensor = tensor.round()

    tensor = tensor.clip(-128, 127).to(torch.int8)
    return tensor, scale.to(original_dtype)


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
            out = scaled_int8_mm_bf16(input_int_data, weight.int_data.T, input_scale.view(-1), weight.scale.view(-1))
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


def quantize_linear_weight_int8(module: nn.Module, *, config: Int8QTConfig = Int8QTConfig()):
    if isinstance(module, nn.Linear):
        module.weight = nn.Parameter(
            Int8LinearWeight.from_float(module.weight, config=config),
            requires_grad=module.weight.requires_grad,
        )
    else:
        for m in module.children():
            quantize_linear_weight_int8(m, config=config)
    return module


class Int4LinearWeight(Tensor):
    @staticmethod
    @torch._dynamo.disable
    def __new__(cls, int_data, scale, zero_point, shape):
        return Tensor._make_wrapper_subclass(
            cls,
            shape,
            dtype=scale.dtype,
            device=int_data.device,
        )

    @torch._dynamo.disable
    def __init__(self, int_data, scale, zero_point, shape):
        assert int_data.dtype is torch.uint8
        assert math.prod(shape) == int_data.numel() * 2
        self.int_data = int_data
        self.scale = scale
        self.zero_point = zero_point
        self.group_size = int_data.numel() * 2 // zero_point.numel()

    def __tensor_flatten__(self):
        return ["int_data", "scale", "zero_point"], [self.shape]

    @classmethod
    def __tensor_unflatten__(cls, tensor_data_dict, tensor_attributes, outer_size=None, outer_stride=None):
        return cls(
            tensor_data_dict["int_data"],
            tensor_data_dict["scale"],
            tensor_data_dict["zero_point"],
            *tensor_attributes,
        )

    @staticmethod
    def quantize(tensor: Tensor, group_size: int, stochastic_rounding: bool = False):
        original_dtype = tensor.dtype
        tensor = tensor.float()
        tensor = tensor.view(-1, group_size)

        # asymmetric quantization
        # x_fp32 = zero_point + x_uint4 * scale, where x in [0, 15]
        # x_uint4 = (x_fp32 - zero_point) / scale
        zero_point = tensor.amin(-1)
        tensor = tensor - zero_point.view(-1, 1)
        scale = tensor.amax(-1) / 15
        tensor = tensor / scale.clip(1e-12).view(-1, 1)

        if stochastic_rounding:
            # floor is not required since .to(torch.uint8) always truncate for positive numbers
            tensor = tensor + torch.rand_like(tensor)
        else:
            tensor = tensor.round()

        tensor = tensor.clip(0, 15).to(torch.uint8)
        tensor = (tensor[:, ::2] << 4) | tensor[:, 1::2]
        return tensor, scale.to(original_dtype), zero_point.to(original_dtype)

    @classmethod
    def from_float(cls, tensor: Tensor, group_size: int = 32):
        int_data, scale, zero_point = cls.quantize(tensor.detach(), group_size)
        out = cls(int_data, scale, zero_point, tensor.shape)
        out.requires_grad_(tensor.requires_grad)
        return out

    def dequantize(self):
        x_uint4 = torch.stack([(self.int_data >> 4), self.int_data & 0b1111], dim=-1).view(-1, self.group_size)
        x_float = self.zero_point.view(-1, 1) + x_uint4 * self.scale.view(-1, 1)
        return x_float.view(self.shape)

    def __repr__(self):
        return (
            f"{self.__class__.__name__}(shape={tuple(self.shape)}, dtype={self.dtype}, device={self.device}, "
            f"requires_grad={self.requires_grad})"
        )

    @classmethod
    def __torch_function__(cls, func, types, args=(), kwargs=None):
        kwargs = kwargs or dict()

        if func is F.linear:
            return int4_weight_only_linear(*args)

        with torch._C.DisableTorchFunctionSubclass():
            return func(*args, **kwargs)

    @classmethod
    def __torch_dispatch__(cls, func, types, args, kwargs):
        if func is aten.detach.default:
            return cls(args[0].int_data, args[0].scale, args[0].zero_point, args[0].shape)

        elif func is aten.clone.default:
            return cls(
                args[0].int_data.clone(),
                args[0].scale.clone(),
                args[0].zero_point.clone(),
                args[0].shape,
            )

        # to make training work with existing PyTorch optimizers, we return a normal tensor instead of Int4LinearWeight
        elif func is aten.zeros_like.default:
            dtype = kwargs.get("dtype", args[0].dtype)
            device = kwargs.get("device", args[0].device)
            return torch.zeros(args[0].shape, dtype=dtype, device=device)

        # optim step
        elif func is aten.addcdiv_.default:
            output = torch.addcdiv(args[0].dequantize(), *args[1:], **kwargs)
            int_data, scale, zero_point = cls.quantize(output, args[0].group_size, stochastic_rounding=True)
            args[0].int_data.copy_(int_data)
            args[0].scale.copy_(scale)
            args[0].zero_point.copy_(zero_point)
            return args[0]

        elif func is aten.sub.Tensor:
            return func(args[0].dequantize(), *args[1:], **kwargs)

        elif func is aten.copy_.default:
            # not sure why torchao.prototype.low_bit_optim.Adam8bit requires this
            if isinstance(args[0], cls) and isinstance(args[1], cls):
                assert args[0].group_size == args[1].group_size
                args[0].int_data.copy_(args[1].int_data)
                args[0].scale.copy_(args[1].scale)
                args[0].zero_point.copy_(args[1].zero_point)

            elif isinstance(args[0], cls):
                int_data, scale, zero_point = cls.quantize(args[1], args[0].group_size, stochastic_rounding=True)
                args[0].int_data.copy_(int_data)
                args[0].scale.copy_(scale)
                args[0].zero_point.copy_(zero_point)

            else:
                args[0].copy_(args[1].dequantize())

            return args[0]

        raise NotImplementedError(f"{cls.__name__} dispatch: attempting to run {func}, this is not supported")


class Int4WeightOnlyLinear(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input: Tensor, weight: Int4LinearWeight, bias: Tensor | None = None):
        ctx.save_for_backward(input, weight)
        ctx.bias = bias is not None
        return F.linear(input, weight.dequantize(), bias)

    @staticmethod
    def backward(ctx, grad_output):
        input, weight = ctx.saved_tensors

        dinput = grad_output @ weight.dequantize()
        dweight = grad_output.flatten(0, -2).T @ input.flatten(0, -2)
        dbias = grad_output.sum(0) if ctx.bias else None
        return dinput, dweight, dbias


int4_weight_only_linear = Int4WeightOnlyLinear.apply


def quantize_linear_weight_int4(module: nn.Module, group_size: int = 32):
    if isinstance(module, nn.Linear):
        module.weight = nn.Parameter(
            Int4LinearWeight.from_float(module.weight, group_size=group_size),
            requires_grad=module.weight.requires_grad,
        )
    else:
        for m in module.children():
            quantize_linear_weight_int4(m)
    return module
