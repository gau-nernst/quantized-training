import math

import torch
import torch.nn.functional as F
from torch import Tensor, nn

aten = torch.ops.aten


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


def convert_int4_quantized_training(module: nn.Module, group_size: int = 32):
    if isinstance(module, nn.Linear):
        module.weight = nn.Parameter(
            Int4LinearWeight.from_float(module.weight, group_size=group_size),
            requires_grad=module.weight.requires_grad,
        )
    else:
        for m in module.children():
            convert_int4_quantized_training(m, group_size)
    return module
