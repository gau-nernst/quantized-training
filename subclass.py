import torch
import torch.nn.functional as F
from torch import Tensor, nn


class Int8LinearWeight(Tensor):
    def __new__(cls, int_data, scale, requires_grad=False):
        return Tensor._make_wrapper_subclass(
            cls,
            int_data.shape,
            dtype=scale.dtype,
            device=int_data.device,
            requires_grad=requires_grad,
        )

    def __init__(self, int_data, scale, requires_grad=False):
        assert int_data.dtype is torch.int8
        self.int_data = int_data
        self.scale = scale

    def __tensor_flatten__(self):
        return ["int_data", "scale"], []

    @classmethod
    def __tensor_unflatten__(cls, tensor_data_dict, tensor_attributes, outer_size=None, outer_stride=None):
        return cls(tensor_data_dict["int_data"], tensor_data_dict["scale"], *tensor_attributes)

    @staticmethod
    def quantize(tensor: Tensor):
        # symmetric quantization
        scale = tensor.abs().amax(-1) / 127.5
        tensor = tensor / scale.view(-1, 1)

        # stochastic rounding
        tensor = (tensor + torch.rand_like(tensor)).clip(-128, 127).to(torch.int8)
        return tensor, scale

    @classmethod
    def from_float(cls, tensor: Tensor):
        int_data, scale = cls.quantize(tensor.detach())
        return cls(int_data, scale, requires_grad=tensor.requires_grad)

    def dequantize(self):
        return self.int_data * self.scale.view(-1, 1)

    def __repr__(self):
        return (
            f"{self.__class__.__name__}(shape={tuple(self.shape)}, dtype={self.dtype}, device={self.device}, "
            f"requires_grad={self.requires_grad})"
        )

    @classmethod
    def __torch_function__(cls, func, types, args=(), kwargs=None):
        kwargs = kwargs or dict()

        if func is F.linear:
            return fp_int8_linear(*args)

        with torch._C.DisableTorchFunctionSubclass():
            return func(*args, **kwargs)

    @classmethod
    def __torch_dispatch__(cls, func, types, args, kwargs):
        aten = torch.ops.aten

        if func is aten.detach.default:
            return cls(args[0].int_data, args[0].scale, requires_grad=False)

        elif func is aten.clone.default:
            return cls(args[0].int_data.clone(), args[0].scale.clone(), requires_grad=args[0].requires_grad)

        raise NotImplementedError(f"{cls.__name__} dispatch: attempting to run {func}, this is not supported")


class FpInt8Linear(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input: Tensor, weight: Int8LinearWeight, bias: Tensor | None = None):
        ctx.save_for_backward(input, weight)
        ctx.bias = bias is not None

        out = (input @ weight.int_data.to(input.dtype).T) * weight.scale
        out = out + bias if bias is not None else out
        return out

    @staticmethod
    def backward(ctx, grad_output):
        input, weight = ctx.saved_tensors

        dinput = (grad_output / weight.scale) @ weight.int_data.to(grad_output.dtype)
        dweight = grad_output.T @ input
        dbias = grad_output.sum(0) if ctx.bias else None
        return dinput, dweight, dbias


fp_int8_linear = FpInt8Linear.apply


def quantize_linear_weight(module: nn.Module):
    if isinstance(module, nn.Linear):
        module.weight = nn.Parameter(
            Int8LinearWeight.from_float(module.weight),
            requires_grad=module.weight.requires_grad,
        )
    else:
        for m in module.children():
            quantize_linear_weight(m)
    return module
