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
    def quantize(tensor: Tensor, stochastic_rounding: bool = False):
        original_dtype = tensor.dtype
        tensor = tensor.float()

        # symmetric quantization
        scale = tensor.abs().amax(-1) / 127.5
        tensor = tensor / scale.clip(1e-12).view(-1, 1)

        if stochastic_rounding:
            # floor is required since .to(torch.int8) will convert 3.1 to 3 but -3.1 to -3
            tensor = (tensor + torch.rand_like(tensor)).floor()
        else:
            tensor = tensor.round()

        tensor = tensor.clip(-128, 127).to(torch.int8)
        return tensor, scale.to(original_dtype)

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
            return int8_weight_only_linear(*args)

        with torch._C.DisableTorchFunctionSubclass():
            return func(*args, **kwargs)

    @classmethod
    def __torch_dispatch__(cls, func, types, args, kwargs):
        aten = torch.ops.aten

        if func is aten.detach.default:
            return cls(args[0].int_data, args[0].scale, requires_grad=False)

        elif func is aten.clone.default:
            return cls(args[0].int_data.clone(), args[0].scale.clone(), requires_grad=args[0].requires_grad)

        # to make training work with existing PyTorch optimizers, we return a normal tensor instead of Int8LinearWeight
        elif func is aten.zeros_like.default:
            kwargs.pop("memory_format", None)
            return torch.zeros(args[0].shape, dtype=args[0].dtype, device=args[0].device, **kwargs)

        # optim step
        elif func is aten.addcdiv_.default:
            output = torch.addcdiv(args[0].dequantize(), *args[1:], **kwargs)
            int_data, scale = cls.quantize(output, stochastic_rounding=True)
            args[0].int_data.copy_(int_data)
            args[0].scale.copy_(scale)
            return args[0]

        elif func is aten.add_.Tensor:
            output = torch.add(args[0].dequantize(), *args[1:], **kwargs)
            int_data, scale = cls.quantize(output, stochastic_rounding=True)
            args[0].int_data.copy_(int_data)
            args[0].scale.copy_(scale)
            return args[0]

        # not sure why torchao.prototype.low_bit_optim.Adam8bit requires this
        elif func is aten.copy_.default:
            if isinstance(args[0], cls) and isinstance(args[1], cls):
                args[0].int_data.copy_(args[1].int_data)
                args[0].scale.copy_(args[1].scale)
                return args[0]

        raise NotImplementedError(f"{cls.__name__} dispatch: attempting to run {func}, this is not supported")


class Int8WeightOnlyLinear(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input: Tensor, weight: Int8LinearWeight, bias: Tensor | None = None):
        ctx.save_for_backward(input, weight)
        ctx.bias = bias is not None

        # NOTE: we have to .T before .to(input.dtype) for torch.compile() mixed matmul to work
        out = (input @ weight.int_data.T.to(input.dtype)) * weight.scale
        out = out + bias if bias is not None else out
        return out

    @staticmethod
    def backward(ctx, grad_output):
        input, weight = ctx.saved_tensors

        dinput = (grad_output * weight.scale) @ weight.int_data.to(grad_output.dtype)
        dweight = grad_output.flatten(0, -2).T @ input.flatten(0, -2)
        dbias = grad_output.sum(0) if ctx.bias else None
        return dinput, dweight, dbias


int8_weight_only_linear = Int8WeightOnlyLinear.apply


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
