import torch


@torch.jit.script
def _gelu_forward(x: torch.Tensor) -> torch.Tensor:
    return x * 0.5 * (1.0 + torch.tanh(0.79788456 * x * (1 + 0.044715 * x * x)))


@torch.jit.script
def _gelu_backward(gradient: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
    tanh_out = torch.tanh(0.79788456 * x * (1 + 0.044715 * x * x))
    ff = 0.5 * x * ((1 - tanh_out * tanh_out) * (0.79788456 + 0.1070322243 * x * x)) + 0.5 * (1 + tanh_out)
    return ff * gradient


class _MathGELU(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input: torch.Tensor) -> torch.Tensor:
        ctx.save_for_backward(input)
        return _gelu_forward(input)

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor) -> torch.Tensor:
        (input,) = ctx.saved_tensors
        tmp = _gelu_backward(grad_output, input)
        return tmp


class MathGELU(torch.nn.Module):
    def forward(self, input: torch.Tensor) -> torch.Tensor:
        return _MathGELU.apply(input)
