from torch import nn, Tensor
from torch.autograd import Function

class _GradientReversalFunction(Function):

    @staticmethod
    def forward(ctx, x: Tensor, lambda_: float) -> Tensor:
        ctx.lambda_ = lambda_
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output: Tensor) -> tuple[Tensor, None]:
        return -ctx.lambda_ * grad_output, None

class GradientReversalLayer(nn.Module):

    def forward(self, x: Tensor, lambda_: float) -> Tensor:
        return _GradientReversalFunction.apply(x, lambda_)