# Copyright (c) Facebook, Inc. and its affiliates.
import math
from functools import lru_cache
import torch
from torch import nn
from torch.autograd import Function
from torch.autograd.function import once_differentiable
from torch.nn.modules.utils import _pair
from torchvision.ops import deform_conv2d

from detectron2 import _C

from .wrappers import _NewEmptyTensorOp


class _DeformConv(Function):
    @staticmethod
    def forward(
        ctx,
        input,
        offset,
        weight,
        stride=1,
        padding=0,
        dilation=1,
        groups=1,
        deformable_groups=1,
        im2col_step=64,
    ):
        if input is not None and input.dim() != 4:
            raise ValueError(
                "Expected 4D tensor as input, got {}D tensor instead.".format(input.dim())
            )
        ctx.stride = _pair(stride)
        ctx.padding = _pair(padding)
        ctx.dilation = _pair(dilation)
        ctx.groups = groups
        ctx.deformable_groups = deformable_groups
        ctx.im2col_step = im2col_step

        ctx.save_for_backward(input, offset, weight)

        output = input.new_empty(
            _DeformConv._output_size(input, weight, ctx.padding, ctx.dilation, ctx.stride)
        )

        ctx.bufs_ = [input.new_empty(0), input.new_empty(0)]  # columns, ones

        if not input.is_cuda:
            if deformable_groups != 1:
                raise NotImplementedError(
                    "Deformable Conv with deformable_groups != 1 is not supported on CPUs!"
                )
            return deform_conv2d(
                input, offset, weight, stride=stride, padding=padding, dilation=dilation
            )
        else:
            cur_im2col_step = _DeformConv._cal_im2col_step(input.shape[0], ctx.im2col_step)
            assert (input.shape[0] % cur_im2col_step) == 0, "im2col step must divide batchsize"

            _C.deform_conv_forward(
                input,
                weight,
                offset,
                output,
                ctx.bufs_[0],
                ctx.bufs_[1],
                weight.size(3),
                weight.size(2),
                ctx.stride[1],
                ctx.stride[0],
                ctx.padding[1],
                ctx.padding[0],
                ctx.dilation[1],
                ctx.dilation[0],
                ctx.groups,
                ctx.deformable_groups,
                cur_im2col_step,
            )
        return output

    @staticmethod
    @once_differentiable
    def backward(ctx, grad_output):
        input, offset, weight = ctx.saved_tensors

        grad_input = grad_offset = grad_weight = None

        if not grad_output.is_cuda:
            raise NotImplementedError("Deformable Conv is not supported on CPUs!")
        else:
            cur_im2col_step = _DeformConv._cal_im2col_step(input.shape[0], ctx.im2col_step)
            assert (input.shape[0] % cur_im2col_step) == 0, "im2col step must divide batchsize"

            if ctx.needs_input_grad[0] or ctx.needs_input_grad[1]:
                grad_input = torch.zeros_like(input)
                grad_offset = torch.zeros_like(offset)
                _C.deform_conv_backward_input(
                    input,
                    offset,
                    grad_output,
                    grad_input,
                    grad_offset,
                    weight,
                    ctx.bufs_[0],
                    weight.size(3),
                    weight.size(2),
                    ctx.stride[1],
                    ctx.stride[0],
                    ctx.padding[1],
                    ctx.padding[0],
                    ctx.dilation[1],
                    ctx.dilation[0],
                    ctx.groups,
                    ctx.deformable_groups,
                    cur_im2col_step,
                )

            if ctx.needs_input_grad[2]:
                grad_weight = torch.zeros_like(weight)
                _C.deform_conv_backward_filter(
                    input,
                    offset,
                    grad_output,
                    grad_weight,
                    ctx.bufs_[0],
                    ctx.bufs_[1],
                    weight.size(3),
                    weight.size(2),
                    ctx.stride[1],
                    ctx.stride[0],
                    ctx.padding[1],
                    ctx.padding[0],
                    ctx.dilation[1],
                    ctx.dilation[0],
                    ctx.groups,
                    ctx.deformable_groups,
                    1,
                    cur_im2col_step,
                )

        return grad_input, grad_offset, grad_weight, None, None, None, None, None, None

    @staticmethod
    def _output_size(input, weight, padding, dilation, stride):
        channels = weight.size(0)
        output_size = (input.size(0), channels)
        for d in range(input.dim() - 2):
            in_size = input.size(d + 2)
            pad = padding[d]
            kernel = dilation[d] * (weight.size(d + 2) - 1) + 1
            stride_ = stride[d]
            output_size += ((in_size + (2 * pad) - kernel) // stride_ + 1,)
        if not all(map(lambda s: s > 0, output_size)):
            raise ValueError(
                "convolution input is too small (output would be {})".format(
                    "x".join(map(str, output_size))
                )
            )
        return output_size

    @staticmethod
    @lru_cache(maxsize=128)
    def _cal_im2col_step(input_size, default_size):
        """
        Calculate proper im2col step size, which should be divisible by input_size and not larger
        than prefer_size. Meanwhile the step size should be as large as possible to be more
        efficient. So we choose the largest one among all divisors of input_size which are smaller
        than prefer_size.
        :param input_size: input batch size .
        :param default_size: default preferred im2col step size.
        :return: the largest proper step size.
        """
        if input_size <= default_size:
            return input_size
        best_step = 1
        for step in range(2, min(int(math.sqrt(input_size)) + 1, default_size)):
            if input_size % step == 0:
                if input_size // step <= default_size:
                    return input_size // step
                best_step = step

        return best_step


class _ModulatedDeformConv(Function):
    @staticmethod
    def forward(
        ctx,
        input,
        offset,
        mask,
        weight,
        bias=None,
        stride=1,
        padding=0,
        dilation=1,
        groups=1,
        deformable_groups=1,
    ):
        ctx.stride = stride
        ctx.padding = padding
        ctx.dilation = dilation
        ctx.groups = groups
        ctx.deformable_groups = deformable_groups
        ctx.with_bias = bias is not None
        if not ctx.with_bias:
            bias = input.new_empty(1)  # fake tensor
        if not input.is_cuda:
            raise NotImplementedError("Deformable Conv is not supported on CPUs!")
        if (
            weight.requires_grad
            or mask.requires_grad
            or offset.requires_grad
            or input.requires_grad
        ):
            ctx.save_for_backward(input, offset, mask, weight, bias)
        output = input.new_empty(_ModulatedDeformConv._infer_shape(ctx, input, weight))
        ctx._bufs = [input.new_empty(0), input.new_empty(0)]
        _C.modulated_deform_conv_forward(
            input,
            weight,
            bias,
            ctx._bufs[0],
            offset,
            mask,
            output,
            ctx._bufs[1],
            weight.shape[2],
            weight.shape[3],
            ctx.stride,
            ctx.stride,
            ctx.padding,
            ctx.padding,
            ctx.dilation,
            ctx.dilation,
            ctx.groups,
            ctx.deformable_groups,
            ctx.with_bias,
        )
        return output

    @staticmethod
    @once_differentiable
    def backward(ctx, grad_output):
        if not grad_output.is_cuda:
            raise NotImplementedError("Deformable Conv is not supported on CPUs!")
        input, offset, mask, weight, bias = ctx.saved_tensors
        grad_input = torch.zeros_like(input)
        grad_offset = torch.zeros_like(offset)
        grad_mask = torch.zeros_like(mask)
        grad_weight = torch.zeros_like(weight)
        grad_bias = torch.zeros_like(bias)
        _C.modulated_deform_conv_backward(
            input,
            weight,
            bias,
            ctx._bufs[0],
            offset,
            mask,
            ctx._bufs[1],
            grad_input,
            grad_weight,
            grad_bias,
            grad_offset,
            grad_mask,
            grad_output,
            weight.shape[2],
            weight.shape[3],
            ctx.stride,
            ctx.stride,
            ctx.padding,
            ctx.padding,
            ctx.dilation,
            ctx.dilation,
            ctx.groups,
            ctx.deformable_groups,
            ctx.with_bias,
        )
        if not ctx.with_bias:
            grad_bias = None

        return (
            grad_input,
            grad_offset,
            grad_mask,
            grad_weight,
            grad_bias,
            None,
            None,
            None,
            None,
            None,
        )

    @staticmethod
    def _infer_shape(ctx, input, weight):
        n = input.size(0)
        channels_out = weight.size(0)
        height, width = input.shape[2:4]
        kernel_h, kernel_w = weight.shape[2:4]
        height_out = (
            height + 2 * ctx.padding - (ctx.dilation * (kernel_h - 1) + 1)
        ) // ctx.stride + 1
        width_out = (
            width + 2 * ctx.padding - (ctx.dilation * (kernel_w - 1) + 1)
        ) // ctx.stride + 1
        return n, channels_out, height_out, width_out


deform_conv = _DeformConv.apply
modulated_deform_conv = _ModulatedDeformConv.apply


class DeformConv(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        stride=1,
        padding=0,
        dilation=1,
        groups=1,
        deformable_groups=1,
        bias=False,
        norm=None,
        activation=None,
    ):
        """
        Deformable convolution from :paper:`deformconv`.

        Arguments are similar to :class:`Conv2D`. Extra arguments:

        Args:
            deformable_groups (int): number of groups used in deformable convolution.
            norm (nn.Module, optional): a normalization layer
            activation (callable(Tensor) -> Tensor): a callable activation function
        """
        super(DeformConv, self).__init__()

        assert not bias
        assert in_channels % groups == 0, "in_channels {} cannot be divisible by groups {}".format(
            in_channels, groups
        )
        assert (
            out_channels % groups == 0
        ), "out_channels {} cannot be divisible by groups {}".format(out_channels, groups)

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = _pair(kernel_size)
        self.stride = _pair(stride)
        self.padding = _pair(padding)
        self.dilation = _pair(dilation)
        self.groups = groups
        self.deformable_groups = deformable_groups
        self.norm = norm
        self.activation = activation

        self.weight = nn.Parameter(
            torch.Tensor(out_channels, in_channels // self.groups, *self.kernel_size)
        )
        self.bias = None

        nn.init.kaiming_uniform_(self.weight, nonlinearity="relu")

    def forward(self, x, offset):
        if x.numel() == 0:
            # When input is empty, we want to return a empty tensor with "correct" shape,
            # So that the following operations will not panic
            # if they check for the shape of the tensor.
            # This computes the height and width of the output tensor
            output_shape = [
                (i + 2 * p - (di * (k - 1) + 1)) // s + 1
                for i, p, di, k, s in zip(
                    x.shape[-2:], self.padding, self.dilation, self.kernel_size, self.stride
                )
            ]
            output_shape = [x.shape[0], self.weight.shape[0]] + output_shape
            return _NewEmptyTensorOp.apply(x, output_shape)

        x = deform_conv(
            x,
            offset,
            self.weight,
            self.stride,
            self.padding,
            self.dilation,
            self.groups,
            self.deformable_groups,
        )
        if self.norm is not None:
            x = self.norm(x)
        if self.activation is not None:
            x = self.activation(x)
        return x

    def extra_repr(self):
        tmpstr = "in_channels=" + str(self.in_channels)
        tmpstr += ", out_channels=" + str(self.out_channels)
        tmpstr += ", kernel_size=" + str(self.kernel_size)
        tmpstr += ", stride=" + str(self.stride)
        tmpstr += ", padding=" + str(self.padding)
        tmpstr += ", dilation=" + str(self.dilation)
        tmpstr += ", groups=" + str(self.groups)
        tmpstr += ", deformable_groups=" + str(self.deformable_groups)
        tmpstr += ", bias=False"
        return tmpstr


class ModulatedDeformConv(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        stride=1,
        padding=0,
        dilation=1,
        groups=1,
        deformable_groups=1,
        bias=True,
        norm=None,
        activation=None,
    ):
        """
        Modulated deformable convolution from :paper:`deformconv2`.

        Arguments are similar to :class:`Conv2D`. Extra arguments:

        Args:
            deformable_groups (int): number of groups used in deformable convolution.
            norm (nn.Module, optional): a normalization layer
            activation (callable(Tensor) -> Tensor): a callable activation function
        """
        super(ModulatedDeformConv, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = _pair(kernel_size)
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.groups = groups
        self.deformable_groups = deformable_groups
        self.with_bias = bias
        self.norm = norm
        self.activation = activation

        self.weight = nn.Parameter(
            torch.Tensor(out_channels, in_channels // groups, *self.kernel_size)
        )
        if bias:
            self.bias = nn.Parameter(torch.Tensor(out_channels))
        else:
            self.bias = None

        nn.init.kaiming_uniform_(self.weight, nonlinearity="relu")
        if self.bias is not None:
            nn.init.constant_(self.bias, 0)

    def forward(self, x, offset, mask):
        if x.numel() == 0:
            output_shape = [
                (i + 2 * p - (di * (k - 1) + 1)) // s + 1
                for i, p, di, k, s in zip(
                    x.shape[-2:], self.padding, self.dilation, self.kernel_size, self.stride
                )
            ]
            output_shape = [x.shape[0], self.weight.shape[0]] + output_shape
            return _NewEmptyTensorOp.apply(x, output_shape)

        x = modulated_deform_conv(
            x,
            offset,
            mask,
            self.weight,
            self.bias,
            self.stride,
            self.padding,
            self.dilation,
            self.groups,
            self.deformable_groups,
        )
        if self.norm is not None:
            x = self.norm(x)
        if self.activation is not None:
            x = self.activation(x)
        return x

    def extra_repr(self):
        tmpstr = "in_channels=" + str(self.in_channels)
        tmpstr += ", out_channels=" + str(self.out_channels)
        tmpstr += ", kernel_size=" + str(self.kernel_size)
        tmpstr += ", stride=" + str(self.stride)
        tmpstr += ", padding=" + str(self.padding)
        tmpstr += ", dilation=" + str(self.dilation)
        tmpstr += ", groups=" + str(self.groups)
        tmpstr += ", deformable_groups=" + str(self.deformable_groups)
        tmpstr += ", bias=" + str(self.with_bias)
        return tmpstr
