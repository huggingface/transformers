# Copyright (c) Facebook, Inc. and its affiliates.
# -*- coding: utf-8 -*-

import typing
import fvcore
from fvcore.nn import activation_count, flop_count, parameter_count, parameter_count_table
from torch import nn

from detectron2.export import TracingAdapter

__all__ = [
    "activation_count_operators",
    "flop_count_operators",
    "parameter_count_table",
    "parameter_count",
]

FLOPS_MODE = "flops"
ACTIVATIONS_MODE = "activations"


# Some extra ops to ignore from counting, including elementwise and reduction ops
_IGNORED_OPS = {
    "aten::add",
    "aten::add_",
    "aten::argmax",
    "aten::argsort",
    "aten::batch_norm",
    "aten::constant_pad_nd",
    "aten::div",
    "aten::div_",
    "aten::exp",
    "aten::log2",
    "aten::max_pool2d",
    "aten::meshgrid",
    "aten::mul",
    "aten::mul_",
    "aten::neg",
    "aten::nonzero_numpy",
    "aten::reciprocal",
    "aten::rsub",
    "aten::sigmoid",
    "aten::sigmoid_",
    "aten::softmax",
    "aten::sort",
    "aten::sqrt",
    "aten::sub",
    "torchvision::nms",  # TODO estimate flop for nms
}


class FlopCountAnalysis(fvcore.nn.FlopCountAnalysis):
    """
    Same as :class:`fvcore.nn.FlopCountAnalysis`, but supports detectron2 models.
    """

    def __init__(self, model, inputs):
        """
        Args:
            model (nn.Module):
            inputs (Any): inputs of the given model. Does not have to be tuple of tensors.
        """
        wrapper = TracingAdapter(model, inputs, allow_non_tensor=True)
        super().__init__(wrapper, wrapper.flattened_inputs)
        self.set_op_handle(**{k: None for k in _IGNORED_OPS})


def flop_count_operators(model: nn.Module, inputs: list) -> typing.DefaultDict[str, float]:
    """
    Implement operator-level flops counting using jit.
    This is a wrapper of :func:`fvcore.nn.flop_count` and adds supports for standard
    detection models in detectron2.
    Please use :class:`FlopCountAnalysis` for more advanced functionalities.

    Note:
        The function runs the input through the model to compute flops.
        The flops of a detection model is often input-dependent, for example,
        the flops of box & mask head depends on the number of proposals &
        the number of detected objects.
        Therefore, the flops counting using a single input may not accurately
        reflect the computation cost of a model. It's recommended to average
        across a number of inputs.

    Args:
        model: a detectron2 model that takes `list[dict]` as input.
        inputs (list[dict]): inputs to model, in detectron2's standard format.
            Only "image" key will be used.
        supported_ops (dict[str, Handle]): see documentation of :func:`fvcore.nn.flop_count`

    Returns:
        Counter: Gflop count per operator
    """
    old_train = model.training
    model.eval()
    ret = FlopCountAnalysis(model, inputs).by_operator()
    model.train(old_train)
    return {k: v / 1e9 for k, v in ret.items()}


def activation_count_operators(
    model: nn.Module, inputs: list, **kwargs
) -> typing.DefaultDict[str, float]:
    """
    Implement operator-level activations counting using jit.
    This is a wrapper of fvcore.nn.activation_count, that supports standard detection models
    in detectron2.

    Note:
        The function runs the input through the model to compute activations.
        The activations of a detection model is often input-dependent, for example,
        the activations of box & mask head depends on the number of proposals &
        the number of detected objects.

    Args:
        model: a detectron2 model that takes `list[dict]` as input.
        inputs (list[dict]): inputs to model, in detectron2's standard format.
            Only "image" key will be used.

    Returns:
        Counter: activation count per operator
    """
    return _wrapper_count_operators(model=model, inputs=inputs, mode=ACTIVATIONS_MODE, **kwargs)


def _wrapper_count_operators(
    model: nn.Module, inputs: list, mode: str, **kwargs
) -> typing.DefaultDict[str, float]:
    # ignore some ops
    supported_ops = {k: lambda *args, **kwargs: {} for k in _IGNORED_OPS}
    supported_ops.update(kwargs.pop("supported_ops", {}))
    kwargs["supported_ops"] = supported_ops

    assert len(inputs) == 1, "Please use batch size=1"
    tensor_input = inputs[0]["image"]
    inputs = [{"image": tensor_input}]  # remove other keys, in case there are any

    old_train = model.training
    if isinstance(model, (nn.parallel.distributed.DistributedDataParallel, nn.DataParallel)):
        model = model.module
    wrapper = TracingAdapter(model, inputs)
    wrapper.eval()
    if mode == FLOPS_MODE:
        ret = flop_count(wrapper, (tensor_input,), **kwargs)
    elif mode == ACTIVATIONS_MODE:
        ret = activation_count(wrapper, (tensor_input,), **kwargs)
    else:
        raise NotImplementedError("Count for mode {} is not supported yet.".format(mode))
    # compatible with change in fvcore
    if isinstance(ret, tuple):
        ret = ret[0]
    model.train(old_train)
    return ret
