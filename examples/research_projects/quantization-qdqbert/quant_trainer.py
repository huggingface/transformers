# coding=utf-8
# Copyright 2021 NVIDIA Corporation. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Helper functions for training models with pytorch-quantization"""

import logging
import re

import pytorch_quantization
import pytorch_quantization.nn as quant_nn
import torch
from pytorch_quantization import calib
from pytorch_quantization.tensor_quant import QuantDescriptor


logger = logging.getLogger(__name__)

name_width = 50  # max width of layer names
qname_width = 70  # max width of quantizer names

# ========================================== Quant Trainer API ==========================================


def add_arguments(parser):
    """Add arguments to parser for functions defined in quant_trainer."""

    group = parser.add_argument_group("quant_trainer arguments")
    group.add_argument("--wprec", type=int, default=8, help="weight precision")
    group.add_argument("--aprec", type=int, default=8, help="activation precision")
    group.add_argument("--quant-per-tensor", action="store_true", help="per tensor weight scaling")
    group.add_argument("--quant-disable", action="store_true", help="disable all quantizers")
    group.add_argument("--quant-disable-embeddings", action="store_true", help="disable all embeddings quantizers")
    group.add_argument("--quant-disable-keyword", type=str, nargs="+", help="disable quantizers by keyword")
    group.add_argument("--quant-disable-layer-module", type=str, help="disable quantizers by keyword under layer.")
    group.add_argument("--quant-enable-layer-module", type=str, help="enable quantizers by keyword under layer")
    group.add_argument("--calibrator", default="max", help="which quantization range calibrator to use")
    group.add_argument("--percentile", default=None, type=float, help="percentile for PercentileCalibrator")
    group.add_argument("--fuse-qkv", action="store_true", help="use the same scale factor for qkv")
    group.add_argument("--clip-gelu", metavar="N", type=float, help="clip gelu output maximum value to N")
    group.add_argument(
        "--recalibrate-weights",
        action="store_true",
        help=(
            "recalibrate weight amaxes by taking the max of the weights."
            " amaxes will be computed with the current quantization granularity (axis)."
        ),
    )


def set_default_quantizers(args):
    """Set default quantizers before creating the model."""

    if args.calibrator == "max":
        calib_method = "max"
    elif args.calibrator == "percentile":
        if args.percentile is None:
            raise ValueError("Specify --percentile when using percentile calibrator")
        calib_method = "histogram"
    elif args.calibrator == "mse":
        calib_method = "histogram"
    else:
        raise ValueError(f"Invalid calibrator {args.calibrator}")

    input_desc = QuantDescriptor(num_bits=args.aprec, calib_method=calib_method)
    weight_desc = QuantDescriptor(num_bits=args.wprec, axis=(None if args.quant_per_tensor else (0,)))
    quant_nn.QuantLinear.set_default_quant_desc_input(input_desc)
    quant_nn.QuantLinear.set_default_quant_desc_weight(weight_desc)


def configure_model(model, args, calib=False, eval=False):
    """Function called before the training loop."""

    logger.info("Configuring Model for Quantization")
    logger.info(f"using quantization package {pytorch_quantization.__file__}")

    if not calib:
        if args.quant_disable_embeddings:
            set_quantizer_by_name(model, ["embeddings"], which="weight", _disabled=True)

        if args.quant_disable:
            set_quantizer_by_name(model, [""], _disabled=True)

        if args.quant_disable_keyword:
            set_quantizer_by_name(model, args.quant_disable_keyword, _disabled=True)

        if args.quant_disable_layer_module:
            set_quantizer_by_name(model, [r"layer.\d+." + args.quant_disable_layer_module], _disabled=True)

        if args.quant_enable_layer_module:
            set_quantizer_by_name(model, [r"layer.\d+." + args.quant_enable_layer_module], _disabled=False)

        if args.recalibrate_weights:
            recalibrate_weights(model)

        if args.fuse_qkv:
            fuse_qkv(model, args)

    if args.clip_gelu:
        clip_gelu(model, args.clip_gelu)

    # if args.local_rank in [-1, 0] and not calib:
    print_quant_summary(model)


def enable_calibration(model):
    """Enable calibration of all *_input_quantizer modules in model."""

    logger.info("Enabling Calibration")
    for name, module in model.named_modules():
        if name.endswith("_quantizer"):
            if module._calibrator is not None:
                module.disable_quant()
                module.enable_calib()
            else:
                module.disable()
            logger.info(f"{name:80}: {module}")


def finish_calibration(model, args):
    """Disable calibration and load amax for all "*_input_quantizer modules in model."""

    logger.info("Loading calibrated amax")
    for name, module in model.named_modules():
        if name.endswith("_quantizer"):
            if module._calibrator is not None:
                if isinstance(module._calibrator, calib.MaxCalibrator):
                    module.load_calib_amax()
                else:
                    module.load_calib_amax("percentile", percentile=args.percentile)
                module.enable_quant()
                module.disable_calib()
            else:
                module.enable()
    model.cuda()
    print_quant_summary(model)


# ========================================== Helper Function ==========================================


def fuse_qkv(model, args):
    """Adjust quantization ranges to match an implementation where the QKV projections are implemented with a single GEMM.
    Force the weight and output scale factors to match by taking the max of (Q,K,V).
    """

    def fuse3(qq, qk, qv):
        for mod in [qq, qk, qv]:
            if not hasattr(mod, "_amax"):
                print("          WARNING: NO AMAX BUFFER")
                return
        q = qq._amax.detach().item()
        k = qk._amax.detach().item()
        v = qv._amax.detach().item()

        amax = max(q, k, v)
        qq._amax.fill_(amax)
        qk._amax.fill_(amax)
        qv._amax.fill_(amax)
        logger.info(f"          q={q:5.2f} k={k:5.2f} v={v:5.2f} -> {amax:5.2f}")

    for name, mod in model.named_modules():
        if name.endswith(".attention.self"):
            logger.info(f"FUSE_QKV: {name:{name_width}}")
            fuse3(mod.matmul_q_input_quantizer, mod.matmul_k_input_quantizer, mod.matmul_v_input_quantizer)
            if args.quant_per_tensor:
                fuse3(mod.query._weight_quantizer, mod.key._weight_quantizer, mod.value._weight_quantizer)


def clip_gelu(model, maxval):
    """Clip activations generated by GELU to maxval when quantized.
    Implemented by adjusting the amax of the following input_quantizer.
    """

    for name, mod in model.named_modules():
        if name.endswith(".output.dense") and not name.endswith("attention.output.dense"):
            amax_init = mod._input_quantizer._amax.data.detach().item()
            mod._input_quantizer._amax.data.detach().clamp_(max=maxval)
            amax = mod._input_quantizer._amax.data.detach().item()
            logger.info(f"CLIP_GELU: {name:{name_width}} amax: {amax_init:5.2f} -> {amax:5.2f}")


def expand_amax(model):
    """Expand per-tensor amax to be per channel, where each channel is assigned the per-tensor amax."""

    for name, mod in model.named_modules():
        if hasattr(mod, "_weight_quantizer") and mod._weight_quantizer.axis is not None:
            k = mod.weight.shape[0]
            amax = mod._weight_quantizer._amax.detach()
            mod._weight_quantizer._amax = torch.ones(k, dtype=amax.dtype, device=amax.device) * amax
            print(f"expanding {name} {amax} -> {mod._weight_quantizer._amax}")


def recalibrate_weights(model):
    """Performs max calibration on the weights and updates amax."""

    for name, mod in model.named_modules():
        if hasattr(mod, "_weight_quantizer"):
            if not hasattr(mod.weight_quantizer, "_amax"):
                print("RECALIB: {name:{name_width}} WARNING: NO AMAX BUFFER")
                continue

            # determine which axes to reduce across
            # e.g. a 4D tensor quantized per axis 0 should reduce over (1,2,3)
            axis_set = set() if mod._weight_quantizer.axis is None else set(mod._weight_quantizer.axis)
            reduce_axis = set(range(len(mod.weight.size()))) - axis_set
            amax = pytorch_quantization.utils.reduce_amax(mod.weight, axis=reduce_axis, keepdims=True).detach()
            logger.info(f"RECALIB: {name:{name_width}} {mod._weight_quantizer._amax.flatten()} -> {amax.flatten()}")
            mod._weight_quantizer._amax = amax


def print_model_summary(model, name_width=25, line_width=180, ignore=None):
    """Print model quantization configuration."""

    if ignore is None:
        ignore = []
    elif not isinstance(ignore, list):
        ignore = [ignore]

    name_width = 0
    for name, mod in model.named_modules():
        if not hasattr(mod, "weight"):
            continue
        name_width = max(name_width, len(name))

    for name, mod in model.named_modules():
        input_q = getattr(mod, "_input_quantizer", None)
        weight_q = getattr(mod, "_weight_quantizer", None)
        if not hasattr(mod, "weight"):
            continue
        if type(mod) in ignore:
            continue
        if [True for s in ignore if isinstance(s, str) and s in name]:
            continue
        act_str = f"Act:{input_q.extra_repr()}"
        wgt_str = f"Wgt:{weight_q.extra_repr()}"
        s = f"{name:{name_width}} {act_str} {wgt_str}"
        if len(s) <= line_width:
            logger.info(s)
        else:
            logger.info(f"{name:{name_width}} {act_str}")
            logger.info(f'{"  ":{name_width}} {wgt_str}')


def print_quant_summary(model):
    """Print summary of all quantizer modules in the model."""

    count = 0
    for name, mod in model.named_modules():
        if isinstance(mod, pytorch_quantization.nn.TensorQuantizer):
            print(f"{name:80} {mod}")
            count += 1
    print(f"{count} TensorQuantizers found in model")


def set_quantizer(name, mod, quantizer, k, v):
    """Set attributes for mod.quantizer."""

    quantizer_mod = getattr(mod, quantizer, None)
    if quantizer_mod is not None:
        assert hasattr(quantizer_mod, k)
        setattr(quantizer_mod, k, v)
    else:
        logger.warning(f"{name} has no {quantizer}")


def set_quantizers(name, mod, which="both", **kwargs):
    """Set quantizer attributes for mod."""

    s = f"Warning: changing {which} quantizers of {name:{qname_width}}"
    for k, v in kwargs.items():
        s += f" {k}={v}"
        if which in ["input", "both"]:
            set_quantizer(name, mod, "_input_quantizer", k, v)
        if which in ["weight", "both"]:
            set_quantizer(name, mod, "_weight_quantizer", k, v)
    logger.info(s)


def set_quantizer_by_name(model, names, **kwargs):
    """Set quantizer attributes for layers where name contains a substring in names."""

    for name, mod in model.named_modules():
        if hasattr(mod, "_input_quantizer") or hasattr(mod, "_weight_quantizer"):
            for n in names:
                if re.search(n, name):
                    set_quantizers(name, mod, **kwargs)
        elif name.endswith("_quantizer"):
            for n in names:
                if re.search(n, name):
                    s = f"Warning: changing {name:{name_width}}"
                    for k, v in kwargs.items():
                        s += f" {k}={v}"
                        setattr(mod, k, v)
                    logger.info(s)
