#-------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation.  All rights reserved.
# Licensed under the MIT License.
#--------------------------------------------------------------------------

# Convert Bert ONNX model converted from TensorFlow or exported from PyTorch to use Attention, Gelu,
# SkipLayerNormalization and EmbedLayerNormalization ops to optimize
# performance on NVidia GPU and CPU.
#
# For Bert model exported from PyTorch, OnnxRuntime has bert model optimization support internally.
# You can use the option --use_onnxruntime to check optimizations from OnnxRuntime.
# For Bert model file like name.onnx, optimized model for GPU or CPU from OnnxRuntime will output as
# name_ort_gpu.onnx or name_ort_cpu.onnx in the same directory.
#
# This script is retained for experiment purpose. Useful senarios like the following:
#  (1) Change model from fp32 to fp16 for mixed precision inference in GPU with Tensor Core.
#  (2) Change input data type from int64 to int32.
#  (3) Some model cannot be handled by OnnxRuntime, and you can modify this script to get optimized model.
#
#
# youki@microsoft.com
# This modele (onnx_graph_optimizer) is an extension of the transformer graph optimizer from onnxruntime repository.
# On top of the onnxruntime 1.4.0, FBGEMM based quantization operators are added.
# fusion_attention_fbgemm.py
# fusion_quantmatmul_fbgemm.py
#

import logging
import coloredlogs
import onnx
import os
import sys
import argparse
import numpy as np
from typing import Dict
from collections import deque
from onnx import ModelProto, TensorProto, numpy_helper, load_model
from .onnx_model_bert import BertOnnxModel, BertOptimizationOptions

logger = logging.getLogger(__name__)

# Map model type to tuple: optimizer class, export tools (pytorch, tf2onnx, keras2onnx) and whether OnnxRuntime has the optimization.
MODEL_CLASSES = {
    "bert": (BertOnnxModel, "pytorch", True),
}


def optimize_by_onnxruntime(onnx_model_path: str,
                            use_gpu: bool = False,
                            optimized_model_path: str = None,
                            opt_level: int = 99) -> str:
    """
    Use onnxruntime to optimize model.

    Args:
        onnx_model_path (str): the path of input onnx model.
        use_gpu (bool): whether the optimized model is targeted to run in GPU.
        optimized_model_path (str or None): the path of optimized model.
        opt_level (int): graph optimization level.

    Returns:
        optimized_model_path (str): the path of optimized model
    """
    import onnxruntime

    if use_gpu and 'CUDAExecutionProvider' not in onnxruntime.get_available_providers():
        logger.error("There is no gpu for onnxruntime to do optimization.")
        return onnx_model_path

    sess_options = onnxruntime.SessionOptions()
    if opt_level == 1:
        sess_options.graph_optimization_level = onnxruntime.GraphOptimizationLevel.ORT_ENABLE_BASIC
    elif opt_level == 2:
        sess_options.graph_optimization_level = onnxruntime.GraphOptimizationLevel.ORT_ENABLE_EXTENDED
    else:
        assert opt_level == 99
        sess_options.graph_optimization_level = onnxruntime.GraphOptimizationLevel.ORT_ENABLE_ALL

    if optimized_model_path is None:
        path_prefix = onnx_model_path[:-5]  #remove .onnx suffix
        optimized_model_path = "{}_o{}_{}.onnx".format(path_prefix, opt_level, "gpu" if use_gpu else "cpu")

    sess_options.optimized_model_filepath = optimized_model_path

    if not use_gpu:
        session = onnxruntime.InferenceSession(onnx_model_path, sess_options, providers=['CPUExecutionProvider'])
    else:
        session = onnxruntime.InferenceSession(onnx_model_path, sess_options)
        assert 'CUDAExecutionProvider' in session.get_providers()  # Make sure there is GPU

    assert os.path.exists(optimized_model_path) and os.path.isfile(optimized_model_path)
    logger.debug("Save optimized model by onnxruntime to {}".format(optimized_model_path))
    return optimized_model_path


def get_fusion_statistics(optimized_model_path: str) -> Dict[str, int]:
    """
    Get counter of fused operators in optimized model.

    Args:
        optimized_model_path (str): the path of onnx model.

    Returns:
        A dictionary with operator type as key, and count as value
    """
    model = load_model(optimized_model_path, format=None, load_external_data=True)
    optimizer = BertOnnxModel(model, num_heads=12, hidden_size=768)
    return optimizer.get_fused_operator_statistics()


def _parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', required=True, type=str, help="input onnx model path")

    parser.add_argument('--output', required=True, type=str, help="optimized onnx model path")

    parser.add_argument('--model_type',
                        required=False,
                        type=str.lower,
                        default="bert",
                        choices=list(MODEL_CLASSES.keys()),
                        help="Model type selected in the list: " + ", ".join(MODEL_CLASSES.keys()))

    parser.add_argument('--num_heads',
                        required=False,
                        type=int,
                        default=12,
                        help="number of attention heads. 12 for bert-base model and 16 for bert-large")

    parser.add_argument('--head_size',
                        required=False,
                        type=int,
                        default=0,
                        help="size of each attention head. 0 means head_size = hidden_size / num_heads.")

    parser.add_argument('--hidden_size',
                        required=False,
                        type=int,
                        default=768,
                        help="bert model hidden size. 768 for bert-base model and 1024 for bert-large")

    parser.add_argument('--input_int32',
                        required=False,
                        action='store_true',
                        help="Use int32 (instead of int64) tensor as input to avoid unnecessary data cast")
    parser.set_defaults(input_int32=False)

    parser.add_argument(
        '--float16',
        required=False,
        action='store_true',
        help="If your target device is V100 or T4 GPU, use this to convert float32 to float16 for best performance")
    parser.set_defaults(float16=False)

    parser.add_argument('--disable_attention_fbgemm', required=False, action='store_true', help="disable Attention fusion with FBGEMM")
    parser.set_defaults(disable_attention_fbgemm=False)
    parser.add_argument('--disable_attention', required=False, action='store_true', help="disable Attention fusion")
    parser.set_defaults(disable_attention=False)
    parser.add_argument('--disable_matmul_fbgemm', required=False, action='store_true', help="disable MatMul Quantization with FBGEMM")
    parser.set_defaults(disable_matmul_fbgemm=False)

    parser.add_argument('--disable_skip_layer_norm',
                        required=False,
                        action='store_true',
                        help="disable SkipLayerNormalization fusion")
    parser.set_defaults(disable_skip_layer_norm=False)

    parser.add_argument('--disable_embed_layer_norm',
                        required=False,
                        action='store_true',
                        help="disable EmbedLayerNormalization fusion")
    parser.set_defaults(disable_embed_layer_norm=False)

    parser.add_argument('--disable_bias_skip_layer_norm',
                        required=False,
                        action='store_true',
                        help="disable Add Bias and SkipLayerNormalization fusion")
    parser.set_defaults(disable_bias_skip_layer_norm=False)

    parser.add_argument('--disable_bias_gelu',
                        required=False,
                        action='store_true',
                        help="disable Add Bias and Gelu/FastGelu fusion")
    parser.set_defaults(disable_bias_gelu=False)

    parser.add_argument('--disable_layer_norm',
                        required=False,
                        action='store_true',
                        help="disable LayerNormalization fusion")
    parser.set_defaults(disable_layer_norm=False)

    parser.add_argument('--disable_gelu', required=False, action='store_true', help="disable Gelu fusion")
    parser.set_defaults(disable_gelu=False)

    parser.add_argument('--enable_gelu_approximation',
                        required=False,
                        action='store_true',
                        help="enable Gelu/BiasGelu to FastGelu conversion")
    parser.set_defaults(enable_gelu_approximation=False)

    parser.add_argument('--use_mask_index',
                        required=False,
                        action='store_true',
                        help="use mask index instead of raw attention mask in attention operator")
    parser.set_defaults(use_mask_index=False)

    parser.add_argument('--no_attention_mask',
                        required=False,
                        action='store_true',
                        help="no attention mask. Only works for model_type=bert")
    parser.set_defaults(no_attention_mask=False)

    parser.add_argument('--verbose', required=False, action='store_true')
    parser.set_defaults(verbose=False)

    parser.add_argument('--use_gpu', required=False, action='store_true', help="use GPU inference")
    parser.set_defaults(use_gpu=False)

    parser.add_argument('--only_onnxruntime', required=False, action='store_true', help="optimized by onnxruntime only")
    parser.set_defaults(only_onnxruntime=False)

    parser.add_argument('--opt_level',
                        required=False,
                        type=int,
                        choices=[0, 1, 2, 99],
                        default=0,
                        help="onnxruntime optimization level. 0 will disable onnxruntime.")

    args = parser.parse_args()

    return args


def _get_optimization_options(args):
    optimization_options = BertOptimizationOptions(args.model_type)
    if args.disable_gelu:
        optimization_options.enable_gelu = False
    if args.disable_layer_norm:
        optimization_options.enable_layer_norm = False
    if args.disable_attention:
        optimization_options.enable_attention = False
    if args.disable_attention_fbgemm:
        optimization_options.enable_attention_fbgemm = False
    if args.disable_matmul_fbgemm:
        optimization_options.enable_quantize_matmul = False
    if args.disable_skip_layer_norm:
        optimization_options.enable_skip_layer_norm = False
    if args.disable_embed_layer_norm:
        optimization_options.enable_embed_layer_norm = False
    if args.disable_bias_skip_layer_norm:
        optimization_options.enable_bias_skip_layer_norm = False
    if args.disable_bias_gelu:
        optimization_options.enable_bias_gelu = False
    if args.enable_gelu_approximation:
        optimization_options.enable_gelu_approximation = True
    if args.use_mask_index:
        optimization_options.use_raw_attention_mask(False)
    if args.no_attention_mask:
        optimization_options.disable_attention_mask()

    return optimization_options


def optimize_model(input,
                   model_type='bert',
                   num_heads=12,
                   head_size=0,
                   hidden_size=768,
                   optimization_options=None,
                   opt_level=0,
                   use_gpu=False,
                   only_onnxruntime=False):
    """ Optimize Model by OnnxRuntime and/or offline fusion logic.

    The following optimizes model by OnnxRuntime only, and no offline fusion logic:
        optimize_model(input, opt_level=1, use_gpu=False, only_onnxruntime=True)
    If you want to optimize model by offline fusion logic.
        optimize_model(input, model_type, num_heads=12, hidden_size=768, optimization_options=your_options)

    Args:
        input (str): input model path.
        model_type (str): model type - like bert, bert_tf, bert_keras or gpt2.
        num_heads (int): number of attention heads.
        head_size (int): size of each head. if it's 0, it's hidden_size / num_heads.
        hidden_size (int): hidden size.
        optimization_options (OptimizationOptions or None): optimization options that can use to turn on/off some fusions.
        opt_level (int): onnxruntime graph optimization level (0, 1, 2 or 99). When the level > 0, onnxruntime will be used to optimize model first.
        use_gpu (bool): use gpu or not for onnxruntime.
        only_onnxruntime (bool): only use onnxruntime to optimize model, and no offline fusion logic is used.

     Returns:
        object of an optimizer class.
    """
    (optimizer_class, producer, run_onnxruntime) = MODEL_CLASSES[model_type]

    temp_model_path = None
    if opt_level > 1:  # Optimization specified for an execution provider.
        temp_model_path = optimize_by_onnxruntime(input, use_gpu=use_gpu, opt_level=opt_level)
    elif run_onnxruntime:
        # Use Onnxruntime to do optimizations (like constant folding and cast elimation) that is not specified to exection provider.
        # CPU provider is used here so that there is no extra node for GPU memory copy.
        temp_model_path = optimize_by_onnxruntime(input, use_gpu=False, opt_level=1)

    model = load_model(temp_model_path or input, format=None, load_external_data=True)

    if model.producer_name and producer != model.producer_name:
        logger.warning(
            f"Model producer not matched: Expect {producer},  Got {model.producer_name} {model.producer_version}. Please specify correct --model_type parameter."
        )

    if optimization_options is None:
        optimization_options = BertOptimizationOptions(model_type)

    if head_size <= 0:
        head_size = hidden_size / num_heads
    optimizer = optimizer_class(model, num_heads, hidden_size, head_size)

    if not only_onnxruntime:
        optimizer.optimize(optimization_options)

    # Remove the temporary model.
    if temp_model_path:
        os.remove(temp_model_path)
        logger.debug("Remove tempoary model: {}".format(temp_model_path))

    optimizer.model.producer_name = "onnxruntime_tools"
    optimizer.model.producer_version = "1.4"

    return optimizer


def _setup_logger(verbose):
    if verbose:
        coloredlogs.install(level='DEBUG', fmt='[%(filename)s:%(lineno)s - %(funcName)20s()] %(message)s')
    else:
        coloredlogs.install(fmt='%(funcName)20s: %(message)s')


def main():
    args = _parse_arguments()

    _setup_logger(args.verbose)

    optimization_options = _get_optimization_options(args)

    optimizer = optimize_model(args.input,
                               args.model_type,
                               args.num_heads,
                               args.head_size,
                               args.hidden_size,
                               opt_level=args.opt_level,
                               optimization_options=optimization_options,
                               use_gpu=args.use_gpu,
                               only_onnxruntime=args.only_onnxruntime)

    if args.float16:
        optimizer.convert_model_float32_to_float16()

    if args.input_int32:
        optimizer.change_input_to_int32()

    optimizer.save_model_to_file(args.output)

    if optimizer.is_fully_optimized():
        logger.info("The model has been fully optimized.")
    else:
        logger.info("The model has been optimized.")


if __name__ == "__main__":
    main()
