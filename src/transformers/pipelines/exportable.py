# coding=utf-8
# Copyright 2024 The HuggingFace Inc. team.
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
"""Export utilities for transformers pipelines."""

import inspect
from typing import Optional, Union

from ..image_processing_utils_fast import BaseImageProcessorFast
from ..utils import is_torch_available, logging


if is_torch_available():
    import torch


logger = logging.get_logger(__name__)


class ExportableModule(torch.nn.Module):
    """
    Wrapper that makes a pipeline exportable to ONNX or torch.export by bundling
    preprocessing and postprocessing into the model's forward pass.

    Args:
        model (`torch.nn.Module`):
            The model to wrap (e.g., AutoModelForObjectDetection).
        processor (`BaseImageProcessor` or `ProcessorMixin`):
            The processor for preprocessing and postprocessing.
        post_process_function_name (`str`):
            Name of the postprocessing method on the processor (e.g., "post_process_object_detection").
        include_preprocessing (`bool`, defaults to `True`):
            Whether to include preprocessing in the forward pass.
        include_postprocessing (`bool`, defaults to `True`):
            Whether to include postprocessing in the forward pass.

    Example:
        ```python
        from transformers import pipeline

        pipe = pipeline("object-detection", model="facebook/detr-resnet-50")
        exportable = pipe.get_exportable_module()

        # Export to ONNX
        torch.onnx.export(exportable, example_inputs, "model.onnx", ...)

        # Or use torch.export
        exported_program = torch.export.export(exportable, ...)
        ```
    """

    def __init__(
        self,
        model: "torch.nn.Module",
        processor: "BaseImageProcessorFast",
        post_process_function_name: str,
        include_preprocessing: bool = True,
        include_postprocessing: bool = True,
    ):
        super().__init__()
        if not isinstance(processor, BaseImageProcessorFast):
            raise ValueError("Processor must be a fast image processor")
        self.model = model
        self.processor = processor
        self.post_process_function = getattr(self.processor, post_process_function_name)
        self.signature_params = inspect.signature(self.post_process_function).parameters
        self.include_preprocessing = include_preprocessing
        self.include_postprocessing = include_postprocessing

        # State for preprocessing split
        self.pre_pre_processed_args = None
        self.pre_pre_processed_kwargs = None

    def get_tensors_inputs(self, images, **preprocess_kwargs):
        """
        First stage of preprocessing: convert images to tensors.

        This extracts the tensor conversion step from preprocessing while keeping
        the tensor operations (resize, normalize, etc.) for the forward pass.

        Args:
            images: Input images (PIL.Image, numpy array, or torch.Tensor)
            **preprocess_kwargs: Additional preprocessing arguments

        Returns:
            torch.Tensor: Stacked image tensors ready for the model
        """
        images, pre_pre_processed_args, pre_pre_processed_kwargs = self.processor.preprocess(
            images=images, intermediate_return=True, return_tensors="pt", **preprocess_kwargs
        )
        self.pre_pre_processed_args = pre_pre_processed_args
        self.pre_pre_processed_kwargs = pre_pre_processed_kwargs
        return torch.stack(images)

    def forward(self, images, post_process_kwargs: Optional[dict] = None):
        """
        Forward pass including preprocessing and/or postprocessing.

        Args:
            images: Preprocessed image tensors (if include_preprocessing=True) or
                   raw pixel_values (if include_preprocessing=False)
            post_process_kwargs: Arguments for postprocessing (e.g., target_sizes, threshold)

        Returns:
            Model outputs (with postprocessing applied if include_postprocessing=True)
        """
        if post_process_kwargs is None:
            post_process_kwargs = {}

        if self.include_preprocessing:
            if self.pre_pre_processed_args is None or self.pre_pre_processed_kwargs is None:
                raise ValueError("You must call get_tensors_inputs() before forward() when include_preprocessing=True")

            preprocessed_inputs = self.processor._preprocess(
                images, *self.pre_pre_processed_args, **self.pre_pre_processed_kwargs
            )
        else:
            preprocessed_inputs = {"pixel_values": images}

        outputs = self.model(**preprocessed_inputs)

        if self.include_postprocessing:
            outputs = self.post_process_function(outputs, **post_process_kwargs)

        return outputs


def export_pipeline_to_torch(
    exportable_module: ExportableModule,
    example_inputs: dict,
    save_path: Optional[str] = None,
    dynamic_shapes: Union[dict, bool, None] = None,
    **export_kwargs,
) -> "torch.export.ExportedProgram":
    """
    Export a pipeline to torch.export format.

    Args:
        exportable_module: The ExportableModule to export
        example_inputs: Example inputs for tracing
        save_path: Path to save the exported model (optional)
        dynamic_shapes: Dynamic shapes configuration
        **export_kwargs: Additional arguments for torch.export.export()

    Returns:
        torch.export.ExportedProgram
    """

    exported_program = torch.export.export(
        exportable_module,
        args=(),
        kwargs=example_inputs,
        dynamic_shapes=dynamic_shapes if dynamic_shapes else None,
        strict=False,
        **export_kwargs,
    )

    if save_path:
        torch.export.save(exported_program, save_path)
        logger.info(f"Exported torch model saved to {save_path}")

    return exported_program


def export_pipeline_to_onnx(
    exportable_module: ExportableModule,
    example_inputs: dict,
    save_path: str,
    dynamic_shapes: Union[dict, bool, None] = None,
    optimize: bool = False,
    **export_kwargs,
) -> str:
    """
    Export a pipeline to ONNX format.

    Args:
        exportable_module: The ExportableModule to export
        example_inputs: Example inputs for tracing
        save_path: Path to save the ONNX model (required)
        dynamic_shapes: Dynamic shapes configuration
        optimize: Whether to optimize the ONNX model
        **export_kwargs: Additional arguments for torch.onnx.export()

    Returns:
        Path to saved ONNX file
    """

    if not save_path:
        raise ValueError("save_path is required for ONNX export")

    if not save_path.endswith(".onnx"):
        save_path += ".onnx"

    input_names = list(example_inputs.keys())

    # Export to ONNX using dynamo
    onnx_program = torch.onnx.export(
        exportable_module,
        args=(),
        kwargs=example_inputs,
        f=save_path,
        input_names=input_names,
        output_names=["output"],
        dynamo=True,
        dynamic_shapes=dynamic_shapes if dynamic_shapes else None,
        **export_kwargs,
    )

    if optimize:
        logger.info("Optimizing ONNX model...")
        onnx_program.optimize()

    onnx_program.save(save_path)
    logger.info(f"Exported ONNX model saved to {save_path}")

    return save_path


def export_pipeline_to_torchscript(
    exportable_module: ExportableModule,
    example_inputs: dict,
    save_path: Optional[str] = None,
    **export_kwargs,
) -> "torch.jit.ScriptModule":
    """
    Export a pipeline to TorchScript format.

    Args:
        exportable_module: The ExportableModule to export
        example_inputs: Example inputs for tracing
        save_path: Path to save the TorchScript model (optional)
        **export_kwargs: Additional arguments for torch.jit.trace()

    Returns:
        torch.jit.ScriptModule
    """

    traced_model = torch.jit.trace(
        exportable_module,
        example_inputs=example_inputs,
        **export_kwargs,
    )

    if save_path:
        if not save_path.endswith(".pt"):
            save_path += ".pt"
        traced_model.save(save_path)
        logger.info(f"Exported TorchScript model saved to {save_path}")

    return traced_model
