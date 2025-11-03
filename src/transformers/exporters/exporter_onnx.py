from typing import TYPE_CHECKING

from ..generation.utils import GenerationMixin
from ..masking_utils import ALL_MASK_ATTENTION_FUNCTIONS
from ..utils import logging
from ..utils.export_config import OnnxConfig
from ..utils.import_utils import is_torch_available, is_torch_greater_or_equal
from .base import HfExporter
from .exporter_dynamo import eager_mask_without_vmap, register_dynamic_cache_export_support, sdpa_mask_without_vmap


if is_torch_available():
    import torch

if TYPE_CHECKING:
    from ..modeling_utils import PreTrainedModel

logger = logging.get_logger(__file__)


class OnnxExporter(HfExporter):
    export_config: OnnxConfig

    required_packages = ["torch", "onnx", "onnxscript"]

    def validate_environment(self, *args, **kwargs):
        super().validate_environment(*args, **kwargs)

        # TODO: check min version where this works
        if not is_torch_greater_or_equal("2.9.0"):
            raise ImportError("OnnxExporter requires torch>=2.9.0 for Dynamo based ONNX export.")

    def export(self, model: "PreTrainedModel"):
        from torch.export import Dim
        from torch.onnx import ONNXProgram

        if self.export_config.sample_inputs is None:
            raise NotImplementedError(
                "OnnxExporter can't automatically generate export inptus. Please provide sample_inputs in the exporter_config as a dictionary. "
                "You can do so by using the tokenizer/processor to prepare a batch of inputs as you would do for a normal forward pass. "
                "OnnxExporter can automatically generate past_key_values and its dynamic shapes if the model is "
                "auto-regressive and model.config.use_cache is set to True."
            )

        args = ()
        kwargs = self.export_config.sample_inputs
        dynamic_shapes = self.export_config.dynamic_shapes

        if isinstance(model, GenerationMixin) and model.config.use_cache:
            register_dynamic_cache_export_support()

            if "past_key_values" not in kwargs:
                sample_outputs = model(**kwargs)
                kwargs["past_key_values"] = sample_outputs.past_key_values

                if dynamic_shapes is not None:
                    dynamic_shapes["past_key_values"] = [
                        [{0: Dim.DYNAMIC, 2: Dim.DYNAMIC} for _ in range(len(kwargs["past_key_values"].layers))],
                        [{0: Dim.DYNAMIC, 2: Dim.DYNAMIC} for _ in range(len(kwargs["past_key_values"].layers))],
                    ]

        if model.config._attn_implementation in ["sdpa", "eager"]:
            ALL_MASK_ATTENTION_FUNCTIONS["sdpa"] = sdpa_mask_without_vmap
            ALL_MASK_ATTENTION_FUNCTIONS["eager"] = eager_mask_without_vmap

        onnx_program: ONNXProgram = torch.onnx.export(
            model,
            args=args,
            kwargs=kwargs,
            f=self.export_config.f,
            dynamic_shapes=dynamic_shapes,
        )
        model.exported_model = onnx_program
