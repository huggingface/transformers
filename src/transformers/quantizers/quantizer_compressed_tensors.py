# Copyright 2024 The HuggingFace Inc. team. All rights reserved.
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


from ..utils import is_compressed_tensors_available, is_torch_available, logging
from ..utils.quantization_config import CompressedTensorsConfig
from .base import HfQuantizer


if is_torch_available():
    import torch

logger = logging.get_logger(__name__)


class CompressedTensorsHfQuantizer(HfQuantizer):
    """
    Quantizer for the compressed_tensors package.  Loads and restores models to
    quantized state with compressed_tensors
    """

    requires_calibration = True

    def __init__(self, quantization_config: CompressedTensorsConfig, **kwargs):
        super().__init__(quantization_config, **kwargs)

        if not is_compressed_tensors_available():
            raise ImportError(
                "Using `compressed_tensors` quantized models requires the compressed-tensors library: "
                "`pip install compressed-tensors`"
            )

        # Call post_init here to ensure proper config setup when `run_compressed`
        # is provided directly via CompressedTensorsConfig, and to avoid duplicate logging.

        quantization_config.post_init()
        from compressed_tensors.compressors import ModelCompressor

        self.compressor = ModelCompressor.from_compression_config(quantization_config)
        self.run_compressed = quantization_config.run_compressed
        self.quantization_config = quantization_config

    def validate_environment(self, *args, **kwargs):
        if not is_compressed_tensors_available():
            raise ImportError(
                "Using `compressed_tensors` quantized models requires the compressed-tensors library: "
                "`pip install compressed-tensors`"
            )

    def update_dtype(self, dtype: "torch.dtype") -> "torch.dtype":
        if dtype != torch.float16:
            logger.info("We suggest you to set `dtype=torch.float16` for better efficiency with compressed_tensors.")
        return dtype

    def _process_model_before_weight_loading(self, model, **kwargs):
        from compressed_tensors.quantization import apply_quantization_config

        ct_quantization_config = self.compressor.quantization_config

        # Always initialize compressed wrappers to match the checkpoint
        apply_quantization_config(model, ct_quantization_config, self.run_compressed)
        if (
            self.quantization_config.is_quantization_compressed
            or self.quantization_config.is_sparsification_compressed
        ):
            self.compressor.compress_model(model=model)

    def _process_model_after_weight_loading(self, model, **kwargs):
        """Decompress loaded model if necessary - need for qat"""

        if (
            self.quantization_config.is_quantization_compressed and not self.run_compressed
        ) or self.quantization_config.is_sparsification_compressed:
            self.compressor.decompress_model(model=model)

    def update_tp_plan(self, config):
        """
        Update the tensor parallelism plan for compressed tensors quantized models.

        This method adds the appropriate TP sharding patterns for both dense and MoE layers,
        including quantization parameters (scales, zero points) that need to follow the same
        sharding pattern as their corresponding weights.

        Tensor Parallelism sharding conventions:
        - Column-wise (colwise): Split the output dimension (first linear layer in MLP, QKV projections)
        - Row-wise (rowwise): Split the input dimension (second linear layer in MLP, output projection)
        - Quantization parameters (scales, zero points) follow the same sharding as their weights

        Args:
            config: The model configuration object

        Returns:
            config: The updated configuration with TP plan
        """
        # Get the actual model config - for encoder-decoder models, use text_config
        text_config = config.get_text_config()
        effective_config = text_config if text_config is not None else config
        model_type = effective_config.__class__.__name__

        # Common TP plan patterns for compressed tensors
        # Attention layers: q_proj, k_proj, v_proj are colwise; o_proj is rowwise
        # MLP layers: gate_proj, up_proj are colwise; down_proj is rowwise
        # For each weight, we also need to shard the corresponding quantization params

        tp_plan = {}

        # Handle MoE models (e.g., Mixtral, Qwen MoE, Grok)
        if any(moe_keyword in model_type for moe_keyword in ["Mixtral", "Qwen2Moe", "Grok"]):
            tp_plan.update({
                # MoE experts - gate and up projections are column-parallel
                "layers.*.feed_forward.experts.*.gate_proj.weight": "colwise",
                "layers.*.feed_forward.experts.*.gate_proj.input_scale": "colwise",
                "layers.*.feed_forward.experts.*.gate_proj.output_scale": "colwise",
                "layers.*.feed_forward.experts.*.gate_proj.weight_scale": "colwise",
                "layers.*.feed_forward.experts.*.gate_proj.act_scale": "colwise",
                "layers.*.feed_forward.experts.*.up_proj.weight": "colwise",
                "layers.*.feed_forward.experts.*.up_proj.input_scale": "colwise",
                "layers.*.feed_forward.experts.*.up_proj.output_scale": "colwise",
                "layers.*.feed_forward.experts.*.up_proj.weight_scale": "colwise",
                "layers.*.feed_forward.experts.*.up_proj.act_scale": "colwise",
                # MoE experts - down projection is row-parallel
                "layers.*.feed_forward.experts.*.down_proj.weight": "rowwise",
                "layers.*.feed_forward.experts.*.down_proj.input_scale": "rowwise",
                "layers.*.feed_forward.experts.*.down_proj.output_scale": "rowwise",
                "layers.*.feed_forward.experts.*.down_proj.weight_scale": "rowwise",
                "layers.*.feed_forward.experts.*.down_proj.act_scale": "rowwise",
            })

        # Handle standard dense models (Llama, Mistral, Qwen2, Gemma, Phi3, Grok, etc.)
        if any(dense_keyword in model_type for dense_keyword in ["Llama", "Mistral", "Qwen2", "Gemma", "Phi3", "Grok"]):
            tp_plan.update({
                # Attention projections - QKV are column-parallel
                "layers.*.self_attn.q_proj.weight": "colwise",
                "layers.*.self_attn.q_proj.input_scale": "colwise",
                "layers.*.self_attn.q_proj.output_scale": "colwise",
                "layers.*.self_attn.q_proj.weight_scale": "colwise",
                "layers.*.self_attn.q_proj.act_scale": "colwise",
                "layers.*.self_attn.k_proj.weight": "colwise",
                "layers.*.self_attn.k_proj.input_scale": "colwise",
                "layers.*.self_attn.k_proj.output_scale": "colwise",
                "layers.*.self_attn.k_proj.weight_scale": "colwise",
                "layers.*.self_attn.k_proj.act_scale": "colwise",
                "layers.*.self_attn.v_proj.weight": "colwise",
                "layers.*.self_attn.v_proj.input_scale": "colwise",
                "layers.*.self_attn.v_proj.output_scale": "colwise",
                "layers.*.self_attn.v_proj.weight_scale": "colwise",
                "layers.*.self_attn.v_proj.act_scale": "colwise",
                # Output projection - row-parallel
                "layers.*.self_attn.o_proj.weight": "rowwise",
                "layers.*.self_attn.o_proj.input_scale": "rowwise",
                "layers.*.self_attn.o_proj.output_scale": "rowwise",
                "layers.*.self_attn.o_proj.weight_scale": "rowwise",
                "layers.*.self_attn.o_proj.act_scale": "rowwise",
                # MLP projections - gate and up are column-parallel
                "layers.*.mlp.gate_proj.weight": "colwise",
                "layers.*.mlp.gate_proj.input_scale": "colwise",
                "layers.*.mlp.gate_proj.output_scale": "colwise",
                "layers.*.mlp.gate_proj.weight_scale": "colwise",
                "layers.*.mlp.gate_proj.act_scale": "colwise",
                "layers.*.mlp.up_proj.weight": "colwise",
                "layers.*.mlp.up_proj.input_scale": "colwise",
                "layers.*.mlp.up_proj.output_scale": "colwise",
                "layers.*.mlp.up_proj.weight_scale": "colwise",
                "layers.*.mlp.up_proj.act_scale": "colwise",
                # Down projection - row-parallel
                "layers.*.mlp.down_proj.weight": "rowwise",
                "layers.*.mlp.down_proj.input_scale": "rowwise",
                "layers.*.mlp.down_proj.output_scale": "rowwise",
                "layers.*.mlp.down_proj.weight_scale": "rowwise",
                "layers.*.mlp.down_proj.act_scale": "rowwise",
            })

        # Handle Qwen3 models (uses different naming convention)
        if "Qwen3" in model_type:
            tp_plan.update({
                # Attention projections
                "layers.*.self_attn.q_proj.weight": "colwise",
                "layers.*.self_attn.q_proj.weight_scale": "colwise",
                "layers.*.self_attn.q_proj.input_scale": "colwise",
                "layers.*.self_attn.k_proj.weight": "colwise",
                "layers.*.self_attn.k_proj.weight_scale": "colwise",
                "layers.*.self_attn.k_proj.input_scale": "colwise",
                "layers.*.self_attn.v_proj.weight": "colwise",
                "layers.*.self_attn.v_proj.weight_scale": "colwise",
                "layers.*.self_attn.v_proj.input_scale": "colwise",
                # Output projection
                "layers.*.self_attn.o_proj.weight": "rowwise",
                "layers.*.self_attn.o_proj.weight_scale": "rowwise",
                "layers.*.self_attn.o_proj.input_scale": "rowwise",
                # MLP projections
                "layers.*.mlp.gate_proj.weight": "colwise",
                "layers.*.mlp.gate_proj.weight_scale": "colwise",
                "layers.*.mlp.gate_proj.input_scale": "colwise",
                "layers.*.mlp.up_proj.weight": "colwise",
                "layers.*.mlp.up_proj.weight_scale": "colwise",
                "layers.*.mlp.up_proj.input_scale": "colwise",
                "layers.*.mlp.down_proj.weight": "rowwise",
                "layers.*.mlp.down_proj.weight_scale": "rowwise",
                "layers.*.mlp.down_proj.input_scale": "rowwise",
            })

        # Apply the TP plan to the appropriate config object
        if tp_plan:
            if text_config is not None:
                # Initialize base_model_tp_plan if it doesn't exist
                if text_config.base_model_tp_plan is None:
                    text_config.base_model_tp_plan = {}
                text_config.base_model_tp_plan.update(tp_plan)
            else:
                # For models without a separate text config
                if getattr(config, "base_model_tp_plan", None) is None:
                    config.base_model_tp_plan = {}
                config.base_model_tp_plan.update(tp_plan)

        return config

    def update_ep_plan(self, config):
        """
        Update the expert parallelism plan for compressed tensors quantized MoE models.

        Expert Parallelism (EP) shards experts across devices. This method sets up the
        appropriate EP plan for MoE models quantized with compressed tensors.

        Args:
            config: The model configuration object

        Returns:
            config: The updated configuration with EP plan
        """
        # Get the actual model config - for encoder-decoder models, use text_config
        text_config = config.get_text_config()
        effective_config = text_config if text_config is not None else config
        model_type = effective_config.__class__.__name__

        # Only MoE models need EP plan
        if not any(moe_keyword in model_type for moe_keyword in ["Mixtral", "Qwen2Moe", "Grok"]):
            return config

        ep_plan = {
            # Expert parallelism - shard experts across devices
            "layers.*.feed_forward.experts": "expert_parallel",
        }

        # Apply the EP plan to the appropriate config object
        if text_config is not None:
            if getattr(text_config, "base_model_ep_plan", None) is None:
                text_config.base_model_ep_plan = {}
            text_config.base_model_ep_plan.update(ep_plan)
        else:
            if getattr(config, "base_model_ep_plan", None) is None:
                config.base_model_ep_plan = {}
            config.base_model_ep_plan.update(ep_plan)

        return config

    @property
    def is_trainable(self):
        return True

    def is_qat_trainable(self) -> bool:
        """Loaded Models can carry out quantization aware training"""
        # models need to be decompressed carry out qat
        return not self.run_compressed or not self.quantization_config.is_quantization_compressed

    def is_serializable(self) -> bool:
        """Models quantized using compressed tensors can be saved to disk"""
        return True
