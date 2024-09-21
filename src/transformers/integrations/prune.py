from ..pytorch_utils import (
    prune_linear_layer,
    prune_embedding_layer,
    find_pruneable_heads_and_indices,
    prune_parameter_layer,
)
from ..utils import logging
import torch
from typing import List, Dict
import torch.nn as nn

logger = logging.get_logger(__name__)

PRUNING_TENSOR_MAPPING = {
    "llama": {
        "token_embd": "embed_tokens",
        "layers": "model.layers",
        "ffn_up": "mlp.up_proj",
        "ffn_down": "mlp.down_proj",
        "ffn_gate": "mlp.gate_proj",
        "ffn_norm": "post_attention_layernorm",
        "attn_norm": "input_layernorm",
        "attn_q": "self_attn.q_proj",
        "attn_v": "self_attn.v_proj",
        "attn_k": "self_attn.k_proj",
        "attn_output": "self_attn.o_proj",
        "output_norm": "norm",
        "output.weight": "lm_head",
    },
    "mistral": {
        "token_embd": "embed_tokens",
        "layers": "model.layers",
        "ffn_up": "mlp.up_proj",
        "ffn_down": "mlp.down_proj",
        "ffn_gate": "mlp.gate_proj",
        "ffn_norm": "post_attention_layernorm",
        "attn_norm": "input_layernorm",
        "attn_q": "self_attn.q_proj",
        "attn_v": "self_attn.v_proj",
        "attn_k": "self_attn.k_proj",
        "attn_output": "self_attn.o_proj",
        "output.weight": "lm_head",
        "output_norm": "norm",
    },
    "qwen2": {
        "token_embd": "embed_tokens",
        "layers": "model.layers",
        "ffn_up": "mlp.up_proj",
        "ffn_down": "mlp.down_proj",
        "ffn_gate": "mlp.gate_proj",
        "ffn_norm": "post_attention_layernorm",
        "attn_norm": "input_layernorm",
        "attn_q": "self_attn.q_proj",
        "attn_v": "self_attn.v_proj",
        "attn_k": "self_attn.k_proj",
        "attn_output": "self_attn.o_proj",
        "output.weight": "lm_head",
        "output_norm": "norm",
    },
}

EMBEDDING_DIM_MAPPING = {
    "ffn_up": 1,
    "ffn_down": 0,
    "ffn_gate": 1,
    "attn_q": 1,
    "attn_v": 1,
    "attn_k": 1,
    "attn_output": 0,
    "output.weight": 1,
}

NEURONS_DIM_MAPPING = {
    "ffn_up": 0,
    "ffn_down": 1,
    "ffn_gate": 0,
}

GQA_DIM_MAPPING = {
    "attn_q": 0,
    "attn_output": 1,
}

HEADS_DIM_MAPPING = {
    "attn_q": 0,
    "attn_k": 0,
    "attn_v": 0,
    "attn_output": 1,
}

PRUNING_TYPE_MAPPING = {
    "embeddings": EMBEDDING_DIM_MAPPING,
    "neurons": NEURONS_DIM_MAPPING,
    "gqa": GQA_DIM_MAPPING,
    "heads": HEADS_DIM_MAPPING,
}


class PrunerMixin:
    """
    A mixin class to perform various types of pruning on transformer models.

    Provides pruning for embeddings, neurons, and GQA heads.
    """

    def prune(self, index: torch.LongTensor, layers: List[str], pruning_type: str) -> None:
        """
        General pruning function to prune layers based on specified layer types and pruning type.

        Args:
            index (torch.LongTensor): The indices to keep in the pruned layers.
            layer_types (List[str]): The types of layers to prune (e.g., "ffn_up", "attn_q").
            pruning_type (str): The type of pruning ("embedding", "neuron", "gqa", "heads").
        """
        if index.numel() == 0:
            logger.warning("Pruning index is empty, skipping pruning.")
            return
        if pruning_type not in PRUNING_TYPE_MAPPING:
            raise ValueError(f"Pruning type {pruning_type} not found in PRUNING_TYPE_MAPPING.")
        if self.config.model_type not in PRUNING_TENSOR_MAPPING:
            raise ValueError(f"Model type {self.config.model_type} not found in PRUNING_TENSOR_MAPPING.")

        tensor_mapping = PRUNING_TENSOR_MAPPING[self.config.model_type]

        self._prune_linear_layers(index, layers, tensor_mapping, PRUNING_TYPE_MAPPING[pruning_type])

        if pruning_type == "embeddings":
            lm_head_name = tensor_mapping["output.weight"]
            new_head = prune_linear_layer(getattr(self, lm_head_name), index, dim=1)
            setattr(self, lm_head_name, new_head)

    def prune_embeddings(self, index: torch.LongTensor) -> None:
        """
        Prune embedding-related layers of the model.

        Args:
            index (torch.LongTensor): The indices to keep.
        """
        layers_to_prune = [
            "token_embd",
            "ffn_up",
            "ffn_down",
            "ffn_gate",
            "attn_q",
            "attn_v",
            "attn_k",
            "attn_output",
            "attn_norm",
            "ffn_norm",
            "output_norm",
        ]
        self.prune(index, layers_to_prune, "embeddings")

        self.config.hidden_size = len(index)

    def prune_neurons(self, index: torch.LongTensor) -> None:
        """
        Prune neuron-related layers of the model.

        Args:
            index (torch.LongTensor): The indices to keep.
        """
        layers_to_prune = ["ffn_up", "ffn_down", "ffn_gate"]
        self.prune(index, layers_to_prune, "neurons")

        self.config.intermediate_size = len(index)

    def prune_gqa(self, heads: List[int]) -> None:
        """
        Prune gqa attention heads.

        Args:
            heads (List[int]): The attention heads to prune in the query.
        """
        assert (
            self.config.num_attention_heads - len(heads)
        ) % self.config.num_key_value_heads == 0, "New num_attention_heads must be a multiple of num_key_value_heads"

        layers_to_prune = ["attn_q", "attn_output"]
        heads, index = find_pruneable_heads_and_indices(
            heads, self.config.num_attention_heads, self.config.head_dim, set()
        )
        self.prune(index, layers_to_prune, "gqa")

        self.config.num_attention_heads -= len(heads)

    def prune_heads(self, heads: List[int]):
        """
        Prune attention heads.

        Args:
            heads (List[int]): The attention heads to prune in the query.
        """
        layers_to_prune = ["attn_q", "attn_k", "attn_v", "attn_output"]
        heads, index = find_pruneable_heads_and_indices(
            heads, self.config.num_key_value_heads, self.config.head_dim, set()
        )

        self.prune(index, layers_to_prune, "heads")

        self.config.num_key_value_heads -= len(heads)
        self.config.num_attention_heads -= len(heads)
        
    def prune_layers(self, layers_to_keep: List[int]) -> None:
        """
        Perform depth pruning by removing layers not in layers_to_keep.

        Args:
            layers_to_keep (List[int]): Indices of the layers to keep.
        """
        if self.config.model_type not in PRUNING_TENSOR_MAPPING:
            raise ValueError(f"Model type {self.config.model_type} not found in PRUNING_TENSOR_MAPPING.")

        layers_path = PRUNING_TENSOR_MAPPING[self.config.model_type]["layers"]
        layers = self.get_submodule(layers_path)

        layers_to_keep = sorted(set(layers_to_keep))
        if any(0 <= i for i in layers_to_keep):
            raise ValueError("layers_to_keep contains negative layer indices")
        if any(len(layers) <= i for i in layers_to_keep):
            raise ValueError(
                "layers_to_keep contains layer indices larger than the number of layers of the current model."
            )

        new_layers = nn.ModuleList([layers[i] for i in layers_to_keep])
        setattr(self, layers_path, new_layers)

        self.config.num_hidden_layers = len(layers_to_keep)

    def _prune_linear_layers(self, index: torch.LongTensor, layers_to_prune, tensor_mapping, pruning_mapping) -> None:
        # Iterate over the model layers and apply pruning where necessary
        for name, _ in self.model.named_modules():
            for layer_name in layers_to_prune:
                if name.endswith(f"{tensor_mapping[layer_name]}"):
                    # print(name)
                    if "norm" in layer_name:
                        parent = self.model.get_submodule(name)
                        target_name = "weight"
                        target_layer = getattr(parent, target_name, None)
                        if target_layer is None:
                            raise ValueError(f"{type(target_layer)} not supported for pruning.")
                        pruned_layer = prune_parameter_layer(target_layer, index)
                    else:
                        parent = self.model.get_submodule(".".join(name.split(".")[:-1]))
                        target_name = name.split(".")[-1]
                        target_layer = getattr(parent, target_name)

                        # Prune the target layer
                        if isinstance(target_layer, nn.Linear):
                            pruned_layer = prune_linear_layer(target_layer, index, dim=pruning_mapping[layer_name])
                        elif isinstance(target_layer, nn.Embedding):
                            pruned_layer = prune_embedding_layer(target_layer, index)
                        else:
                            raise ValueError(f"{type(target_layer)} not supported for pruning.")

                    # Replace the layer with the pruned version
                    setattr(parent, target_name, pruned_layer)
                    break
