from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer, LlamaConfig

from ..attention import (
    interleave_query_key_value_tensor_for_attention,
    split_query_key_value_tensor_for_attention,
)
from ..config import GraniteConfig
from ..enums import AttentionHeadType
from ..mlp import interleave_up_gate_tensor_for_mlp, split_up_gate_tensor_for_mlp
from ..safetensors import SafeTensorsWeightsManager
from ..utils import download_repo


def import_from_huggingface_llama(pretrained_model_name_or_path: str, save_path: str) -> None:
    original_config, tokenizer, downloaded_model_path = download_repo(pretrained_model_name_or_path)
    config = _import_config_from_huggingface(original_config)

    safetensors_weight_manager = SafeTensorsWeightsManager(downloaded_model_path)
    state_dict = _import_state_dict_from_huggingface(
        safetensors_weight_manager,
        config.n_layer,
        config.n_head,
        config.num_key_value_heads,
        config.n_embd // config.n_head,
        AttentionHeadType(config.attention_head_type),
    )

    model = AutoModelForCausalLM.from_config(config)
    model.load_state_dict(state_dict)
    model.save_pretrained(save_path)

    config.save_pretrained(save_path)

    if tokenizer is not None:
        tokenizer.save_pretrained(save_path)


def _import_config_from_huggingface(original_config: LlamaConfig) -> GraniteConfig:
    assert original_config.hidden_act == "silu"

    if original_config.num_attention_heads == original_config.num_key_value_heads:
        attention_head_type = "mha"
    elif original_config.num_key_value_heads == 1:
        attention_head_type = "mqa"
    elif original_config.num_attention_heads > original_config.num_key_value_heads:
        attention_head_type = "gqa"

    config = GraniteConfig(
        vocab_size=original_config.vocab_size,
        n_positions=original_config.max_position_embeddings,
        n_embd=original_config.hidden_size,
        n_layer=original_config.num_hidden_layers,
        n_head=original_config.num_attention_heads,
        num_key_value_heads=original_config.num_key_value_heads,
        attention_head_type=attention_head_type,
        position_embedding_type="rope",
        n_inner=original_config.intermediate_size,
        activation_function="swiglu",
        normalization_function="rmsnorm",
        layer_norm_epsilon=original_config.rms_norm_eps,
        use_cache=original_config.use_cache,
        add_bias=original_config.attention_bias,
        tie_word_embeddings=original_config.tie_word_embeddings,
        initializer_range=original_config.initializer_range,
        rope_theta=original_config.rope_theta,
        rope_scaling=original_config.rope_scaling,
        attn_pdrop=original_config.attention_dropout,
    )

    return config


def _import_state_dict_from_huggingface(
    safetensors_weight_manager: SafeTensorsWeightsManager,
    num_layers: int,
    num_heads: int,
    num_key_value_heads: int,
    head_dim: int,
    attention_head_type: AttentionHeadType,
) -> dict:
    state_dict = {
        "transformer.wte.weight": safetensors_weight_manager.get_tensor("model.embed_tokens.weight"),
        "transformer.ln_f.weight": safetensors_weight_manager.get_tensor("model.norm.weight"),
    }

    if safetensors_weight_manager.has_tensor("lm_head.weight"):
        state_dict["lm_head.weight"] = safetensors_weight_manager.get_tensor("lm_head.weight")

    for layer_idx in range(num_layers):
        state_dict[f"transformer.h.{layer_idx}.ln_1.weight"] = safetensors_weight_manager.get_tensor(
            f"model.layers.{layer_idx}.input_layernorm.weight"
        )
        state_dict[f"transformer.h.{layer_idx}.ln_2.weight"] = safetensors_weight_manager.get_tensor(
            f"model.layers.{layer_idx}.post_attention_layernorm.weight"
        )

        state_dict[f"transformer.h.{layer_idx}.mlp.c_fc.weight"] = interleave_up_gate_tensor_for_mlp(
            safetensors_weight_manager.get_tensor(f"model.layers.{layer_idx}.mlp.up_proj.weight"),
            safetensors_weight_manager.get_tensor(f"model.layers.{layer_idx}.mlp.gate_proj.weight"),
        )
        state_dict[f"transformer.h.{layer_idx}.mlp.c_proj.weight"] = safetensors_weight_manager.get_tensor(
            f"model.layers.{layer_idx}.mlp.down_proj.weight"
        )

        state_dict[f"transformer.h.{layer_idx}.attn.c_attn.weight"] = interleave_query_key_value_tensor_for_attention(
            safetensors_weight_manager.get_slice(f"model.layers.{layer_idx}.self_attn.q_proj.weight"),
            safetensors_weight_manager.get_slice(f"model.layers.{layer_idx}.self_attn.k_proj.weight"),
            safetensors_weight_manager.get_slice(f"model.layers.{layer_idx}.self_attn.v_proj.weight"),
            num_heads,
            num_key_value_heads,
            head_dim,
            attention_head_type,
        )
        state_dict[f"transformer.h.{layer_idx}.attn.c_proj.weight"] = safetensors_weight_manager.get_tensor(
            f"model.layers.{layer_idx}.self_attn.o_proj.weight"
        )

    return state_dict


def export_to_huggingface_llama(pretrained_model_name_or_path: str, save_path: str) -> None:
    config: GraniteConfig = AutoConfig.from_pretrained(pretrained_model_name_or_path)
    original_config = _export_config_to_huggingface(config)

    safetensors_weight_manager = SafeTensorsWeightsManager(pretrained_model_name_or_path)
    state_dict = _export_state_dict_to_huggingface(
        safetensors_weight_manager,
        config.n_layer,
        config.n_head,
        config.num_key_value_heads,
        config.n_embd // config.n_head,
        AttentionHeadType(config.attention_head_type),
    )

    model = AutoModelForCausalLM.from_config(original_config)
    model.load_state_dict(state_dict)
    model.save_pretrained(save_path)

    original_config.save_pretrained(save_path)

    try:
        tokenizer = AutoTokenizer.from_pretrained(pretrained_model_name_or_path)
        tokenizer.save_pretrained(save_path)
    except:
        pass


def _export_config_to_huggingface(config: GraniteConfig) -> LlamaConfig:
    assert config.activation_function == "swiglu"
    assert config.normalization_function == "rmsnorm"
    assert config.position_embedding_type == "rope"

    original_config = LlamaConfig(
        vocab_size=config.vocab_size,
        max_position_embeddings=config.n_positions,
        hidden_size=config.n_embd,
        num_hidden_layers=config.n_layer,
        num_attention_heads=config.n_head,
        num_key_value_heads=config.num_key_value_heads,
        intermediate_size=4 * config.n_embd if config.n_inner is None else config.n_inner,
        hidden_act="silu",
        rms_norm_eps=config.layer_norm_epsilon,
        use_cache=config.use_cache,
        attention_bias=config.add_bias,
        tie_word_embeddings=config.tie_word_embeddings,
        initializer_range=config.initializer_range,
        rope_theta=config.rope_theta,
        rope_scaling=config.rope_scaling,
        attention_dropout=config.attn_pdrop,
    )

    return original_config


def _export_state_dict_to_huggingface(
    safetensors_weight_manager: SafeTensorsWeightsManager,
    num_layers: int,
    num_heads: int,
    num_key_value_heads: int,
    head_dim: int,
    attention_head_type: AttentionHeadType,
) -> dict:
    state_dict = {
        "model.embed_tokens.weight": safetensors_weight_manager.get_tensor("transformer.wte.weight"),
        "model.norm.weight": safetensors_weight_manager.get_tensor("transformer.ln_f.weight"),
    }

    if safetensors_weight_manager.has_tensor("lm_head.weight"):
        state_dict["lm_head.weight"] = safetensors_weight_manager.get_tensor("lm_head.weight")

    for layer_idx in range(num_layers):
        state_dict[f"model.layers.{layer_idx}.input_layernorm.weight"] = safetensors_weight_manager.get_tensor(
            f"transformer.h.{layer_idx}.ln_1.weight"
        )
        state_dict[
            f"model.layers.{layer_idx}.post_attention_layernorm.weight"
        ] = safetensors_weight_manager.get_tensor(f"transformer.h.{layer_idx}.ln_2.weight")

        up_weight, gate_weight = split_up_gate_tensor_for_mlp(
            safetensors_weight_manager.get_tensor(f"transformer.h.{layer_idx}.mlp.c_fc.weight")
        )
        state_dict[f"model.layers.{layer_idx}.mlp.up_proj.weight"] = up_weight
        state_dict[f"model.layers.{layer_idx}.mlp.gate_proj.weight"] = gate_weight

        state_dict[f"model.layers.{layer_idx}.mlp.down_proj.weight"] = safetensors_weight_manager.get_tensor(
            f"transformer.h.{layer_idx}.mlp.c_proj.weight"
        )

        query_weight, key_weight, value_weight = split_query_key_value_tensor_for_attention(
            safetensors_weight_manager.get_tensor(f"transformer.h.{layer_idx}.attn.c_attn.weight"),
            num_heads,
            num_key_value_heads,
            head_dim,
            attention_head_type,
        )
        state_dict[f"model.layers.{layer_idx}.self_attn.q_proj.weight"] = query_weight
        state_dict[f"model.layers.{layer_idx}.self_attn.k_proj.weight"] = key_weight
        state_dict[f"model.layers.{layer_idx}.self_attn.v_proj.weight"] = value_weight

        state_dict[f"model.layers.{layer_idx}.self_attn.o_proj.weight"] = safetensors_weight_manager.get_tensor(
            f"transformer.h.{layer_idx}.attn.c_proj.weight"
        )

    return state_dict
