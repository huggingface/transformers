import shutil

from transformers import AutoConfig, AutoTokenizer, GPTBigCodeConfig

from ..config import GPTMegatronConfig
from ..enums import AttentionHeadType, PositionEmbeddingType


def import_from_huggingface_starcoder(pretrained_model_name_or_path: str, save_path: str) -> None:
    shutil.copytree(pretrained_model_name_or_path, save_path)

    original_config: GPTBigCodeConfig = AutoConfig.from_pretrained(save_path)
    config = _import_config_from_huggingface(original_config)
    config.save_pretrained(save_path)

    try:
        tokenizer = AutoTokenizer.from_pretrained(pretrained_model_name_or_path)
        tokenizer.save_pretrained(save_path)
    except:
        pass


def _import_config_from_huggingface(original_config: GPTBigCodeConfig) -> GPTMegatronConfig:
    assert original_config.activation_function == "gelu_pytorch_tanh"

    config = GPTMegatronConfig(
        vocab_size=original_config.vocab_size,
        n_positions=original_config.n_positions,
        n_embd=original_config.n_embd,
        n_layer=original_config.n_layer,
        n_head=original_config.n_head,
        attention_head_type="mqa" if original_config.multi_query else "mha",
        position_embedding_type="learned_absolute",
        n_inner=original_config.n_inner,
        activation_function=original_config.activation_function,
        normalization_function="layernorm",
        layer_norm_epsilon=original_config.layer_norm_epsilon,
        use_cache=original_config.use_cache,
        add_bias=True,
        tie_word_embeddings=original_config.tie_word_embeddings,
        initializer_range=original_config.initializer_range,
        attn_pdrop=original_config.attn_pdrop,
        resid_pdrop=original_config.resid_pdrop,
    )

    return config


def export_to_huggingface_starcoder(pretrained_model_name_or_path: str, save_path: str) -> None:
    shutil.copytree(pretrained_model_name_or_path, save_path)

    config: GPTMegatronConfig = AutoConfig.from_pretrained(save_path)
    original_config = _export_config_to_huggingface(config)
    original_config.save_pretrained(save_path)

    try:
        tokenizer = AutoTokenizer.from_pretrained(pretrained_model_name_or_path)
        tokenizer.save_pretrained(save_path)
    except:
        pass


def _export_config_to_huggingface(config: GPTMegatronConfig) -> GPTBigCodeConfig:
    assert config.activation_function == "gelu_pytorch_tanh"
    assert config.normalization_function == "layernorm"
    assert AttentionHeadType(config.attention_head_type) in [AttentionHeadType.mha, AttentionHeadType.mqa]
    assert PositionEmbeddingType(config.position_embedding_type) == PositionEmbeddingType.learned_absolute

    original_config = GPTBigCodeConfig(
        vocab_size=config.vocab_size,
        n_positions=config.n_positions,
        n_embd=config.n_embd,
        n_layer=config.n_layer,
        n_head=config.n_head,
        n_inner=config.n_inner,
        activation_function=config.activation_function,
        resid_pdrop=config.resid_pdrop,
        embd_pdrop=config.embd_pdrop,
        attn_pdrop=config.attn_pdrop,
        layer_norm_epsilon=config.layer_norm_epsilon,
        initializer_range=config.initializer_range,
        scale_attn_weights=config.scale_attn_weights,
        use_cache=config.use_cache,
        bos_token_id=config.bos_token_id,
        eos_token_id=config.eos_token_id,
        attention_softmax_in_fp32=config.attention_softmax_in_fp32,
        scale_attention_softmax_in_fp32=config.scale_attention_softmax_in_fp32,
        multi_query=config.multi_query,
        tie_word_embeddings=config.tie_word_embeddings,
    )

    return original_config
