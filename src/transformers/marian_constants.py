NEEDS_TRANSPOSE = {
    "self_Wq": True,
    "self_Wk": True,
    "self_Wv": True,
    "self_bq": True,
    "self_bk": True,
    "self_bv": True,
    "self_Wo": True,
    "self_bo": True,
    "self_Wo_ln_scale": True,
    "self_Wo_ln_bias": True,
    "ffn_W1": True,
    "ffn_b1": True,
    "ffn_W2": True,
    "ffn_b2": True,
    "ffn_ffn_ln_scale": True,
    "ffn_ffn_ln_bias": True,
}
BERT_LAYER_CONVERTER = {
    "self_Wq": "attention.self.query.weight",
    "self_Wk": "attention.self.key.weight",
    "self_Wv": "attention.self.value.weight",
    "self_Wo": "attention.output.dense.weight",
    "self_bq": "attention.self.query.bias",
    "self_bk": "attention.self.key.bias",
    "self_bv": "attention.self.value.bias",
    "self_bo": "attention.output.dense.bias",
    "self_Wo_ln_scale": "attention.output.LayerNorm.weight",
    "self_Wo_ln_bias": "attention.output.LayerNorm.bias",
    "ffn_W1": "intermediate.dense.weight",
    "ffn_b1": "intermediate.dense.bias",
    "ffn_W2": "output.dense.weight",
    "ffn_b2": "output.dense.bias",
    "ffn_ffn_ln_scale": "output.LayerNorm.weight",
    "ffn_ffn_ln_bias": "output.LayerNorm.bias",
    # Decoder Cross Attention
    "context_Wk": "crossattention.self.key.weight",
    "context_Wo": "crossattention.output.dense.weight",
    "context_Wq": "crossattention.self.query.weight",
    "context_Wv": "crossattention.self.value.weight",
    "context_bk": "crossattention.self.key.bias",
    "context_bo": "crossattention.output.dense.bias",
    "context_bq": "crossattention.self.query.bias",
    "context_bv": "crossattention.self.value.bias",
    "context_Wo_ln_bias": "crossattention.output.LayerNorm.weight",
    "context_Wo_ln_scale": "crossattention.output.LayerNorm.bias",
}

EMBED_CONVERTER = {
    "Wemb": "embed_tokens.weight",
    "Wpos": "embed_positions.weight",
    # "Wtype": "token_type_embeddings.weight",
    "encoder_emb_ln_scale_pre": "encoder.layernorm_embedding.weight",
    "encoder_emb_ln_bias_pre": "encoder.layernorm_embedding.bias",
    "decoder_emb_ln_scale_pre": "decoder.layernorm_embedding.weight",
    "decoder_emb_ln_bias_pre": "decoder.layernorm_embedding.bias",
}

EN_DE_CONFIG = {
    "bert-train-type-embeddings": "true",
    "bert-type-vocab-size": "2",
    "dec-cell": "gru",
    "dec-cell-base-depth": "2",
    "dec-cell-high-depth": "1",
    "dec-depth": 6,
    "dim-emb": "512",
    "dim-rnn": "1024",  # IGNORE
    "dim-vocabs": ["58100", "58100"],
    "enc-cell": "gru",  # IGNORE
    "enc-cell-depth": "1",
    "enc-depth": 6,
    "enc-type": "bidirectional",
    "input-types": [],
    "layer-normalization": "false",
    "lemma-dim-emb": "0",
    "right-left": "false",
    "skip": "false",
    "tied-embeddings": "false",
    "tied-embeddings-all": "true",  # "Tie all embedding layers and output layer"
    "tied-embeddings-src": "false",
    ## FFN and AAN params identical
    "transformer-aan-activation": "swish",
    "transformer-aan-depth": "2",  # What does AAN stand for?
    "transformer-aan-nogate": "false",
    "transformer-decoder-autoreg": "self-attention",
    "transformer-dim-aan": "2048",
    "transformer-dim-ffn": "2048",
    "transformer-ffn-activation": "swish",
    "transformer-ffn-depth": "2",
    "transformer-guided-alignment-layer": "last",
    "transformer-heads": 8,
    "transformer-no-projection": "false",  # Omit linear projection after multi-head attention (transformer)
    "transformer-postprocess": "dan",  # Dropout, add, normalize
    "transformer-postprocess-emb": "d",  # Operation after transformer embedding layer: d = dropout, a = add, n = normalize
    "transformer-preprocess": "",  # Operation before each transformer layer: d = dropout, a = add, n = normalize
    "transformer-tied-layers": [],
    "transformer-train-position-embeddings": "false",  # Train positional embeddings instead of using static sinusoidal embeddings
    "type": "transformer",
    "ulr": "false",
    "ulr-dim-emb": "0",
    "ulr-trainable-transformation": "false",
    "version": "v1.8.2 2111c28 2019-10-16 08:36:48 -0700",
}

BART_CONVERTER = {
    "self_Wq": "self_attn.q_proj.weight",
    "self_Wk": "self_attn.k_proj.weight",
    "self_Wv": "self_attn.v_proj.weight",
    "self_Wo": "self_attn.out_proj.weight",
    "self_bq": "self_attn.q_proj.bias",
    "self_bk": "self_attn.k_proj.bias",
    "self_bv": "self_attn.v_proj.bias",
    "self_bo": "self_attn.out_proj.bias",
    "self_Wo_ln_scale": "self_attn_layer_norm.weight",
    "self_Wo_ln_bias": "self_attn_layer_norm.bias",
    "ffn_W1": "fc1.weight",
    "ffn_b1": "fc1.bias",
    "ffn_W2": "fc2.weight",
    "ffn_b2": "fc2.bias",
    "ffn_ffn_ln_scale": "final_layer_norm.weight",
    "ffn_ffn_ln_bias": "final_layer_norm.bias",
    # Decoder Cross Attention
    "context_Wk": "encoder_attn.k_proj.weight",
    "context_Wo": "encoder_attn.out_proj.weight",
    "context_Wq": "encoder_attn.q_proj.weight",
    "context_Wv": "encoder_attn.v_proj.weight",
    "context_bk": "encoder_attn.k_proj.bias",
    "context_bo": "encoder_attn.out_proj.bias",
    "context_bq": "encoder_attn.q_proj.bias",
    "context_bv": "encoder_attn.v_proj.bias",
    "context_Wo_ln_bias": "encoder_attn_layer_norm.weight",
    "context_Wo_ln_scale": "encoder_attn_layer_norm.bias",
}

from transformers import BartConfig, BertConfig


# vocab_size = vocab_size
def convert_bert_config_to_bart(cfg: BertConfig) -> BartConfig:
    # dict(vocab_size=cfg.vocab_size,
    #      hidden_size = cfg.d_model,
    #     num_hidden_layers =decoder_layers,
    #     num_attention_heads = decoder_attention_heads,
    #     #int(marian_cfg["transformer-heads"]),  # getattr(yaml_cfg, 'transformer-heads', 'swish'),
    #     intermediate_size = intermediate_shape,
    #     activation_function=hidden_act

    bart_cfg = BartConfig(
        vocab_size=cfg.vocab_size,
        decoder_layers=cfg.num_hidden_layers,
        decoder_attention_heads=cfg.num_attention_heads,
        decoder_ffn_dim=cfg.intermediate_size,
        d_model=cfg.hidden_size,
        pad_token_id=cfg.pad_token_id,
        eos_token_id=cfg.eos_token_id,
        bos_token_id=None,
        max_position_embeddings=cfg.max_position_embeddings,
        activation_function=cfg.hidden_act,
        scale_embedding=True,
    )
    return bart_cfg


assume_vals = {
    "tied-embeddings-all": True,
    "layer-normalization": False,
    "right-left": False,
    "transformer-ffn-depth": 2,
    "transformer-aan-depth": 2,
    "transformer-no-projection": False,
    "transformer-postprocess-emb": "d",
    "transformer-postprocess": "dan",  # Dropout, add, normalize
    "transformer-preprocess": "",
    "type": "transformer",
    "ulr-dim-emb": 0,
    "dec-cell-base-depth": 2,
    "dec-cell-high-depth": 1,
    "transformer-aan-nogate": False,
}
