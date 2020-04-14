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
    "Wemb": "word_embeddings.weight",
    "Wpos": "position_embeddings.weight",
    "Wtype": "token_type_embeddings.weight",
    # "encoder_emb_ln_scale_pre" = "bert/embeddings/LayerNorm/gamma:0"
    # "encoder_emb_ln_bias_pre" = "bert/embeddings/LayerNorm/beta:0"

}
