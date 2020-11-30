def fetch_mapping(config):
    PARAM_MAPPING = {
        "embeddings.word_embeddings.weight": "electra/embeddings/word_embeddings",
        "embeddings.position_embeddings.weight": "electra/embeddings/position_embeddings",
        "embeddings.token_type_embeddings.weight": "electra/embeddings/token_type_embeddings",
        "embeddings.LayerNorm.weight": "electra/embeddings/LayerNorm/gamma",
        "embeddings.LayerNorm.bias": "electra/embeddings/LayerNorm/beta",
        "embeddings_project.weight": "electra/embeddings_project/kernel",
        "embeddings_project.bias": "electra/embeddings_project/bias",
    }
    if config.num_groups > 1:
        group_dense_name = "g_dense"
    else:
        group_dense_name = "dense"

    for j in range(config.num_hidden_layers):
        PARAM_MAPPING[
            f"encoder.layer.{j}.attention.self.query.weight"
        ] = f"electra/encoder/layer_{j}/attention/self/query/kernel"
        PARAM_MAPPING[
            f"encoder.layer.{j}.attention.self.query.bias"
        ] = f"electra/encoder/layer_{j}/attention/self/query/bias"
        PARAM_MAPPING[
            f"encoder.layer.{j}.attention.self.key.weight"
        ] = f"electra/encoder/layer_{j}/attention/self/key/kernel"
        PARAM_MAPPING[
            f"encoder.layer.{j}.attention.self.key.bias"
        ] = f"electra/encoder/layer_{j}/attention/self/key/bias"
        PARAM_MAPPING[
            f"encoder.layer.{j}.attention.self.value.weight"
        ] = f"electra/encoder/layer_{j}/attention/self/value/kernel"
        PARAM_MAPPING[
            f"encoder.layer.{j}.attention.self.value.bias"
        ] = f"electra/encoder/layer_{j}/attention/self/value/bias"
        PARAM_MAPPING[
            f"encoder.layer.{j}.attention.self.key_conv_attn_layer.depthwise.weight"
        ] = f"electra/encoder/layer_{j}/attention/self/conv_attn_key/depthwise_kernel"
        PARAM_MAPPING[
            f"encoder.layer.{j}.attention.self.key_conv_attn_layer.pointwise.weight"
        ] = f"electra/encoder/layer_{j}/attention/self/conv_attn_key/pointwise_kernel"
        PARAM_MAPPING[
            f"encoder.layer.{j}.attention.self.key_conv_attn_layer.bias"
        ] = f"electra/encoder/layer_{j}/attention/self/conv_attn_key/bias"
        PARAM_MAPPING[
            f"encoder.layer.{j}.attention.self.conv_kernel_layer.weight"
        ] = f"electra/encoder/layer_{j}/attention/self/conv_attn_kernel/kernel"
        PARAM_MAPPING[
            f"encoder.layer.{j}.attention.self.conv_kernel_layer.bias"
        ] = f"electra/encoder/layer_{j}/attention/self/conv_attn_kernel/bias"
        PARAM_MAPPING[
            f"encoder.layer.{j}.attention.self.conv_out_layer.weight"
        ] = f"electra/encoder/layer_{j}/attention/self/conv_attn_point/kernel"
        PARAM_MAPPING[
            f"encoder.layer.{j}.attention.self.conv_out_layer.bias"
        ] = f"electra/encoder/layer_{j}/attention/self/conv_attn_point/bias"
        PARAM_MAPPING[
            f"encoder.layer.{j}.attention.output.dense.weight"
        ] = f"electra/encoder/layer_{j}/attention/output/dense/kernel"
        PARAM_MAPPING[
            f"encoder.layer.{j}.attention.output.LayerNorm.weight"
        ] = f"electra/encoder/layer_{j}/attention/output/LayerNorm/gamma"
        PARAM_MAPPING[
            f"encoder.layer.{j}.attention.output.dense.bias"
        ] = f"electra/encoder/layer_{j}/attention/output/dense/bias"
        PARAM_MAPPING[
            f"encoder.layer.{j}.attention.output.LayerNorm.bias"
        ] = f"electra/encoder/layer_{j}/attention/output/LayerNorm/beta"
        PARAM_MAPPING[
            f"encoder.layer.{j}.intermediate.dense.weight"
        ] = f"electra/encoder/layer_{j}/intermediate/{group_dense_name}/kernel"
        PARAM_MAPPING[
            f"encoder.layer.{j}.intermediate.dense.bias"
        ] = f"electra/encoder/layer_{j}/intermediate/{group_dense_name}/bias"
        PARAM_MAPPING[
            f"encoder.layer.{j}.output.dense.weight"
        ] = f"electra/encoder/layer_{j}/output/{group_dense_name}/kernel"
        PARAM_MAPPING[
            f"encoder.layer.{j}.output.dense.bias"
        ] = f"electra/encoder/layer_{j}/output/{group_dense_name}/bias"
        PARAM_MAPPING[
            f"encoder.layer.{j}.output.LayerNorm.weight"
        ] = f"electra/encoder/layer_{j}/output/LayerNorm/gamma"
        PARAM_MAPPING[f"encoder.layer.{j}.output.LayerNorm.bias"] = f"electra/encoder/layer_{j}/output/LayerNorm/beta"

    return PARAM_MAPPING
