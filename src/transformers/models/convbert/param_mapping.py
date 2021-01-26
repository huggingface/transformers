# coding=utf-8
# Copyright 2021 The HuggingFace Inc. team. All rights reserved.
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
""" PyTorch ConvBERT model: param mapping """


def fetch_mapping(config):
    param_mapping = {
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
        param_mapping[
            f"encoder.layer.{j}.attention.self.query.weight"
        ] = f"electra/encoder/layer_{j}/attention/self/query/kernel"
        param_mapping[
            f"encoder.layer.{j}.attention.self.query.bias"
        ] = f"electra/encoder/layer_{j}/attention/self/query/bias"
        param_mapping[
            f"encoder.layer.{j}.attention.self.key.weight"
        ] = f"electra/encoder/layer_{j}/attention/self/key/kernel"
        param_mapping[
            f"encoder.layer.{j}.attention.self.key.bias"
        ] = f"electra/encoder/layer_{j}/attention/self/key/bias"
        param_mapping[
            f"encoder.layer.{j}.attention.self.value.weight"
        ] = f"electra/encoder/layer_{j}/attention/self/value/kernel"
        param_mapping[
            f"encoder.layer.{j}.attention.self.value.bias"
        ] = f"electra/encoder/layer_{j}/attention/self/value/bias"
        param_mapping[
            f"encoder.layer.{j}.attention.self.key_conv_attn_layer.depthwise.weight"
        ] = f"electra/encoder/layer_{j}/attention/self/conv_attn_key/depthwise_kernel"
        param_mapping[
            f"encoder.layer.{j}.attention.self.key_conv_attn_layer.pointwise.weight"
        ] = f"electra/encoder/layer_{j}/attention/self/conv_attn_key/pointwise_kernel"
        param_mapping[
            f"encoder.layer.{j}.attention.self.key_conv_attn_layer.bias"
        ] = f"electra/encoder/layer_{j}/attention/self/conv_attn_key/bias"
        param_mapping[
            f"encoder.layer.{j}.attention.self.conv_kernel_layer.weight"
        ] = f"electra/encoder/layer_{j}/attention/self/conv_attn_kernel/kernel"
        param_mapping[
            f"encoder.layer.{j}.attention.self.conv_kernel_layer.bias"
        ] = f"electra/encoder/layer_{j}/attention/self/conv_attn_kernel/bias"
        param_mapping[
            f"encoder.layer.{j}.attention.self.conv_out_layer.weight"
        ] = f"electra/encoder/layer_{j}/attention/self/conv_attn_point/kernel"
        param_mapping[
            f"encoder.layer.{j}.attention.self.conv_out_layer.bias"
        ] = f"electra/encoder/layer_{j}/attention/self/conv_attn_point/bias"
        param_mapping[
            f"encoder.layer.{j}.attention.output.dense.weight"
        ] = f"electra/encoder/layer_{j}/attention/output/dense/kernel"
        param_mapping[
            f"encoder.layer.{j}.attention.output.LayerNorm.weight"
        ] = f"electra/encoder/layer_{j}/attention/output/LayerNorm/gamma"
        param_mapping[
            f"encoder.layer.{j}.attention.output.dense.bias"
        ] = f"electra/encoder/layer_{j}/attention/output/dense/bias"
        param_mapping[
            f"encoder.layer.{j}.attention.output.LayerNorm.bias"
        ] = f"electra/encoder/layer_{j}/attention/output/LayerNorm/beta"
        param_mapping[
            f"encoder.layer.{j}.intermediate.dense.weight"
        ] = f"electra/encoder/layer_{j}/intermediate/{group_dense_name}/kernel"
        param_mapping[
            f"encoder.layer.{j}.intermediate.dense.bias"
        ] = f"electra/encoder/layer_{j}/intermediate/{group_dense_name}/bias"
        param_mapping[
            f"encoder.layer.{j}.output.dense.weight"
        ] = f"electra/encoder/layer_{j}/output/{group_dense_name}/kernel"
        param_mapping[
            f"encoder.layer.{j}.output.dense.bias"
        ] = f"electra/encoder/layer_{j}/output/{group_dense_name}/bias"
        param_mapping[
            f"encoder.layer.{j}.output.LayerNorm.weight"
        ] = f"electra/encoder/layer_{j}/output/LayerNorm/gamma"
        param_mapping[f"encoder.layer.{j}.output.LayerNorm.bias"] = f"electra/encoder/layer_{j}/output/LayerNorm/beta"

    return param_mapping
