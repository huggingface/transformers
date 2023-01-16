# coding=utf-8
# Copyright 2022 The HuggingFace Inc. team. All rights reserved.
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
""" Informer model configuration"""

from typing import List, Optional

from ...configuration_utils import PretrainedConfig
from ...utils import logging


logger = logging.get_logger(__name__)

TIME_SERIES_TRANSFORMER_PRETRAINED_CONFIG_ARCHIVE_MAP = {
    "huggingface/time-series-transformer-tourism-monthly": (
        "https://huggingface.co/huggingface/time-series-transformer-tourism-monthly/resolve/main/config.json"
    ),
    # See all TimeSeriesTransformer models at https://huggingface.co/models?filter=time_series_transformer
}


class InformerConfig(PretrainedConfig):
    def __init__(
        self,
        context_length: int,
        prediction_length: int,
        num_feat_dynamic_real: int,  # num_dynamic_real_features
        num_feat_static_real: int,  # num_static_real_features
        num_feat_static_cat: int,  # num_static_categorical_features
        cardinality: List[int],
        # Informer arguments
        nhead: int,
        num_encoder_layers: int, # encoder_layers
        num_decoder_layers: int, # decoder_layers
        dim_feedforward: int,
        activation: str = "gelu", # activation_function
        dropout: float = 0.1,
        attn: str = "prob",
        factor: int = 5,
        distil: bool = True,
        # univariate input
        input_size: int = 1,
        embedding_dimension: Optional[List[int]] = None,
        distr_output: str = "student_t",
        lags_seq: Optional[List[int]] = None, # used to be freq.
        scaling: bool = True,
        num_parallel_samples: int = 100,
        is_encoder_decoder: bool = True,
    ):
        # time series specific configuration
        self.prediction_length = prediction_length
        self.context_length = context_length or prediction_length
        self.distr_output = distr_output # Eli: change to distribution_output
        # self.loss = loss # Eli: From vanilla ts transformer
        self.input_size = input_size
        # self.target_shape = distr_output.event_shape  # Eli: I think can be removed
        # self.num_time_features = num_time_features # Eli: From vanilla ts transformer
        self.lags_seq = lags_seq
        # self.scaling = scaling # Eli: From vanilla ts transformer
        self.num_feat_dynamic_real = num_feat_dynamic_real
        self.num_feat_static_cat = num_feat_static_cat
        self.num_feat_static_real = num_feat_static_real

        # set cardinality
        if cardinality and num_feat_static_cat > 0:
            if len(cardinality) != num_feat_static_cat:
                raise ValueError(
                    "The cardinality should be a list of the same length as `num_static_categorical_features`"
                )
            self.cardinality = cardinality
        else:
            self.cardinality = [1]

        # set embedding_dimension
        if embedding_dimension and num_feat_static_cat > 0:
            if len(embedding_dimension) != num_feat_static_cat:
                raise ValueError(
                    "The embedding dimension should be a list of the same length as `num_static_categorical_features`"
                )
            self.embedding_dimension = embedding_dimension
        else:
            self.embedding_dimension = [min(50, (cat + 1) // 2) for cat in self.cardinality]
            
        self.num_parallel_samples = num_parallel_samples

        # self.history_length = context_length + max(self.lags_seq)
        self.embedder = FeatureEmbedder(
            cardinalities=cardinality,
            embedding_dims=self.embedding_dimension,
        )
        if scaling:
            self.scaler = MeanScaler(dim=1, keepdim=True)
        else:
            self.scaler = NOPScaler(dim=1, keepdim=True)

        # total feature size
        d_model = self.input_size * len(self.lags_seq) + self._number_of_features

        self.param_proj = distr_output.get_args_proj(d_model)

        super().__init__(is_encoder_decoder=is_encoder_decoder, **kwargs)
