# coding=utf-8
# Copyright 2010, DPR authors
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
""" DPR model configuration """

from .configuration_bert import BertConfig
from .utils import logging


logger = logging.get_logger(__name__)

DPR_PRETRAINED_CONFIG_ARCHIVE_MAP = {
    "facebook/dpr-ctx_encoder-single-nq-base": "https://s3.amazonaws.com/models.huggingface.co/bert/facebook/dpr-ctx_encoder-single-nq-base/config.json",
    "facebook/dpr-question_encoder-single-nq-base": "https://s3.amazonaws.com/models.huggingface.co/bert/facebook/dpr-question_encoder-single-nq-base/config.json",
    "facebook/dpr-reader-single-nq-base": "https://s3.amazonaws.com/models.huggingface.co/bert/facebook/dpr-reader-single-nq-base/config.json",
}


class DPRConfig(BertConfig):
    r"""
    :class:`~transformers.DPRConfig` is the configuration class to store the configuration of a
    `DPRModel`.

    This is the configuration class to store the configuration of a `DPRContextEncoder`, `DPRQuestionEncoder`, or a `DPRReader`.
    It is used to instantiate the components of the DPR model.

    Args:
        projection_dim (:obj:`int`, optional, defaults to 0):
            Dimension of the projection for the context and question encoders.
            If it is set to zero (default), then no projection is done.
    """
    model_type = "dpr"

    def __init__(self, projection_dim: int = 0, **kwargs):  # projection of the encoders, 0 for no projection
        super().__init__(**kwargs)
        self.projection_dim = projection_dim
