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


import logging
from typing import Optional

from .configuration_utils import PretrainedConfig


logger = logging.getLogger(__name__)

DPR_PRETRAINED_CONFIG_ARCHIVE_MAP = {
    "facebook/dpr-ctx_encoder-single-nq-base": "https://s3.amazonaws.com/models.huggingface.co/bert/facebook/dpr-ctx_encoder-single-nq-base/config.json",
    "facebook/dpr-question_encoder-single-nq-base": "https://s3.amazonaws.com/models.huggingface.co/bert/facebook/dpr-question_encoder-single-nq-base/config.json",
    "facebook/dpr-reader-single-nq-base": "https://s3.amazonaws.com/models.huggingface.co/bert/facebook/dpr-reader-single-nq-base/config.json",
}


class DPRConfig(PretrainedConfig):
    r"""
        :class:`~transformers.DPRConfig` is the configuration class to store the configuration of a
        `DPRModel`.

        This is the configuration class to store the configuration of a `DPRContextEncoder`, `DPRQuestionEncoder`, or a `DPRReader`.
        It is used to instantiate the components of the DPR model.

        Args:
            pretrained_model_config_name (:obj:`str`, optional, defaults to bert-base-uncased):
                Configuration of the encoders inside each DPR component
            projection_dim (:obj:`int`, optional, defaults to 0):
                Dimension of the projection for the context and question encoders.
                If it is set to zero (default), then no projection is done.
            sequence_length (:obj:`int`, optional, defaults to 512):
                Maximum length of the sequence.
            bi_encoder_model_file (:obj:`str`, optional, defaults to None):
                If not None, load weights from files provided in the official DPR repository for the context
                and question encoders
            reader_model_file (:obj:`str`, optional, defaults to None):
                If not None, load weights from files provided in the official DPR repository for the context
                and question encoders
            pad_token_id (:obj:`int`, optional, defaults to 0):
                The id of the pad token. This is used in the reader to combine the different `input_ids`
    """
    model_type = "dpr"

    def __init__(
        self,
        pretrained_model_config_name: str = "bert-base-uncased",  # base config
        projection_dim: int = 0,  # projection of the encoders, 0 for no projection
        sequence_length: int = 512,
        bi_encoder_model_file: Optional[str] = None,  # load weights from official repo
        reader_model_file: Optional[str] = None,  # load weights from official repo
        pad_token_id: int = 0,
        **kwargs
    ):
        super().__init__(pad_token_id=pad_token_id, **kwargs)
        self.pretrained_model_config_name = pretrained_model_config_name
        self.projection_dim = projection_dim
        self.sequence_length = sequence_length
        self.bi_encoder_model_file = bi_encoder_model_file
        self.reader_model_file = reader_model_file
