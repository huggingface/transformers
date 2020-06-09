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
    "dpr-base-uncased": "https://s3.amazonaws.com/models.huggingface.co/bert/dpr-base-uncased-config.json",
    "dpr-large-uncased": "https://s3.amazonaws.com/models.huggingface.co/bert/dpr-large-uncased-config.json",
}


class DprConfig(PretrainedConfig):
    r"""
        :class:`~transformers.DprConfig` is the configuration class to store the configuration of a
        `DprModel`.


        Arguments:
            k: number of documents to retrieve.
    """
    model_type = "dpr"

    def __init__(
        self,
        k: int = 16,
        pretrained_model_cfg: str = "bert-base-uncased",
        projection_dim: int = 0,
        sequence_length: int = 512,
        do_lower_case: bool = True,
        biencoder_model_file: Optional[str] = None,
        reader_model_file: Optional[str] = None,
        pad_id: int = 0,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.k = k
        self.pretrained_model_cfg = pretrained_model_cfg
        self.projection_dim = projection_dim
        self.sequence_length = sequence_length
        self.do_lower_case = do_lower_case
        self.biencoder_model_file = biencoder_model_file
        self.reader_model_file = reader_model_file
        self.pad_id = pad_id
