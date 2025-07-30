# Copyright 2020 The HuggingFace Team. All rights reserved.
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

import unittest

from huggingface_hub import QuestionAnsweringOutputElement

from transformers import (
    MODEL_FOR_QUESTION_ANSWERING_MAPPING,
    TF_MODEL_FOR_QUESTION_ANSWERING_MAPPING,
    LxmertConfig,
    QuestionAnsweringPipeline,
)
from transformers.data.processors.squad import SquadExample
from transformers.pipelines import QuestionAnsweringArgumentHandler, pipeline
from transformers.testing_utils import (
    compare_pipeline_output_to_hub_spec,
    is_pipeline_test,
    is_torch_available,
    nested_simplify,
    require_torch,
    require_torch_or_tf,
    slow,
)


if is_torch_available():
    import torch

from .test_pipelines_common import ANY


# These 2 model types require different inputs than those of the usual text models.
_TO_SKIP = {"LayoutLMv2Config", "LayoutLMv3Config"}


@is_pipeline_test
class QAPipelineTests(unittest.TestCase):
    model_mapping = MODEL_FOR_QUESTION_ANSWERING_MAPPING
    tf_model_mapping = TF_MODEL_FOR_QUESTION_ANSWERING_MAPPING

