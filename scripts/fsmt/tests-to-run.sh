#!/usr/bin/env bash
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

# these scripts need to be run before any changes to FSMT-related code - it should cover all bases

CUDA_VISIBLE_DEVICES="" RUN_SLOW=1 pytest --disable-warnings tests/test_tokenization_fsmt.py tests/test_configuration_auto.py tests/test_modeling_fsmt.py examples/seq2seq/test_fsmt_bleu_score.py
RUN_SLOW=1 pytest --disable-warnings tests/test_tokenization_fsmt.py tests/test_configuration_auto.py tests/test_modeling_fsmt.py examples/seq2seq/test_fsmt_bleu_score.py
