#!/usr/bin/env python
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

# this script builds a small sample spm file tests/fixtures/test_sentencepiece_no_bos.model, with features needed by pegasus

# 1. pip install sentencepiece
#
# 2. wget https://raw.githubusercontent.com/google/sentencepiece/master/data/botchan.txt

# 3. build
import sentencepiece as spm


# pegasus:
# 1. no bos
# 2. eos_id is 1
# 3. unk_id is 2
# build a sample spm file accordingly
spm.SentencePieceTrainer.train('--input=botchan.txt --model_prefix=test_sentencepiece_no_bos --bos_id=-1 --unk_id=2  --eos_id=1  --vocab_size=1000')

# 4. now update the fixture
# mv test_sentencepiece_no_bos.model ../../tests/fixtures/
