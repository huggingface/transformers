
# this is the process of uploading the updated models to s3. As I can't upload them directly to the correct orgs, this script shows how this is done
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

1. upload updated models to my account

transformers-cli upload -y wmt19-ru-en
transformers-cli upload -y wmt19-en-ru
transformers-cli upload -y wmt19-de-en
transformers-cli upload -y wmt19-en-de
transformers-cli upload -y wmt19-de-en-6-6-base
transformers-cli upload -y wmt19-de-en-6-6-big
transformers-cli upload -y wmt16-en-de-dist-12-1
transformers-cli upload -y wmt16-en-de-dist-6-1
transformers-cli upload -y wmt16-en-de-12-1


2. ask someone to move them to:

* to facebook: "wmt19-ru-en", "wmt19-en-ru", "wmt19-en-de", "wmt19-de-en"
* to allenai: "wmt16-en-de-dist-12-1", "wmt16-en-de-dist-6-1", "wmt16-en-de-12-1", "wmt19-de-en-6-6-base", "wmt19-de-en-6-6-big"

export b="s3://models.huggingface.co/bert"
stas_to_fb () {
	src=$1
	shift
	aws s3 sync $b/stas/$src $b/facebook/$src $@
}

stas_to_allenai () {
	src=$1
	shift
	aws s3 sync $b/stas/$src $b/allenai/$src $@
}

stas_to_fb wmt19-en-ru
stas_to_fb wmt19-ru-en
stas_to_fb wmt19-en-de
stas_to_fb wmt19-de-en

stas_to_allenai wmt16-en-de-dist-12-1
stas_to_allenai wmt16-en-de-dist-6-1
stas_to_allenai wmt16-en-de-6-1
stas_to_allenai wmt16-en-de-12-1
stas_to_allenai wmt19-de-en-6-6-base
stas_to_allenai wmt19-de-en-6-6-big


3. and then remove all these model files from my account

transformers-cli s3 rm wmt16-en-de-12-1/config.json
transformers-cli s3 rm wmt16-en-de-12-1/merges.txt
transformers-cli s3 rm wmt16-en-de-12-1/pytorch_model.bin
transformers-cli s3 rm wmt16-en-de-12-1/tokenizer_config.json
transformers-cli s3 rm wmt16-en-de-12-1/vocab-src.json
transformers-cli s3 rm wmt16-en-de-12-1/vocab-tgt.json
transformers-cli s3 rm wmt16-en-de-dist-12-1/config.json
transformers-cli s3 rm wmt16-en-de-dist-12-1/merges.txt
transformers-cli s3 rm wmt16-en-de-dist-12-1/pytorch_model.bin
transformers-cli s3 rm wmt16-en-de-dist-12-1/tokenizer_config.json
transformers-cli s3 rm wmt16-en-de-dist-12-1/vocab-src.json
transformers-cli s3 rm wmt16-en-de-dist-12-1/vocab-tgt.json
transformers-cli s3 rm wmt16-en-de-dist-6-1/config.json
transformers-cli s3 rm wmt16-en-de-dist-6-1/merges.txt
transformers-cli s3 rm wmt16-en-de-dist-6-1/pytorch_model.bin
transformers-cli s3 rm wmt16-en-de-dist-6-1/tokenizer_config.json
transformers-cli s3 rm wmt16-en-de-dist-6-1/vocab-src.json
transformers-cli s3 rm wmt16-en-de-dist-6-1/vocab-tgt.json
transformers-cli s3 rm wmt19-de-en-6-6-base/config.json
transformers-cli s3 rm wmt19-de-en-6-6-base/merges.txt
transformers-cli s3 rm wmt19-de-en-6-6-base/pytorch_model.bin
transformers-cli s3 rm wmt19-de-en-6-6-base/tokenizer_config.json
transformers-cli s3 rm wmt19-de-en-6-6-base/vocab-src.json
transformers-cli s3 rm wmt19-de-en-6-6-base/vocab-tgt.json
transformers-cli s3 rm wmt19-de-en-6-6-big/config.json
transformers-cli s3 rm wmt19-de-en-6-6-big/merges.txt
transformers-cli s3 rm wmt19-de-en-6-6-big/pytorch_model.bin
transformers-cli s3 rm wmt19-de-en-6-6-big/tokenizer_config.json
transformers-cli s3 rm wmt19-de-en-6-6-big/vocab-src.json
transformers-cli s3 rm wmt19-de-en-6-6-big/vocab-tgt.json
transformers-cli s3 rm wmt19-de-en/config.json
transformers-cli s3 rm wmt19-de-en/merges.txt
transformers-cli s3 rm wmt19-de-en/pytorch_model.bin
transformers-cli s3 rm wmt19-de-en/tokenizer_config.json
transformers-cli s3 rm wmt19-de-en/vocab-src.json
transformers-cli s3 rm wmt19-de-en/vocab-tgt.json
transformers-cli s3 rm wmt19-en-de/config.json
transformers-cli s3 rm wmt19-en-de/merges.txt
transformers-cli s3 rm wmt19-en-de/pytorch_model.bin
transformers-cli s3 rm wmt19-en-de/tokenizer_config.json
transformers-cli s3 rm wmt19-en-de/vocab-src.json
transformers-cli s3 rm wmt19-en-de/vocab-tgt.json
transformers-cli s3 rm wmt19-en-ru/config.json
transformers-cli s3 rm wmt19-en-ru/merges.txt
transformers-cli s3 rm wmt19-en-ru/pytorch_model.bin
transformers-cli s3 rm wmt19-en-ru/tokenizer_config.json
transformers-cli s3 rm wmt19-en-ru/vocab-src.json
transformers-cli s3 rm wmt19-en-ru/vocab-tgt.json
transformers-cli s3 rm wmt19-ru-en/config.json
transformers-cli s3 rm wmt19-ru-en/merges.txt
transformers-cli s3 rm wmt19-ru-en/pytorch_model.bin
transformers-cli s3 rm wmt19-ru-en/tokenizer_config.json
transformers-cli s3 rm wmt19-ru-en/vocab-src.json
transformers-cli s3 rm wmt19-ru-en/vocab-tgt.json
