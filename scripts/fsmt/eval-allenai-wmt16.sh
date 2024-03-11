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

# this script evals the following fsmt models
# it covers:
# - allenai/wmt16-en-de-dist-12-1
# - allenai/wmt16-en-de-dist-6-1
# - allenai/wmt16-en-de-12-1

# this script needs to be run from the top level of the transformers repo
if [ ! -d "src/transformers" ]; then
    echo "Error: This script needs to be run from the top of the transformers repo"
    exit 1
fi

# In these scripts you may have to lower BS if you get CUDA OOM (or increase it if you have a large GPU)

### Normal eval ###

export PAIR=en-de
export DATA_DIR=data/$PAIR
export SAVE_DIR=data/$PAIR
export BS=64
export NUM_BEAMS=5
mkdir -p $DATA_DIR
sacrebleu -t wmt19 -l $PAIR --echo src > $DATA_DIR/val.source
sacrebleu -t wmt19 -l $PAIR --echo ref > $DATA_DIR/val.target

MODEL_PATH=allenai/wmt16-en-de-dist-12-1
echo $PAIR $MODEL_PATH
PYTHONPATH="src:examples/seq2seq" python examples/seq2seq/run_eval.py $MODEL_PATH $DATA_DIR/val.source $SAVE_DIR/test_translations.txt --reference_path $DATA_DIR/val.target --score_path $SAVE_DIR/test_bleu.json --bs $BS --task translation --num_beams $NUM_BEAMS

MODEL_PATH=allenai/wmt16-en-de-dist-6-1
echo $PAIR $MODEL_PATH
PYTHONPATH="src:examples/seq2seq" python examples/seq2seq/run_eval.py $MODEL_PATH $DATA_DIR/val.source $SAVE_DIR/test_translations.txt --reference_path $DATA_DIR/val.target --score_path $SAVE_DIR/test_bleu.json --bs $BS --task translation --num_beams $NUM_BEAMS

MODEL_PATH=allenai/wmt16-en-de-12-1
echo $PAIR $MODEL_PATH
PYTHONPATH="src:examples/seq2seq" python examples/seq2seq/run_eval.py $MODEL_PATH $DATA_DIR/val.source $SAVE_DIR/test_translations.txt --reference_path $DATA_DIR/val.target --score_path $SAVE_DIR/test_bleu.json --bs $BS --task translation --num_beams $NUM_BEAMS



### Searching hparams eval ###


export PAIR=en-de
export DATA_DIR=data/$PAIR
export SAVE_DIR=data/$PAIR
export BS=32
export NUM_BEAMS=5
mkdir -p $DATA_DIR
sacrebleu -t wmt19 -l $PAIR --echo src > $DATA_DIR/val.source
sacrebleu -t wmt19 -l $PAIR --echo ref > $DATA_DIR/val.target

MODEL_PATH=allenai/wmt16-en-de-dist-12-1
echo $PAIR $MODEL_PATH
PYTHONPATH="src:examples/seq2seq" python examples/seq2seq/run_eval_search.py $MODEL_PATH $DATA_DIR/val.source $SAVE_DIR/test_translations.txt --reference_path $DATA_DIR/val.target --score_path $SAVE_DIR/test_bleu.json --bs $BS --task translation --search="num_beams=5:10:15 length_penalty=0.6:0.7:0.8:0.9:1.0:1.1"


MODEL_PATH=allenai/wmt16-en-de-dist-6-1
echo $PAIR $MODEL_PATH
PYTHONPATH="src:examples/seq2seq" python examples/seq2seq/run_eval_search.py $MODEL_PATH $DATA_DIR/val.source $SAVE_DIR/test_translations.txt --reference_path $DATA_DIR/val.target --score_path $SAVE_DIR/test_bleu.json --bs $BS --task translation --search="num_beams=5:10:15 length_penalty=0.6:0.7:0.8:0.9:1.0:1.1"


MODEL_PATH=allenai/wmt16-en-de-12-1
echo $PAIR $MODEL_PATH
PYTHONPATH="src:examples/seq2seq" python examples/seq2seq/run_eval_search.py $MODEL_PATH $DATA_DIR/val.source $SAVE_DIR/test_translations.txt --reference_path $DATA_DIR/val.target --score_path $SAVE_DIR/test_bleu.json --bs $BS --task translation --search="num_beams=5:10:15 length_penalty=0.6:0.7:0.8:0.9:1.0:1.1"
