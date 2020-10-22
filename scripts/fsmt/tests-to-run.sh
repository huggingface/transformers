#!/usr/bin/env bash

# these scripts need to be run before any changes to FSMT-related code - it should cover all bases

CUDA_VISIBLE_DEVICES="" RUN_SLOW=1 pytest --disable-warnings tests/test_tokenization_fsmt.py tests/test_configuration_auto.py tests/test_modeling_fsmt.py examples/seq2seq/test_fsmt_bleu_score.py
RUN_SLOW=1 pytest --disable-warnings tests/test_tokenization_fsmt.py tests/test_configuration_auto.py tests/test_modeling_fsmt.py examples/seq2seq/test_fsmt_bleu_score.py
