#!/usr/bin/env python

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
