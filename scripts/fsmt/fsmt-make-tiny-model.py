#!/usr/bin/env python
# coding: utf-8

# this script creates a tiny model that is useful inside tests, when we just want to test that the machinery works,
# without needing to the check the quality of the outcomes.
# it will be used then as "stas/tiny-wmt19-en-de"

from transformers import FSMTTokenizer, FSMTConfig, FSMTForConditionalGeneration
mname = "facebook/wmt19-en-de"
tokenizer = FSMTTokenizer.from_pretrained(mname)
# get the correct vocab sizes, etc. from the master model
config = FSMTConfig.from_pretrained(mname)
config.update(dict(
    d_model=4,
    encoder_layers=1, decoder_layers=1,
    encoder_ffn_dim=4, decoder_ffn_dim=4,
    encoder_attention_heads=1, decoder_attention_heads=1))

tiny_model = FSMTForConditionalGeneration(config)
print(f"num of params {tiny_model.num_parameters()}")
# Test it
batch = tokenizer.prepare_seq2seq_batch(["Making tiny model"])
outputs = tiny_model(**batch, return_dict=True)

print(len(outputs.logits[0]))
# Save
mname_tiny = "tiny-wmt19-en-de"
tiny_model.half() # makes it smaller
tiny_model.save_pretrained(mname_tiny)
tokenizer.save_pretrained(mname_tiny)

# Upload
# transformers-cli upload tiny-wmt19-en-de
