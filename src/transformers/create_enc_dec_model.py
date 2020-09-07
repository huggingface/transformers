#!/usr/bin/env python3

from transformers import EncoderDecoderModel

model = EncoderDecoderModel.from_encoder_decoder_pretrained("/home/patrick/hugging_face/roberta2roberta_L-24_cnn_daily_mail/bertseq2seq_decoder" , "/home/patrick/hugging_face/roberta2roberta_L-24_cnn_daily_mail/bertseq2seq_decoder", tie_encoder_decoder=True)
model.save_pretrained("/home/patrick/hugging_face/roberta2roberta_L-24_cnn_daily_mail/bertseq2seq_full")
