#Ref https://github.com/Morizeyao/GPT2-Chinese
python pregenerate_training_data.py \
	--train_corpus corpus.txt \
	--bert_model bert-base-chinese \
	--do_lower_case \
	--output_dir training/ \
	--epochs_to_generate 3 \
	--max_seq_len 256
