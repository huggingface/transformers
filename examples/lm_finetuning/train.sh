CUDA_VISIBLE_DEVICES=1 python finetune_on_pregenerated.py \
	--pregenerated_data training/ \
	--bert_model bert-base-chinese \
	--do_lower_case \
	--output_dir model/ \
	--epochs 3 \
	--train_batch_size 1
