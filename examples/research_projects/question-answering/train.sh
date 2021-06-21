python3 run_question_answering_flax.py \
--train_file= \
--validation_file= \
--model_name_or_path=google/bigbird-roberta-base \
--per_device_train_batch_size=1 \
--per_device_eval_batch_size=2 \
--gradient_accumulation_steps=8 \
--lr1=5.e-5 \
--lr2=1.e-4 \
--block_size=128
