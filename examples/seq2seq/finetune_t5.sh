# Add parent directory to python path to access lightning_base.py
export PYTHONPATH="../":"${PYTHONPATH}"

python finetune.py \
--data_dir=$CNN_DIR \
--model_name_or_path=t5-base \
--learning_rate=3e-5 \
--train_batch_size=$BS \
--eval_batch_size=$BS \
--output_dir=$OUTPUT_DIR \
--max_source_length=512 \
--freeze_encoder --freeze_embeds \
--do_train --do_predict \
 $@
