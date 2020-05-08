## Multiple Choice

Based on the script [`run_multiple_choice.py`]().

#### Fine-tuning on SWAG
Download [swag](https://github.com/rowanz/swagaf/tree/master/data) data

```bash
#training on 4 tesla V100(16GB) GPUS
export SWAG_DIR=/path/to/swag_data_dir
python ./examples/multiple-choice/run_multiple_choice.py \
--task_name swag \
--model_name_or_path roberta-base \
--do_train \
--do_eval \
--data_dir $SWAG_DIR \
--learning_rate 5e-5 \
--num_train_epochs 3 \
--max_seq_length 80 \
--output_dir models_bert/swag_base \
--per_gpu_eval_batch_size=16 \
--per_gpu_train_batch_size=16 \
--gradient_accumulation_steps 2 \
--overwrite_output
```
Training with the defined hyper-parameters yields the following results:
```
***** Eval results *****
eval_acc = 0.8338998300509847
eval_loss = 0.44457291918821606
```
