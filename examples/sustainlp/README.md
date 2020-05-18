# SustaiNLP

Authors: [Alex Wang](https://w4ngatang.github.io/)

This folder contains instructions for training and evaluating models on SuperGLUE benchmark using `examples/run_superglue.py`.
This code is also intended as a starting point for the [SustaiNLP 2020 Shared Task](https://sites.google.com/view/sustainlp2020/shared-task).

## Using the script

### Training models

The following command will evaluate a model on the test set of a task and write the predictions to disk in the official SuperGLUE format.

```bash
python run_superglue.py --data_dir ${data_dir} --task_name ${task} \
                        --output_dir ${out_dir} --overwrite_output_dir \
                        --model_type ${model_type} --model_name_or_path ${model} ${casing_config} \
                        --use_gpuid ${gpuid} --seed ${seed} \
                        --do_train ${opt_len_train} \
                        --do_eval --eval_and_save_steps ${eval_freq} --save_only_best \
                        --learning_rate ${lr} \
                        --warmup_ratio 0.06 \
                        --weight_decay 0.01 \
                        --per_gpu_train_batch_size 4 \
                        --gradient_accumulation_steps ${grad_acc_steps} \
                        --logging_steps 100 \
```

### Evaluating models

The following command will evaluate a model on the test set of a task and write the predictions to disk in the official SuperGLUE format.

```bash
python run_superglue.py --data_dir ${data_dir} --task_name ${task} \
                        --output_dir ${out_dir} \
                        --model_type ${model_type} --model_name_or_path ${model} ${casing_config} \
                        --use_gpuid ${gpuid} --seed ${seed} \
                        --do_eval --evaluate_test --skip_evaluate_dev
                        --log_energy_consumption
```

## Extending this script

- formatting tasks: WiC; COPA
- span classification architecture for new models: WiC, WSC
