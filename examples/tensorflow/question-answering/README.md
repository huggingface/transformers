<!---
Copyright 2020 The HuggingFace Team. All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
-->

## SQuAD with the Tensorflow Trainer

```bash
python run_tf_squad.py \
    --model_name_or_path bert-base-uncased \
    --output_dir model \
    --max_seq_length 384 \
    --num_train_epochs 2 \
    --per_gpu_train_batch_size 8 \
    --per_gpu_eval_batch_size 16 \
    --do_train \
    --logging_dir logs \    
    --logging_steps 10 \
    --learning_rate 3e-5 \
    --doc_stride 128    
```

For the moment evaluation is not available in the Tensorflow Trainer only the training.
