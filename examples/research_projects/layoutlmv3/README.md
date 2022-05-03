<!---
Copyright 2022 The HuggingFace Team. All rights reserved.

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

# Token classification with LayoutLMv3

## PyTorch version

Fine-tuning LayoutLMv3 for token classification on [FUNSD](https://guillaumejaume.github.io/FUNSD/) can be done as follows:

```bash
python run_funsd_cord.py \
  --model_name_or_path microsoft/layoutlmv3-base-uncased \
  --dataset_name funsd \
  --output_dir layoutlmv3-test \
  --do_train \
  --do_eval \
  --max_steps 1000 \
  --evaluation_strategy steps \
  --eval_steps 100
  --learning_rate 1e-5 \
  --load_best_model_at_end \
  --metric_for_best_model "eval_f1" \
  --push_to_hub \
  --push_to_hub°model_id layoutlmv3-finetuned-funsd
```

👀 The resulting model can be found here: https://huggingface.co/nielsr/layoutlmv3-finetuned-funsd. Note that a model card gets generated automatically in case you specify the `push_to_hub` flag.

The main script `run_funsd_cord.py` leverages the 🤗 Datasets library and the Trainer API. You can easily customize it to your needs.

Fine-tuning LayoutLMv3 for token classification on [CORD](https://github.com/clovaai/cord) can be done as follows:

```bash
python run_funsd_cord.py \
  --model_name_or_path microsoft/layoutlmv3-base-uncased \
  --dataset_name cord \
  --output_dir layoutlmv3-test \
  --do_train \
  --do_eval \
  --max_steps 1000 \
  --evaluation_strategy steps \
  --eval_steps 100
  --learning_rate 5e-5 \
  --load_best_model_at_end \
  --metric_for_best_model "eval_f1" \
  --push_to_hub \
  --push_to_hub°model_id layoutlmv3-finetuned-cord
```

👀 The resulting model can be found here: https://huggingface.co/nielsr/layoutlmv3-finetuned-cord. Note that a model card gets generated automatically in case you specify the `push_to_hub` flag.