<!---
Copyright 2021 The HuggingFace Team. All rights reserved.

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

# Speech Recognition Pre-Training


## Wav2Vec2 Speech Pre-Training

The script [`run_speech_wav2vec2_pretraining_no_trainer.py`](https://github.com/huggingface/transformers/blob/master/examples/pytorch/speech-pretraining/run_wav2vec2_pretraining_no_trainer.py) can be used to pre-train a [Wav2Vec2](https://huggingface.co/transformers/model_doc/wav2vec2.html?highlight=wav2vec2) model from scratch.

In the script [`run_speech_wav2vec2_pretraining_no_trainer`](https://github.com/huggingface/transformers/blob/master/examples/pytorch/speech-pretraining/run_wav2vec2_pretraining_no_trainer.py), a Wav2Vec2 model is pre-trained on audio data alone using [Wav2Vec2's contrastive loss objective](https://arxiv.org/abs/2006.11477).

The following examples show how to fine-tune a `"base"`-sized Wav2Vec2 model as well as a `"large"`-sized Wav2Vec2 model using [`accelerate`](https://github.com/huggingface/accelerate).

---
**NOTE**

Wav2Vec2's pre-training is known to be quite unstable. 
It is advised to do a couple of test runs with a smaller dataset,
*i.e.* `--dataset_config_names clean clean`, `--dataset_split_names validation test`
to finetune hyper-parameters, such as `learning_rate`, `batch_size`, `num_warmup_steps`,
 and optimizer settings.

---

### Base

```bash
python run_wav2vec2_pretraining_no_trainer.py \
	--dataset_name="librispeech_asr" \
	--dataset_config_name clean clean
	--dataset_train_split_names validation test
	--model_name_or_path="patrickvonplaten/wav2vec2-base-v2" \
	--output_dir="./wav2vec2-pretrained-base-demo" \
	--max_train_steps="200000" \
	--num_warmup_steps="32000" \
	--gradient_accumulation_steps="2" \
	--learning_rate="0.001" \
	--weight_decay="0.01" \
	--max_duration_in_seconds="15.0" \
	--min_duration_in_seconds="2.0" \
	--model_name_or_path="./" \
	--logging_steps="1" \
	--saving_steps="10000" \
	--per_device_train_batch_size="1" \
	--per_device_eval_batch_size="4" \
	--adam_beta1="0.9" \
	--adam_beta2="0.98" \
	--adam_epsilon="1e-06" \
```

On a single V100 GPU, this script should run in *ca.* 1 hour 20 minutes and yield a CTC loss of **0.39** and word error rate
of **0.35**.
