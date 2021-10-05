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

In the script [`run_speech_recognition_ctc`], we first create a vocabulary from all unique characters of both the training data and evaluation data. Then, we preprocesses the speech recognition dataset, which includes correct resampling, normalization and padding. Finally, the pretrained speech recognition model is fine-tuned on the annotated speech recognition datasets using CTC loss.

---
**NOTE**

If you encounter problems with data preprocessing by setting `--preprocessing_num_workers` > 1, 
you might want to set the environment variable `OMP_NUM_THREADS` to 1 as follows:

```bash
OMP_NUM_THREADS=1 python run_speech_recognition_ctc ...
```

If the environment variable is not set, the training script might freeze, *i.e.* see: https://github.com/pytorch/audio/issues/1021#issuecomment-726915239

---

### Base

```bash
python run_wav2vec2_pretraining_no_trainer.py \
	--dataset_name="librispeech_asr" \
	--dataset_config_name clean clean
	--dataset_train_split_names validation test
	--model_name_or_path="patrickvonplaten/wav2vec2-base-v2" \
	--output_dir="./wav2vec2-pretrained-demo" \
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
