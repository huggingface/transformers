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

# Automatic Speech Recognition examples


## Connectionist Temporal Classification without Language Model (CTC w/o LM)

The script [`run_speech_recognition_ctc.py`](https://github.com/huggingface/transformers/blob/master/examples/pytorch/speech-recognition/run_speech_recognition_ctc.py) can be used to fine-tune any pretrained [Connectionist Temporal Classification Model](https://huggingface.co/transformers/master/model_doc/auto.html?highlight=automodelforctc#automodelforctc) for automatic speech 
recognition on one of the [official speech recognition datasets](https://huggingface.co/datasets?task_ids=task_ids:automatic-speech-recognition) or a custom dataset.

Speech recognition models that have been pretrained in unsupervised fashion on audio data alone, *e.g.* [Wav2Vec2](https://huggingface.co/transformers/master/model_doc/wav2vec2.html), [HuBERT](https://huggingface.co/transformers/master/model_doc/hubert.html), [XLSR-Wav2Vec2](https://huggingface.co/transformers/master/model_doc/xlsr_wav2vec2.html), have shown to require only 
very little annotated data to yield good performance on automatic speech recognition datasets.

In the script [`run_speech_recognition_ctc`], we first create a vocabulary from all unique characters of both the training data and evaluation data. Then, we preprocesses the speech recognition dataset, which includes correct resampling, normalization and padding. Finally, the pretrained speech recognition model is fine-tuned on the annotated speech recognition datasets using CTC loss.

---
**NOTE**

If you wish to use multi-processing for data preprocessing by setting `--preprocessing_num_workers` > 1, 
please make sure to set the environment variable `OMP_NUM_THREADS` to 1 as follows:

```bash
OMP_NUM_THREADS=1 python run_speech_recognition_ctc ...
```

If the environment variable is not set, the training script might freeze, *i.e.* see: https://github.com/pytorch/audio/issues/1021#issuecomment-726915239

---

### Single-GPU

The following command shows how to fine-tune [XLSR-Wav2Vec2](https://huggingface.co/transformers/master/model_doc/xlsr_wav2vec2.html) on [Common Voice](https://huggingface.co/datasets/common_voice) using a single GPU in half-precision.

```bash
python run_speech_recognition_ctc.py \
	--dataset_name="common_voice" \
	--model_name_or_path="facebook/wav2vec2-large-xlsr-53" \
	--dataset_config_name="tr" \
	--output_dir="./wav2vec2-common_voice-tr-demo" \
	--overwrite_output_dir \
	--num_train_epochs="15" \
	--per_device_train_batch_size="16" \
	--gradient_accumulation_steps="2" \
	--learning_rate="3e-4" \
	--warmup_steps="500" \
	--evaluation_strategy="steps" \
	--audio_column_name="path" \
	--text_column_name="sentence" \
	--save_steps="400" \
	--eval_steps="100" \
	--layerdrop="0.0" \
	--save_total_limit="3" \
	--freeze_feature_extractor \
	--gradient_checkpointing \
	--chars_to_ignore , ? . ! - \; \: \" “ % ‘ ” � \
	--fp16 \
	--group_by_length \
	--push_to_hub \
	--do_train --do_eval 
```

On a single V100 GPU, this script should run in *ca.* 1 hour 20 minutes and yield a CTC loss of **0.39** and word error rate
of **0.35**.

### Multi-GPU

The following command shows how to fine-tune [XLSR-Wav2Vec2](https://huggingface.co/transformers/master/model_doc/xlsr_wav2vec2.html) on [Common Voice](https://huggingface.co/datasets/common_voice) using 8 GPUs in half-precision.

```bash
OMP_NUM_THREADS=1 python -m torch.distributed.launch \
	--nproc_per_node 8 run_speech_recognition_ctc.py \
	--dataset_name="common_voice" \
	--model_name_or_path="facebook/wav2vec2-large-xlsr-53" \
	--dataset_config_name="tr" \
	--output_dir="./wav2vec2-common_voice-tr-demo-dist" \
	--preprocessing_num_workers="16" \
	--overwrite_output_dir \
	--num_train_epochs="15" \
	--per_device_train_batch_size="4" \
	--learning_rate="3e-4" \
	--warmup_steps="500" \
	--evaluation_strategy="steps" \
	--audio_column_name="path" \
	--text_column_name="sentence" \
	--save_steps="400" \
	--eval_steps="100" \
	--logging_steps="1" \
	--layerdrop="0.0" \
	--save_total_limit="3" \
	--freeze_feature_extractor \
	--gradient_checkpointing \
	--chars_to_ignore , ? . ! - \; \: \" “ % ‘ ” � \
	--fp16 \
	--group_by_length \
	--push_to_hub \
	--do_train --do_eval
```

On 8 V100 GPUs, this script should run in *ca.* 18 minutes and yield a CTC loss of **0.39** and word error rate
of **0.36**.

### Examples

In the following a couple of demonstration fine-tuning runs are listed.
It has been verified that the script works for the following datasets:

- [Common Voice](https://huggingface.co/datasets/common_voice)
- [Librispeech](https://huggingface.co/datasets/librispeech_asr)

| Dataset | Dataset Config | Pretrained Model | Word error rate on eval | GPU setup | Training time | Fine-tuned Model & Logs |
|-------|------------------------------|-------------|---------------|---------------|----------------------|-------------|
| [Librispeech](https://huggingface.co/datasets/librispeech_asr)| `"clean"` - `"train.100"` |  [facebook/wav2vec2-large-lv60](https://huggingface.co/facebook/wav2vec2-large-lv60) | 0.042 | 8 GPU V100 | 1h30min  | [here](https://huggingface.co/patrickvonplaten/wav2vec2-librispeech-clean-100h-demo-dist) |
| [Librispeech](https://huggingface.co/datasets/librispeech_asr)| `"clean"` - `"train.100"` |  [facebook/hubert-large-ll60k](https://huggingface.co/facebook/hubert-large-ll60k) | 0.088 | 8 GPU V100 | 1h30min  | [here](https://huggingface.co/patrickvonplaten/hubert-librispeech-clean-100h-demo-dist) |
| [Common Voice](https://huggingface.co/datasets/common_voice)| `"tr"`  | [facebook/wav2vec2-large-xlsr-53](https://huggingface.co/facebook/wav2vec2-large-xlsr-53)  | 0.36     | 8 GPU V100   |  18min                 | [here](https://huggingface.co/patrickvonplaten/wav2vec2-common_voice-tr-demo-dist)      |  
| [Common Voice](https://huggingface.co/datasets/common_voice)| `"tr"`  | [facebook/wav2vec2-large-xlsr-53](https://huggingface.co/facebook/wav2vec2-large-xlsr-53) | 0.35 | 1 GPU V100   |  1h20min                      | [here](https://huggingface.co/patrickvonplaten/wav2vec2-common_voice-tr-demo)  | 
