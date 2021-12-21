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

If you encounter problems with data preprocessing by setting `--preprocessing_num_workers` > 1, 
you might want to set the environment variable `OMP_NUM_THREADS` to 1 as follows:

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
python -m torch.distributed.launch \
	--nproc_per_node 8 run_speech_recognition_ctc.py \
	--dataset_name="common_voice" \
	--model_name_or_path="facebook/wav2vec2-large-xlsr-53" \
	--dataset_config_name="tr" \
	--output_dir="./wav2vec2-common_voice-tr-demo-dist" \
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

The following tables present a couple of example runs on the most popular speech-recognition datasets. 
The presented performances are by no means optimal as no hyper-parameter tuning was done. Nevertheless, 
they can serve as a baseline to improve upon.


- [TIMIT](https://huggingface.co/datasets/timit_asr)

| Dataset | Dataset Config | Pretrained Model | Word error rate on eval | Phoneme error rate on eval | GPU setup | Training time | Fine-tuned Model & Logs | Command to reproduce |
|-------|------------------------------|-------------|---------------|---------------|----------------------|-------------| -------------| ------- |
| [TIMIT](https://huggingface.co/datasets/timit_asr)| -  | [wav2vec2-base](https://huggingface.co/facebook/wav2vec2-base) | 0.21 | - | 1 GPU TITAN RTX |  32min                      | [here](https://huggingface.co/patrickvonplaten/wav2vec2-base-timit-fine-tuned)  | [run.sh](https://huggingface.co/patrickvonplaten/wav2vec2-base-timit-fine-tuned/blob/main/run.sh) |
| [TIMIT](https://huggingface.co/datasets/timit_asr)| -  | [wav2vec2-base](https://huggingface.co/facebook/wav2vec2-base) | 0.21 | - | 1 GPU TITAN RTX |  32min                      | [here](https://huggingface.co/patrickvonplaten/wav2vec2-base-timit-fine-tuned)  | [run.sh](https://huggingface.co/patrickvonplaten/wav2vec2-base-timit-fine-tuned/blob/main/run.sh) |
| [TIMIT](https://huggingface.co/datasets/timit_asr)| -  | [unispeech-large-1500h-cv](https://huggingface.co/microsoft/unispeech-large-1500h-cv) | 0.22 | - | 1 GPU TITAN RTX |  35min                      | [here](https://huggingface.co/patrickvonplaten/unispeech-large-1500h-cv-timit)  | [run.sh](https://huggingface.co/patrickvonplaten/unispeech-large-1500h-cv-timit/blob/main/run.sh) |
| [TIMIT](https://huggingface.co/datasets/timit_asr)| -  | [asapp/sew-mid-100k](https://huggingface.co/asapp/sew-mid-100k) | 0.30 | - | 1 GPU TITAN RTX |  28min                      | [here](https://huggingface.co/patrickvonplaten/sew-small-100k-timit)  | [run.sh](https://huggingface.co/patrickvonplaten/sew-small-100k-timit/blob/main/run.sh) |
| [TIMIT](https://huggingface.co/datasets/timit_asr)| -  | [ntu-spml/distilhubert](https://huggingface.co/ntu-spml/distilhubert) | 0.68 | - | 1 GPU TITAN RTX |  26min                      | [here](https://huggingface.co/patrickvonplaten/distilhubert-timit)  | [run.sh](https://huggingface.co/patrickvonplaten/distilhubert-timit/blob/main/run.sh) |



- [Librispeech](https://huggingface.co/datasets/librispeech_asr)

| Dataset | Dataset Config | Pretrained Model | Word error rate on eval | Phoneme error rate on eval | GPU setup | Training time | Fine-tuned Model & Logs | Command to reproduce |
|-------|------------------------------|-------------|---------------|---------------|----------------------|-------------| -------------| ------- |
| [Librispeech](https://huggingface.co/datasets/librispeech_asr)| `"clean"` - `"train.100"` |  [microsoft/wavlm-large](https://huggingface.co/microsoft/wavlm-large) | 0.049 | - | 8 GPU V100 | 1h30min  | [here](https://huggingface.co/patrickvonplaten/wavlm-libri-clean-100h-large) | [run.sh](https://huggingface.co/patrickvonplaten/wavlm-libri-clean-100h-large/blob/main/run.sh) |
| [Librispeech](https://huggingface.co/datasets/librispeech_asr)| `"clean"` - `"train.100"` |  [microsoft/wavlm-base-plus](https://huggingface.co/microsoft/wavlm-base-plus) | 0.068 | - | 8 GPU V100 | 1h30min  | [here](https://huggingface.co/patrickvonplaten/wavlm-libri-clean-100h-base-plus) | [run.sh](https://huggingface.co/patrickvonplaten/wavlm-libri-clean-100h-base-plus/blob/main/run.sh) |
| [Librispeech](https://huggingface.co/datasets/librispeech_asr)| `"clean"` - `"train.100"` |  [facebook/wav2vec2-large-lv60](https://huggingface.co/facebook/wav2vec2-large-lv60) | 0.042 | - | 8 GPU V100 | 1h30min  | [here](https://huggingface.co/patrickvonplaten/wav2vec2-librispeech-clean-100h-demo-dist) | [run.sh](https://huggingface.co/patrickvonplaten/wav2vec2-librispeech-clean-100h-demo-dist/blob/main/run.sh) |
| [Librispeech](https://huggingface.co/datasets/librispeech_asr)| `"clean"` - `"train.100"` |  [facebook/wav2vec2-large-lv60](https://huggingface.co/facebook/wav2vec2-large-lv60) | 0.042 | - | 8 GPU V100 | 1h30min  | [here](https://huggingface.co/patrickvonplaten/wav2vec2-librispeech-clean-100h-demo-dist) | [run.sh](https://huggingface.co/patrickvonplaten/wav2vec2-librispeech-clean-100h-demo-dist/blob/main/run.sh) |
| [Librispeech](https://huggingface.co/datasets/librispeech_asr)| `"clean"` - `"train.100"` |  [facebook/hubert-large-ll60k](https://huggingface.co/facebook/hubert-large-ll60k) | 0.088 | - | 8 GPU V100 | 1h30min  | [here](https://huggingface.co/patrickvonplaten/hubert-librispeech-clean-100h-demo-dist) | [run.sh](https://huggingface.co/patrickvonplaten/hubert-librispeech-clean-100h-demo-dist/blob/main/run.sh) |
| [Librispeech](https://huggingface.co/datasets/librispeech_asr)| `"clean"` - `"train.100"` |  [asapp/sew-mid-100k](https://huggingface.co/asapp/sew-mid-100k) | 0.167 | | | 8 GPU V100 | 54min  | [here](https://huggingface.co/patrickvonplaten/sew-mid-100k-librispeech-clean-100h-ft) | [run.sh](https://huggingface.co/patrickvonplaten/sew-mid-100k-librispeech-clean-100h-ft/blob/main/run.sh) |

- [Common Voice](https://huggingface.co/datasets/common_voice)

| Dataset | Dataset Config | Pretrained Model | Word error rate on eval | Phoneme error rate on eval | GPU setup | Training time | Fine-tuned Model & Logs | Command to reproduce |
|-------|------------------------------|-------------|---------------|---------------|----------------------|-------------| -------------| ------- |
| [Common Voice](https://huggingface.co/datasets/mozilla-foundation/common_voice_3_0)| `"tr"`  | [facebook/wav2vec2-large-xls-r-300m](https://huggingface.co/facebook/wav2vec2-xls-r-300m)  | - |  0.099   | 8 GPU V100   |  23min                 | [here](https://huggingface.co/patrickvonplaten/xls-r-300m-tr-phoneme)      |  [run.sh](https://huggingface.co/patrickvonplaten/xls-r-300m-tr-phoneme/blob/main/run.sh) |
| [Common Voice](https://huggingface.co/datasets/mozilla-foundation/common_voice_3_0)| `"it"`  | [facebook/wav2vec2-large-xls-r-300m](https://huggingface.co/facebook/wav2vec2-xls-r-300m)  | - |  0.077   | 8 GPU V100   |  23min                 | [here](https://huggingface.co/patrickvonplaten/xls-r-300m-it-phoneme)      |  [run.sh](https://huggingface.co/patrickvonplaten/xls-r-300m-it-phoneme/blob/main/run.sh) |
| [Common Voice](https://huggingface.co/datasets/mozilla-foundation/common_voice_3_0)| `"sv-SE"`  | [facebook/wav2vec2-large-xls-r-300m](https://huggingface.co/facebook/wav2vec2-xls-r-300m)  | - |  0.099   | 8 GPU V100   |  23min                 | [here](https://huggingface.co/patrickvonplaten/xls-r-300m-sv-phoneme)      |  [run.sh](https://huggingface.co/patrickvonplaten/xls-r-300m-sv-phoneme/blob/main/run.sh) |
| [Common Voice](https://huggingface.co/datasets/common_voice)| `"tr"`  | [facebook/wav2vec2-large-xlsr-53](https://huggingface.co/facebook/wav2vec2-large-xlsr-53)  | 0.36 |  -      | 8 GPU V100   |  18min                 | [here](https://huggingface.co/patrickvonplaten/wav2vec2-common_voice-tr-demo-dist)      |  [run.sh](https://huggingface.co/patrickvonplaten/wav2vec2-common_voice-tr-demo-dist/blob/main/run_dist.sh) |
| [Common Voice](https://huggingface.co/datasets/common_voice)| `"tr"`  | [facebook/wav2vec2-large-xlsr-53](https://huggingface.co/facebook/wav2vec2-large-xlsr-53)  | 0.31  | -    | 8 GPU V100   |  1h05                 | [here](https://huggingface.co/patrickvonplaten/wav2vec2-large-xlsr-53-common_voice-tr-ft)      |  [run.sh](https://huggingface.co/patrickvonplaten/wav2vec2-large-xlsr-53-common_voice-tr-ft/blob/main/run.sh) |
| [Common Voice](https://huggingface.co/datasets/common_voice)| `"tr"`  | [facebook/wav2vec2-large-xlsr-53](https://huggingface.co/facebook/wav2vec2-large-xlsr-53) | 0.35 | - | 1 GPU V100   |  1h20min                      | [here](https://huggingface.co/patrickvonplaten/wav2vec2-common_voice-tr-demo)  | [run.sh](https://huggingface.co/patrickvonplaten/wav2vec2-common_voice-tr-demo/blob/main/run.sh) |
| [Common Voice](https://huggingface.co/datasets/common_voice)| `"tr"`  | [facebook/wav2vec2-xls-r-300m](https://huggingface.co/facebook/wav2vec2-xls-r-300m)  | 0.31     | - | 8 GPU V100   |  1h05            | [here](https://huggingface.co/patrickvonplaten/wav2vec2-large-xls-r-300m-common_voice-tr-ft)      |  [run.sh](https://huggingface.co/patrickvonplaten/wav2vec2-large-xls-r-300m-common_voice-tr-ft/blob/main/run.sh) |
| [Common Voice](https://huggingface.co/datasets/common_voice)| `"tr"`  | [facebook/wav2vec2-xls-r-1b](https://huggingface.co/facebook/wav2vec2-xls-r-1b)  | 0.21 | -  | 2 GPU Titan 24 GB RAM   |  15h10            | [here](https://huggingface.co/patrickvonplaten/wav2vec2-xls-r-1b-common_voice-tr-ft)      |  [run.sh](https://huggingface.co/patrickvonplaten/wav2vec2-large-xls-r-1b-common_voice-tr-ft/blob/main/run.sh) |

- [Multilingual Librispeech](https://huggingface.co/datasets/multilingual_librispeech)

| Dataset | Dataset Config | Pretrained Model | Word error rate on eval | Phoneme error rate on eval | GPU setup | Training time | Fine-tuned Model & Logs | Command to reproduce |
|-------|------------------------------|-------------|---------------|---------------|----------------------|-------------| -------------| ------- |
| [Multilingual Librispeech](https://huggingface.co/datasets/multilingual_librispeech)| `"german"`  | [facebook/wav2vec2-large-xlsr-53](https://huggingface.co/facebook/wav2vec2-large-xlsr-53)  | 0.13  | -     | 1 GPU Titan 24 GB RAM  |  15h04                 | [here](https://huggingface.co/patrickvonplaten/wav2vec2-xlsr-53-300m-mls-german-ft)      |  [run.sh](https://huggingface.co/patrickvonplaten/wav2vec2-xlsr-53-300m-mls-german-ft/blob/main/run.sh) |
| [Multilingual Librispeech](https://huggingface.co/datasets/multilingual_librispeech)| `"german"`  | [facebook/wav2vec2-xls-r-300m](https://huggingface.co/facebook/wav2vec2-xls-r-300m)  | 0.15 | -     | 1 GPU Titan 24 GB RAM  |  15h04                 | [here](https://huggingface.co/patrickvonplaten/wav2vec2-300m-mls-german-ft)      |  [run.sh](https://huggingface.co/patrickvonplaten/wav2vec2-300m-mls-german-ft/blob/main/run.sh) |
