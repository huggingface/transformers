<!---
Copyright 2023 The HuggingFace Team. All rights reserved.

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

# Automatic Speech Recognition - Flax Examples

## Sequence to Sequence

The script [`run_flax_speech_recognition_seq2seq.py`](https://github.com/huggingface/transformers/blob/main/examples/flax/speech-recognition/run_flax_speech_recognition_seq2seq.py) 
can be used to fine-tune any [Flax Speech Sequence-to-Sequence Model](https://huggingface.co/docs/transformers/main/en/model_doc/auto#transformers.FlaxAutoModelForSpeechSeq2Seq) 
for automatic speech recognition on one of the [official speech recognition datasets](https://huggingface.co/datasets?task_ids=task_ids:automatic-speech-recognition) 
or a custom dataset. This includes the Whisper model from OpenAI, or a warm-started Speech-Encoder-Decoder Model, 
an example for which is included below.

### Whisper Model

We can load all components of the Whisper model directly from the pretrained checkpoint, including the pretrained model 
weights, feature extractor and tokenizer. We simply have to specify the id of fine-tuning dataset and the necessary
training hyperparameters.

The following example shows how to fine-tune the [Whisper small](https://huggingface.co/openai/whisper-small) checkpoint 
on the Hindi subset of the [Common Voice 13](https://huggingface.co/datasets/mozilla-foundation/common_voice_13_0) dataset.
Note that before running this script you must accept the dataset's [terms of use](https://huggingface.co/datasets/mozilla-foundation/common_voice_13_0) 
and register your Hugging Face Hub token on your device by running `huggingface-hub login`.

```bash
python run_flax_speech_recognition_seq2seq.py \
	--model_name_or_path="openai/whisper-small" \
	--dataset_name="mozilla-foundation/common_voice_13_0" \
	--dataset_config_name="hi" \
	--language="hindi" \
	--train_split_name="train+validation" \
	--eval_split_name="test" \
	--output_dir="./whisper-small-hi-flax" \
	--per_device_train_batch_size="16" \
	--per_device_eval_batch_size="16" \
	--num_train_epochs="10" \
	--learning_rate="1e-4" \
	--warmup_steps="500" \
	--logging_steps="25" \
	--generation_max_length="40" \
	--preprocessing_num_workers="32" \
	--dataloader_num_workers="32" \
	--max_duration_in_seconds="30" \
	--text_column_name="sentence" \
	--overwrite_output_dir \
	--do_train \
	--do_eval \
	--predict_with_generate \
	--push_to_hub \
	--use_auth_token
```

On a TPU v4-8, training should take approximately 25 minutes, with a final cross-entropy loss of 0.02 and word error 
rate of **34%**. See the checkpoint [sanchit-gandhi/whisper-small-hi-flax](https://huggingface.co/sanchit-gandhi/whisper-small-hi-flax)
for an example training run.
