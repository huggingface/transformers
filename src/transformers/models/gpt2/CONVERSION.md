Here is how to convert a GPT2 model generated outside of `transformers`

* [Megatron-LM](https://github.com/NVIDIA/Megatron-LM)-generated model:

Use [convert_megatron_gpt2_checkpoint.py](../megatron_gpt2/convert_megatron_gpt2_checkpoint.py)

* [big-science fork of Megatron-Deepspeed](https://github.com/bigscience-workshop/Megatron-DeepSpeed/)-generated model:

Use the instructions [here](https://github.com/bigscience-workshop/bigscience/tree/aa872e754106f6678e8a9dac8c6962404ba39a6d/train/tr1-13B-base#checkpoint-conversion-and-upload). This approach uses a set of scripts that require the use of this particular fork of Megatron-Deepspeed.
