## Language generation

Based on the script [`run_generation.py`](https://github.com/huggingface/transformers/blob/master/examples/text-generation/run_generation.py).

Conditional text generation using the auto-regressive models of the library: GPT, GPT-2, Transformer-XL, XLNet, CTRL.
A similar script is used for our official demo [Write With Transfomer](https://transformer.huggingface.co), where you
can try out the different models available in the library.

Example usage:

```bash
python run_generation.py \
    --model_type=gpt2 \
    --model_name_or_path=gpt2
```
