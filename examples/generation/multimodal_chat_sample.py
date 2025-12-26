import torch

from transformers import DynamicCache, Qwen3VLForConditionalGeneration, Qwen3VLProcessor, TextStreamer


def first_diff_nd(a: torch.Tensor, b: torch.Tensor):
    """
    Returns a tuple of indices (i0, i1, ..., ik) for the first element
    where a != b in row-major order. Returns None if no difference.
    """
    if a.shape[1] != b.shape[1]:
        len = min(a.shape[1], b.shape[1])
        a = a[:, :len]
        b = b[:, :len]

    mismatch = a != b
    ids = mismatch.nonzero(as_tuple=False)
    if ids.numel() == 0:
        return a.shape
    # nonzero() returns coordinates sorted in row-major order, so ids[0] is the first
    return tuple(int(x.item()) for x in ids[0])


def trim_cache(past_key_values, diverged_idx):
    for keys, values, _sliding_window_tensor in past_key_values:
        if keys is None:
            continue
        keys = keys[:, : diverged_idx[1], :]
        values = values[:, : diverged_idx[1], :]


def main():
    model = Qwen3VLForConditionalGeneration.from_pretrained(
        "Qwen/Qwen3-VL-2B-Instruct", dtype="auto", device_map="auto"
    )
    processor = Qwen3VLProcessor.from_pretrained("Qwen/Qwen3-VL-2B-Instruct")
    past_key_values = DynamicCache(config=model.config)
    input_ids_in_cache = torch.tensor([[]], dtype=torch.int64)
    messages = [
        {
            "role": "user",
            "content": [
                {
                    "type": "image",
                    "image": "https://qianwen-res.oss-cn-beijing.aliyuncs.com/Qwen-VL/assets/demo.jpeg",
                },
                {"type": "text", "text": "Describe this image in 5 words."},
            ],
        }
    ]
    for prompt in None, "What's the dog's color?":
        if prompt:
            messages.append({"role": "user", "content": [{"type": "text", "text": prompt}]})
        inputs = processor.apply_chat_template(
            messages, tokenize=True, add_generation_prompt=True, return_dict=True, return_tensors="pt"
        )
        diverged_idx = first_diff_nd(inputs["input_ids"], input_ids_in_cache)
        trim_cache(past_key_values, diverged_idx)
        input_length = inputs["input_ids"].shape[1]
        if prompt:
            del inputs["pixel_values"]
            del inputs["image_grid_thw"]
        outputs = model.generate(
            **inputs,
            streamer=TextStreamer(processor.tokenizer, skip_prompt=True),
            max_new_tokens=10,
            past_key_values=past_key_values,
        )
        input_ids_in_cache = outputs
        completion = processor.batch_decode(
            outputs[0, input_length:], skip_special_tokens=True, clean_up_tokenization_spaces=False
        )[0]
        messages.append({"role": "assistant", "content": [{"type": "text", "text": completion}]})


if __name__ == "__main__":
    main()
