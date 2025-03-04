import time
from PIL import Image
import requests
import torch
from transformers.models.gemma import GemmaTokenizer
from transformers.models.gemma3 import (
    Gemma3ForCausalLM,
    Gemma3ForConditionalGeneration,
    Gemma3Processor,
)

torch.set_default_device('cpu')

LOCAL_MODEL = "/usr/local/google/home/ryanmullins/gemma3/gemma3_12b_it_safetensors"


def main(*args):
    del args
    start = time.time()

    prompt = "<image> Where is the cow standing?"
    prompt2 = "What is in this image? <image>"
    url = "https://media.istockphoto.com/id/1192867753/photo/cow-in-berchida-beach-siniscola.jpg?s=612x612&w=0&k=20&c=v0hjjniwsMNfJSuKWZuIn8pssmD5h5bSN1peBd1CmH4="
    image = Image.open(requests.get(url, stream=True).raw)
    data_t = time.time()
    print("timecheck - data", data_t - start)

    # tokenizer = GemmaTokenizer.from_pretrained(LOCAL_MODEL)
    processor = Gemma3Processor.from_pretrained(LOCAL_MODEL)
    processor_ready = time.time()
    print("timecheck - processor ready", processor_ready - start, processor.tokenizer.is_fast)

    inputs = processor(
        # Text-only input
        images=None,
        text="Write me a poem about Machine Learning.",
        # Single image per-prompt inputs
        # images=[image, image],
        # text=[prompt, prompt2],
        return_tensors="pt",
        padding=True,
        padding_side="left",
    )
    inputs = inputs.to(torch.get_default_device())
    processed = time.time()
    print("timecheck - processed", processed - start, inputs.keys())

    model = Gemma3ForConditionalGeneration.from_pretrained(
        LOCAL_MODEL,
        attn_implementation="sdpa",
        torch_dtype=torch.bfloat16,
    )
    print("model dtypes", model.dtype, model.language_model.dtype, model.vision_model.dtype)
    model.to(torch.get_default_device())
    loaded = time.time()
    print("timecheck - loaded", loaded - start)

    generate_ids = model.generate(**inputs, max_new_tokens=24)
    generated = time.time()
    print("timecheck - generated", generated - start)

    outputs = processor.batch_decode(
        generate_ids,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False
    )
    print(outputs)
    done = time.time()
    print("timecheck - done", done - start)


if __name__ == "__main__":
    main()
