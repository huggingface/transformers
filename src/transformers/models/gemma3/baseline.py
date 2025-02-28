import time
from PIL import Image
import requests
import torch
from transformers import AutoProcessor, PaliGemmaForConditionalGeneration

torch.set_default_device('cpu')

MODEL_ID = "google/paligemma2-3b-mix-448"


def main(*args):
    del args
    start = time.time()
    prompt = "<image> Where is the cow standing?"
    url = "https://media.istockphoto.com/id/1192867753/photo/cow-in-berchida-beach-siniscola.jpg?s=612x612&w=0&k=20&c=v0hjjniwsMNfJSuKWZuIn8pssmD5h5bSN1peBd1CmH4="
    image = Image.open(requests.get(url, stream=True).raw)
    data_t = time.time()
    print("timecheck - data", data_t - start)

    processor = AutoProcessor.from_pretrained(MODEL_ID)
    inputs = processor(images=image, text=prompt,  return_tensors="pt")
    inputs = inputs.to(torch.get_default_device())
    processed = time.time()
    print("timecheck - processed", processed - data_t, processed - start, inputs.keys())


    model = PaliGemmaForConditionalGeneration.from_pretrained(MODEL_ID)
    model.to(torch.get_default_device())
    loaded = time.time()
    print(
        "timecheck - loaded",
        loaded - processed,
        loaded - data_t,
        loaded - start,
    )

    generate_ids = model.generate(**inputs, max_new_tokens=24)
    generated = time.time()
    print(
        "timecheck - generated",
        generated - loaded,
        generated - processed,
        generated - data_t,
        generated - start,
    )

    outputs = processor.batch_decode(
        generate_ids,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False
    )
    done = time.time()
    print(outputs)
    print(
        "timecheck - done",
        done - generated,
        done - loaded,
        done - processed,
        done - data_t,
        done - start,
    )


if __name__ == "__main__":
    main()
