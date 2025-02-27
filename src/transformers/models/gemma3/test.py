from PIL import Image
import requests
import torch
from transformers.models.gemma import GemmaTokenizer
from transformers.models.gemma3 import Gemma3ForConditionalGeneration, Gemma3Processor

torch.set_default_device('cpu')

LOCAL_MODEL = "/usr/local/google/home/ryanmullins/gemma3/gemma3_4b_pt_safetensors"


def main(*args):
    del args

    # tokenizer = GemmaTokenizer.from_pretrained(LOCAL_MODEL)
    processor = Gemma3Processor.from_pretrained(LOCAL_MODEL)
    model = Gemma3ForConditionalGeneration.from_pretrained(LOCAL_MODEL, ignore_mismatched_sizes=True)

    prompt = "<image> Where is the cow standing?"
    url = "https://media.istockphoto.com/id/1192867753/photo/cow-in-berchida-beach-siniscola.jpg?s=612x612&w=0&k=20&c=v0hjjniwsMNfJSuKWZuIn8pssmD5h5bSN1peBd1CmH4="
    image = Image.open(requests.get(url, stream=True).raw)

    inputs = processor(images=image, text=prompt,  return_tensors="pt")
    inputs = inputs.to(torch.get_default_device())

    generate_ids = model.generate(**inputs, max_new_tokens=24)
    outputs = processor.batch_decode(
        generate_ids,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False
    )
    print(outputs)


if __name__ == "__main__":
    main()
