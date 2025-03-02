import time
from PIL import Image
import requests
import torch
from transformers.models.gemma import GemmaTokenizer
from transformers.models.gemma3 import Gemma3ForConditionalGeneration, Gemma3Processor

torch.set_default_device('cpu')

LOCAL_MODEL = "/usr/local/google/home/ryanmullins/gemma3/gemma3_4b_it_safetensors"


def main(*args):
    del args
    start = time.time()
    prompt = "<image> Where is the cow standing?"
    url = "https://media.istockphoto.com/id/1192867753/photo/cow-in-berchida-beach-siniscola.jpg?s=612x612&w=0&k=20&c=v0hjjniwsMNfJSuKWZuIn8pssmD5h5bSN1peBd1CmH4="
    image = Image.open(requests.get(url, stream=True).raw)
    data_t = time.time()
    print("timecheck - data", data_t - start)

    # tokenizer = GemmaTokenizer.from_pretrained(LOCAL_MODEL)
    processor = Gemma3Processor.from_pretrained(LOCAL_MODEL)
    processor_ready = time.time()
    print("timecheck - processor ready", processor_ready - start, processor.tokenizer.is_fast)

    inputs = processor(images=image, text=prompt,  return_tensors="pt")
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

    # inputs_embeds = model.get_input_embeddings()(inputs.input_ids)
    # image_mask = inputs.image_soft_token_mask.unsqueeze(-1)
    # image_mask = image_mask.expand_as(inputs_embeds).to(inputs_embeds.device)

    # print(
    #     "embs and masks shapes, ",
    #     inputs_embeds.shape,
    #     inputs.image_soft_token_mask.shape,
    #     image_mask.shape,
    # )

    # pixel_values = inputs.pixel_values[0].to(model.device, model.dtype)
    # print("pre-vision encode", type(pixel_values), pixel_values.shape, pixel_values.dtype)
    # vision_outputs = model.vision_model(pixel_values=pixel_values).last_hidden_state
    # print("vision_outputs", vision_outputs.shape)

    # b, n, l = vision_outputs.shape
    # kernel = vision_outputs.shape[1] // 256
    # avg_pool = torch.nn.AvgPool1d(kernel_size=kernel, stride=kernel)

    # reshaped_vision_outputs = vision_outputs.permute(0, 2, 1)
    # reshaped_vision_outputs = reshaped_vision_outputs.contiguous()
    # reshaped_vision_outputs = reshaped_vision_outputs.view(b, l, n)
    # print("reshaped_vision_outputs", reshaped_vision_outputs.shape)

    # pooled_vision_outputs = avg_pool(reshaped_vision_outputs)
    # pooled_vision_outputs = pooled_vision_outputs.permute(0, 2, 1)
    # print("pooled_vision_outputs", pooled_vision_outputs.shape)

    # image_features = model.encode_vision(pooled_vision_outputs)
    # vision_encoded = time.time()
    # print(
    #     "timecheck - vision encoded",
    #     vision_encoded - start,
    #     image_features.shape,
    # )

    # inputs_embeds = inputs_embeds.masked_scatter(image_mask, image_features)
    # print(inputs_embeds.shape)

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
