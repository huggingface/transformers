from transformers import pipeline


# OK:
# model_id = "microsoft/git-base-coco"
# model_id = "Salesforce/blip-image-captioning-base"
# model_id = "Salesforce/blip2-opt-2.7b" ok, although it doesn't include the text prompt in the output
# model_id = "Salesforce/instructblip-flan-t5-xl" ok, although it doesn't include the text prompt in the output
# model_id = "llava-hf/llava-1.5-7b-hf"
# model_id = "adept/fuyu-8b"
# model_id = "google/pix2struct-textcaps-base"
model_id = "microsoft/udop-large"

# not OK:
# model_id = microsoft/kosmos-2-patch14-224
# model_id = "naver-clova-ix/donut-base-finetuned-rvlcdip"

pipe = pipeline(
    task="image-text-to-text", model=model_id, processor=model_id
)  # TODO should work without explicitly passing processor

outputs = pipe(
    images="https://datasets-server.huggingface.co/assets/nielsr/funsd-layoutlmv3/--/funsd/train/0/image/image.jpg?Expires=1709991136&Signature=utc3TG54pIi9M1sF4SBjqbrveU7gKJWTpTUFzs8hq0h1sS3Kne7U8CCKJqixcplVuw7HsF9pyxrgfZwnsztXFoybMKPadV0XDUX8rGS3xOcb~BNtyafWLcA24HT14HFkUTsJVqAiSSmNef1vy6qmWlh7BRVn7i6AFBkFiwf-~02g6TPeF84VdMctReQq~gPImOrA~UcSxRIwXnebkOK7Q2jX1HOJ9Aa6Hi-6atSpRgNyoPdSNMZu1QHyXQqLVh5gze0BUu2TmSmCFViOwicy7n4uJKjdvvaw4uksi96OvREzcd1gm-sWOehSzS1EsSVlR0rQOQu54IWB-HDM2Qz-Ww__&Key-Pair-Id=K3EI6M078Z3AC3",
    text="Document image classification.",
    generate_kwargs=dict(max_new_tokens=30),  # TODO why can't we pass max_new_tokens directly
)

print(outputs)
