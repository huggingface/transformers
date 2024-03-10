from transformers import pipeline


# OK:
# model_id = "microsoft/git-base-coco"
# model_id = "Salesforce/blip-image-captioning-base"
# model_id = "Salesforce/blip2-opt-2.7b" ok, although it doesn't include the text prompt in the output
# model_id = "Salesforce/instructblip-flan-t5-xl" ok, although it doesn't include the text prompt in the output
# model_id = "llava-hf/llava-1.5-7b-hf"
model_id = "adept/fuyu-8b"
# model_id = "google/pix2struct-textcaps-base"
# model_id = "microsoft/udop-large"
# model_id = "naver-clova-ix/donut-base-finetuned-docvqa"
# model_id = "microsoft/kosmos-2-patch14-224"

pipe = pipeline(
    task="image-text-to-text", model=model_id, processor=model_id
)  # TODO should work without explicitly passing processor

outputs = pipe(
    images="http://images.cocodataset.org/val2017/000000039769.jpg",
    text="A photo of",
    generate_kwargs=dict(max_new_tokens=30),  # TODO why can't we pass max_new_tokens directly
)

print(outputs)
