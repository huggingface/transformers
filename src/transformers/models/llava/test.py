from transformers import pipeline


# OK:
# model_id = "microsoft/git-base-coco"
model_id = "Salesforce/blip-image-captioning-base"
# model_id = "Salesforce/blip2-opt-2.7b" ok, although it doesn't include the text prompt in the output
# model_id = "Salesforce/instructblip-flan-t5-xl" ok, although it doesn't include the text prompt in the output
# model_id = "llava-hf/llava-1.5-7b-hf"
# model_id = "adept/fuyu-8b"
# model_id = "google/pix2struct-textcaps-base"
# model_id = "microsoft/udop-large"
# model_id = "naver-clova-ix/donut-base-finetuned-docvqa"
# model_id = "microsoft/kosmos-2-patch14-224"

pipe = pipeline(task="image-text-to-text", model=model_id)

outputs = pipe(
    images=["http://images.cocodataset.org/val2017/000000039769.jpg"],
    # text="USER: <image>\nWhat does the label 15 represent? (1) lava (2) core (3) tunnel (4) ash cloud\nASSISTANT:",
    text=["A photo of", "The cats are"],
    max_new_tokens=200,
)

print(outputs)
