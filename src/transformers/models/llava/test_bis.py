from transformers import pipeline


captioner = pipeline(model="Salesforce/blip-image-captioning-base")

result = captioner("https://huggingface.co/datasets/Narsil/image_dummy/resolve/main/parrots.png")

print(result)

## [{'generated_text': 'two birds are standing next to each other '}]
