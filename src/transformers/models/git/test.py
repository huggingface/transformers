from transformers import pipeline

from transformers import AutoProcessor

processor = AutoProcessor.from_pretrained("microsoft/git-base-coco")

captioner = pipeline(model="microsoft/git-base-coco", tokenizer=processor.tokenizer, feature_extractor=processor.image_processor)
print(captioner("https://huggingface.co/datasets/sayakpaul/sample-datasets/resolve/main/pokemon.png"))