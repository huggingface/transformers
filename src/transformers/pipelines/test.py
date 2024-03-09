from transformers import pipeline


pipe = pipeline(task="image-text-to-text", model="microsoft/udop-large", processor="microsoft/udop-large")

outputs = pipe(
    images="https://huggingface.co/datasets/hf-internal-testing/fixtures_docvqa/raw/main/document_2.png",
    text="This is a document about",
)

print(outputs)
